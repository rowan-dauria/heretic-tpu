# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
Checkpoint resolution, safetensors reading, the parameter pytree, host stacking and
device placement, and the mappings from abliterable modules to safetensors keys and
transformers module paths.

Parameter pytree
----------------

Parameters are nested dicts of arrays. Linear weights keep the Hugging Face
orientation `[out, in]`, and per-layer tensors are stacked along a leading layer axis.
Shapes use L (layers), D (hidden size), V (vocabulary), H and KV (query and key/value
heads), hd (head dimension), I (intermediate size), E and I_e (experts and their
intermediate size) and rot (rotated dimensions per head). Unless stated otherwise,
leaves are in the session dtype.

| Key | Shape | Present | Hugging Face module (under the text model) |
| :-- | :-- | :-- | :-- |
| `embed` | [V, D] | always | `embed_tokens` |
| `head` | [V, D] | not `arch.tied_head` | `lm_head` (when tied, the head is `embed`) |
| `final_norm` | [D] | always | `norm` |
| `layers/attn_norm` | [L, D] | always | `input_layernorm` |
| `layers/q` | [L, H·hd, D] | always | `self_attn.q_proj` (Phi-3: rows of `qkv_proj`) |
| `layers/k`, `layers/v` | [L, KV·hd, D] | always | `self_attn.k_proj`, `v_proj` (Phi-3: rows of `qkv_proj`) |
| `layers/q_bias` | [L, H·hd] | "q" in `arch.biases` | `self_attn.q_proj.bias` |
| `layers/k_bias`, `layers/v_bias` | [L, KV·hd] | "k"/"v" in `arch.biases` | `self_attn.k_proj.bias`, `v_proj.bias` |
| `layers/q_norm`, `layers/k_norm` | [L, hd] | `arch.has_qk_norm` | `self_attn.q_norm`, `k_norm` |
| `layers/o_proj` | [L, 1, D, H·hd] | always | `self_attn.o_proj` (component "attn.o_proj") |
| `layers/o_proj_bias` | [L, 1, D] | "o_proj" in `arch.biases` | `self_attn.o_proj.bias` |
| `layers/attn_out_norm` | [L, D] | `arch.has_sandwich_norms` | `post_attention_layernorm` (Gemma 3) |
| `layers/mlp_norm` | [L, D] | always | `post_attention_layernorm` (Gemma 3: `pre_feedforward_layernorm`) |
| `layers/gate`, `layers/up` | [L, I, D] | not `arch.is_moe` | `mlp.gate_proj`, `up_proj` (Phi-3: rows of `gate_up_proj`) |
| `layers/gate_bias`, `layers/up_bias` | [L, I] | "gate"/"up" in `arch.biases` | `mlp.gate_proj.bias`, `up_proj.bias` |
| `layers/down_proj` | [L, 1, D, I] | not `arch.is_moe` | `mlp.down_proj` (component "mlp.down_proj") |
| `layers/down_proj_bias` | [L, 1, D] | "down_proj" in `arch.biases` | `mlp.down_proj.bias` |
| `layers/router` | [L, E, D] | `arch.is_moe` | `mlp.gate` (Mixtral: `block_sparse_moe.gate`) |
| `layers/expert_gate`, `layers/expert_up` | [L, E, I_e, D] | `arch.is_moe` | `mlp.experts.{e}.gate_proj`, `up_proj` (Mixtral: `block_sparse_moe.experts.{e}.w1`, `w3`; fused: rows of `mlp.experts.gate_up_proj`) |
| `layers/expert_down` | [L, E, D, I_e] | `arch.is_moe` | `mlp.experts.{e}.down_proj` (Mixtral: `w2`; fused: `mlp.experts.down_proj`) |
| `layers/mlp_out_norm` | [L, D] | `arch.has_sandwich_norms` | `post_feedforward_layernorm` (Gemma 3) |
| `buffers/inv_freq` | [L, rot/2] float32 | always | `rope.build_tables` |
| `buffers/long_inv_freq` | [L, rot/2] float32 | `arch.rope_switch == "long_factor"` | `rope.build_tables` |
| `buffers/attention_scaling` | [L] float32 | always | `rope.build_tables` |
| `buffers/window` | [L] int32 | always | `arch.windows` |

The abliterable components (`COMPONENTS`) are stored as `[L, M, d_out, d_in]` with
M = 1, so that the stored array can be handed out without a copy, and their biases as
`[L, M, d_out]`. The buffers are ordinary leaves, so they are traced jit arguments
rather than compile-time constants.
"""

from __future__ import annotations

import gc
import json
import math
import os
import threading
import warnings
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Self

import httpx
import jax
import ml_dtypes  # noqa: F401 (registers bfloat16 with NumPy, which reading relies on)
import numpy as np
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import (
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
)
from safetensors import safe_open
from transformers import AutoConfig, GenerationConfig, PretrainedConfig

from .errors import (
    DeviceMemoryError,
    UnsupportedArchitectureError,
    UnsupportedCheckpointError,
)
from .rope import build_tables

if TYPE_CHECKING:
    from .arch import ArchConfig
    from .sharding import ShardingPlan

# Nested dicts of arrays (or of jax.ShapeDtypeStruct, for a skeleton).
Params = dict[str, Any]

# Abliterable component (upstream name) -> key of its weight under "layers".
COMPONENTS: dict[str, str] = {
    "attn.o_proj": "o_proj",
    "mlp.down_proj": "down_proj",
}

CONFIG_FILE = "config.json"
GENERATION_CONFIG_FILE = "generation_config.json"
INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"

# The files resolve_checkpoint downloads.
METADATA_FILES = (CONFIG_FILE, GENERATION_CONFIG_FILE, INDEX_FILE)

# Candidate text-model prefixes. Multimodal wrappers saved by transformers
# (including transformers 5) use "language_model.model.", which transformers
# renames to "model.language_model." at load time.
TEXT_PREFIXES = ("model.", "model.language_model.", "language_model.model.")

# Storage dtypes that can be loaded as linear or embedding weights.
FLOAT_DTYPES = ("F32", "BF16", "F16")

# Key suffixes of pre-quantised checkpoints (fp8, GPTQ, AWQ, compressed-tensors).
QUANTIZATION_SUFFIXES = (
    "weight_scale",
    "weight_scale_inv",
    "qweight",
    "qzeros",
    "scales",
    "g_idx",
)

# Safetensors key suffixes (under the text prefix and "layers.{i}.") of
# the per-layer parameters other than the MLP.
_LAYER_KEYS: dict[str, str] = {
    "attn_norm": "input_layernorm.weight",
    "q": "self_attn.q_proj.weight",
    "k": "self_attn.k_proj.weight",
    "v": "self_attn.v_proj.weight",
    "q_bias": "self_attn.q_proj.bias",
    "k_bias": "self_attn.k_proj.bias",
    "v_bias": "self_attn.v_proj.bias",
    "q_norm": "self_attn.q_norm.weight",
    "k_norm": "self_attn.k_norm.weight",
    "o_proj": "self_attn.o_proj.weight",
    "o_proj_bias": "self_attn.o_proj.bias",
    "mlp_norm": "post_attention_layernorm.weight",
}

# The dense MLP.
_DENSE_MLP_KEYS: dict[str, str] = {
    "gate": "mlp.gate_proj.weight",
    "up": "mlp.up_proj.weight",
    "gate_bias": "mlp.gate_proj.bias",
    "up_bias": "mlp.up_proj.bias",
    "down_proj": "mlp.down_proj.weight",
    "down_proj_bias": "mlp.down_proj.bias",
}

# Gemma 3 normalises before and after both the attention and the MLP.
_GEMMA3_LAYER_KEYS: dict[str, str] = {
    "attn_out_norm": "post_attention_layernorm.weight",
    "mlp_norm": "pre_feedforward_layernorm.weight",
    "mlp_out_norm": "post_feedforward_layernorm.weight",
}

# Phi-3 fuses the attention input projections and the MLP gate and up projections.
_PHI3_FUSED_KEYS: dict[str, str] = {
    "q": "self_attn.qkv_proj.weight",
    "k": "self_attn.qkv_proj.weight",
    "v": "self_attn.qkv_proj.weight",
    "q_bias": "self_attn.qkv_proj.bias",
    "k_bias": "self_attn.qkv_proj.bias",
    "v_bias": "self_attn.qkv_proj.bias",
    "gate": "mlp.gate_up_proj.weight",
    "up": "mlp.gate_up_proj.weight",
    "gate_bias": "mlp.gate_up_proj.bias",
    "up_bias": "mlp.gate_up_proj.bias",
}

# Mixture-of-experts MLPs, with "{e}" standing for the expert index.
_MOE_KEYS: dict[str, dict[str, str]] = {
    "qwen3_moe": {
        "router": "mlp.gate.weight",
        "expert_gate": "mlp.experts.{e}.gate_proj.weight",
        "expert_up": "mlp.experts.{e}.up_proj.weight",
        "expert_down": "mlp.experts.{e}.down_proj.weight",
    },
    "mixtral": {
        "router": "block_sparse_moe.gate.weight",
        "expert_gate": "block_sparse_moe.experts.{e}.w1.weight",
        "expert_up": "block_sparse_moe.experts.{e}.w3.weight",
        "expert_down": "block_sparse_moe.experts.{e}.w2.weight",
    },
}

# Fused experts, as transformers 5 holds them in memory: gate_up_proj [E, 2·I_e, D]
# (gate rows, then up rows) and down_proj [E, D, I_e].
_FUSED_MOE_KEYS: dict[str, str] = {
    "router": "mlp.gate.weight",
    "expert_gate": "mlp.experts.gate_up_proj",
    "expert_up": "mlp.experts.gate_up_proj",
    "expert_down": "mlp.experts.down_proj",
}

# Parameters stored with a module axis (M = 1) after the layer axis.
_MODULE_AXIS_KEYS = ("o_proj", "o_proj_bias", "down_proj", "down_proj_bias")

# Transformers module path suffix (under the decoder layer) of each component.
_COMPONENT_MODULES: dict[str, str] = {
    "attn.o_proj": "self_attn.o_proj",
    "mlp.down_proj": "mlp.down_proj",
}


@dataclass(frozen=True)
class Checkpoint:
    """A checkpoint resolved to one commit, with its configuration but no weights."""

    # The Hub id or local directory, as given.
    model: str

    snapshot_dir: str

    # The resolved commit (None for a local directory).
    sha: str | None

    # Whether the Hub could not be reached, so that the commit was resolved, and the
    # files are taken, from the local Hugging Face cache only (False for a local
    # directory).
    offline: bool

    # Parsed config.json.
    raw_config: dict[str, Any]

    # As resolved by AutoConfig.
    config: PretrainedConfig

    # Parsed model.safetensors.index.json, if present and not stale (see _shard_set).
    index: dict[str, Any] | None

    # The shard set: the files the index maps tensors to, or model.safetensors.
    shard_files: tuple[str, ...]

    generation_config: GenerationConfig


@dataclass(frozen=True)
class TensorInfo:
    # Shard file name, relative to snapshot_dir.
    shard: str

    shape: tuple[int, ...]

    # Safetensors dtype string, e.g. "BF16" or "F8_E4M3".
    dtype: str


@dataclass(frozen=True)
class TensorIndex:
    snapshot_dir: str

    # Every key of every shard in the shard set.
    tensors: Mapping[str, TensorInfo]

    # The text-model prefix (one of TEXT_PREFIXES).
    prefix: str

    # The output projection, or None when the embedding matrix serves as the head.
    head_key: str | None

    @classmethod
    def from_infos(
        cls,
        snapshot_dir: str,
        tensors: Mapping[str, TensorInfo],
        config: PretrainedConfig,
    ) -> TensorIndex:
        """
        Detects the text prefix, rejects quantised tensors and resolves the head,
        in that order.
        """

        prefix = _detect_prefix(tensors)
        _check_storage_dtypes(tensors, prefix)
        head_key = _resolve_head_key(tensors, prefix, config)
        return cls(
            snapshot_dir=snapshot_dir,
            tensors=tensors,
            prefix=prefix,
            head_key=head_key,
        )


class Leaf(NamedTuple):
    """Shape and dtype of a parameter (dtype None: the session dtype)."""

    shape: tuple[int, ...]
    dtype: np.dtype | None = None


def resolve_checkpoint(model: str, model_commit: str | None) -> Checkpoint:
    """
    Resolves a Hub id or a local directory and downloads the metadata files
    (configuration, generation configuration and safetensors index) only.

    When the Hub cannot be reached, a Hub id is resolved from the local Hugging Face
    cache instead, as transformers does, with a warning (see `Checkpoint.offline`).
    """

    if os.path.isdir(model):
        snapshot_dir = model
        sha = None
        offline = False
        # The files of the commit on the Hub (None: those in snapshot_dir).
        repo_files: set[str] | None = None
    else:
        try:
            # Resolve the commit once, so that all files come from the same commit.
            # snapshot_download records the commit of a branch or tag in the cache,
            # where a later offline run finds it.
            snapshot_dir = snapshot_download(
                model,
                revision=model_commit,
                allow_patterns=list(METADATA_FILES),
            )
            info = HfApi().model_info(model, revision=os.path.basename(snapshot_dir))
            offline = False
            repo_files = {sibling.rfilename for sibling in info.siblings or []}
        except Exception as error:
            if not _is_hub_unreachable(error):
                raise
            snapshot_dir = _cached_snapshot(model, model_commit, error)
            offline = True
            # Only the cached files can be loaded.
            repo_files = None

        # Snapshot directories are named after their commit.
        sha = os.path.basename(snapshot_dir)

    try:
        config = AutoConfig.from_pretrained(snapshot_dir, trust_remote_code=False)
    except (KeyError, ValueError) as error:
        # Unknown model types and configs that need remote code.
        raise UnsupportedArchitectureError(
            f"The configuration of {model} cannot be loaded: {error}"
        ) from error

    with open(os.path.join(snapshot_dir, CONFIG_FILE), encoding="utf-8") as file:
        raw_config = json.load(file)

    # Weight files are fetched later with the shard set, so on the Hub they are looked
    # up in the file list of the commit.
    def has_file(name: str) -> bool:
        if repo_files is None:
            return os.path.isfile(os.path.join(snapshot_dir, name))
        return name in repo_files

    index, shard_files = _shard_set(model, snapshot_dir, offline, has_file)

    # As transformers resolves it for generate(): without generation_config.json, from
    # the raw config.json, which keeps the token ids and the legacy generation
    # parameters that the resolved config fills with class defaults or drops.
    if os.path.isfile(os.path.join(snapshot_dir, GENERATION_CONFIG_FILE)):
        generation_config = GenerationConfig.from_pretrained(snapshot_dir)
    else:
        generation_config = GenerationConfig.from_pretrained(
            snapshot_dir,
            config_file_name=CONFIG_FILE,
            _from_model_config=True,
        )

    return Checkpoint(
        model=model,
        snapshot_dir=snapshot_dir,
        sha=sha,
        offline=offline,
        raw_config=raw_config,
        config=config,
        index=index,
        shard_files=shard_files,
        generation_config=generation_config,
    )


def _shard_set(
    model: str,
    snapshot_dir: str,
    offline: bool,
    has_file: Callable[[str], bool],
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    """
    The parsed safetensors index (None without one) and the shard set. `has_file`
    tells whether the checkpoint has a weight file (when `offline`, whether the cache
    has it).

    The index is preferred over model.safetensors while every shard it names exists.
    transformers prefers model.safetensors, but save_pretrained leaves the other
    layout's files behind when it saves into a folder that holds an earlier save of
    the other layout: an unsharded save deletes the old shards but keeps the index,
    and a sharded save keeps the old model.safetensors. Missing shards therefore mark
    the index as stale, and a complete index as newer than model.safetensors.
    """

    index_path = os.path.join(snapshot_dir, INDEX_FILE)
    if not os.path.isfile(index_path):
        if has_file(SINGLE_FILE):
            return None, (SINGLE_FILE,)
        raise UnsupportedCheckpointError(
            f"{model} has neither {SINGLE_FILE} nor {INDEX_FILE}. "
            "Only safetensors checkpoints are supported."
        )

    with open(index_path, encoding="utf-8") as file:
        index = json.load(file)
    shard_files = tuple(sorted(set(index["weight_map"].values())))

    missing = [shard for shard in shard_files if not has_file(shard)]
    if not missing:
        return index, shard_files

    if has_file(SINGLE_FILE):
        count = f"{len(missing)} of the {len(shard_files)} files it names"
        if offline:
            problem = f"{count} are not cached"
        else:
            problem = f"it is stale: {count} do not exist"
        warnings.warn(
            f"{INDEX_FILE} of {model} cannot be used, as {problem} "
            f"({', '.join(missing)}). {SINGLE_FILE} is loaded instead, "
            "as transformers does.",
            stacklevel=3,
        )
        return None, (SINGLE_FILE,)

    # Shards that are merely not cached are reported by fetch_shards.
    if offline:
        return index, shard_files

    raise UnsupportedCheckpointError(
        f"{len(missing)} of the {len(shard_files)} safetensors files that {INDEX_FILE} "
        f"of {model} names do not exist ({', '.join(missing)}), and there is no "
        f"{SINGLE_FILE}."
    )


def _is_hub_unreachable(error: Exception) -> bool:
    """
    Whether an error means that the Hub cannot be reached: offline mode, a failed
    connection or a timeout (on which huggingface_hub's snapshot_download also falls
    back to the cache), a server error, or rate limiting (429) or a request timeout
    (408), which huggingface_hub also treats as transient and which the Hub's API
    returns on its own while file downloads still succeed. Errors about the repository
    or the revision (RepositoryNotFoundError, GatedRepoError, RevisionNotFoundError
    and other client errors) are not, as the cache must not hide them.
    """

    if isinstance(error, HfHubHTTPError):
        response = error.response
        return response is not None and (
            response.status_code >= 500 or response.status_code in (408, 429)
        )

    return isinstance(
        error,
        (OfflineModeIsEnabled, httpx.ConnectError, httpx.TimeoutException),
    )


def _cached_snapshot(model: str, model_commit: str | None, error: Exception) -> str:
    """
    The cached snapshot directory of the commit the cache records for `model_commit`
    (the default branch if None), for when the Hub cannot be reached (`error`).
    """

    try:
        snapshot_dir = snapshot_download(
            model,
            revision=model_commit,
            allow_patterns=list(METADATA_FILES),
            local_files_only=True,
        )
    except LocalEntryNotFoundError as cache_error:
        raise cache_error from error

    sha = os.path.basename(snapshot_dir)

    # Without a cached file listing of the commit (which transformers does not
    # write), snapshot_download returns the snapshot directory even if files are
    # missing from it.
    if not os.path.isfile(os.path.join(snapshot_dir, CONFIG_FILE)) or not any(
        os.path.isfile(os.path.join(snapshot_dir, name))
        for name in (INDEX_FILE, SINGLE_FILE)
    ):
        raise LocalEntryNotFoundError(
            f"The Hugging Face Hub cannot be reached, and the cached snapshot of "
            f"{model} at commit {sha} lacks {CONFIG_FILE}, or both "
            f"{SINGLE_FILE} and {INDEX_FILE}."
        ) from error

    # Like load_prompts, which tolerates an unreachable Hub for cached datasets.
    warnings.warn(
        f"The Hugging Face Hub cannot be reached ({error}). "
        f"{model} is loaded from the local cache at commit {sha}, "
        "which could not be checked against the Hub.",
        stacklevel=3,
    )

    return snapshot_dir


def fetch_shards(ckpt: Checkpoint) -> None:
    """
    Downloads exactly the shard set (no-op for a local directory). When the Hub could
    not be reached (`ckpt.offline`), checks that the shard set is cached instead.
    """

    if ckpt.sha is None:
        return

    if ckpt.offline:
        # snapshot_download would list the files of the commit on the Hub, unless
        # the listing is cached (which transformers does not do).
        missing = [
            shard
            for shard in ckpt.shard_files
            if not os.path.isfile(os.path.join(ckpt.snapshot_dir, shard))
        ]
        if missing:
            raise LocalEntryNotFoundError(
                f"The Hugging Face Hub cannot be reached, and {len(missing)} of the "
                f"{len(ckpt.shard_files)} safetensors files of {ckpt.model} at commit "
                f"{ckpt.sha} are not cached: {', '.join(missing)}."
            )
        return

    snapshot_download(
        ckpt.model,
        revision=ckpt.sha,
        allow_patterns=list(ckpt.shard_files),
    )


def build_tensor_index(ckpt: Checkpoint) -> TensorIndex:
    """Reads the safetensors headers of the shard set (no tensor data)."""

    tensors: dict[str, TensorInfo] = {}

    for shard in ckpt.shard_files:
        with safe_open(os.path.join(ckpt.snapshot_dir, shard), framework="np") as file:
            keys = file.keys()
            for key in keys:
                tensor = file.get_slice(key)
                tensors[key] = TensorInfo(
                    shard=shard,
                    shape=tuple(tensor.get_shape()),
                    dtype=tensor.get_dtype(),
                )

    return TensorIndex.from_infos(ckpt.snapshot_dir, tensors, ckpt.config)


def _detect_prefix(tensors: Mapping[str, TensorInfo]) -> str:
    matches = [
        prefix
        for prefix in TEXT_PREFIXES
        if f"{prefix}layers.0.self_attn.o_proj.weight" in tensors
    ]

    if len(matches) != 1:
        raise UnsupportedCheckpointError(
            "Cannot locate the text model in the checkpoint: expected exactly one of "
            f"the prefixes {', '.join(TEXT_PREFIXES)} to contain "
            f"'layers.0.self_attn.o_proj.weight', found {len(matches)}."
        )

    return matches[0]


def _head_key_candidate(prefix: str) -> str:
    if prefix == "language_model.model.":
        return "language_model.lm_head.weight"
    return "lm_head.weight"


def _check_storage_dtypes(tensors: Mapping[str, TensorInfo], prefix: str) -> None:
    head_key = _head_key_candidate(prefix)

    for key, info in tensors.items():
        if not key.startswith(prefix) and key != head_key:
            continue

        if key.endswith(QUANTIZATION_SUFFIXES):
            raise UnsupportedCheckpointError(
                f"The checkpoint is pre-quantised (it contains '{key}'), and quantised "
                "checkpoints are not supported. "
                "Please use the unquantised repository of the model instead."
            )

        # Linear and embedding weights are the tensors with at least two dimensions.
        if len(info.shape) >= 2 and info.dtype not in FLOAT_DTYPES:
            raise UnsupportedCheckpointError(
                f"Tensor '{key}' is stored as {info.dtype}, but only "
                f"{', '.join(FLOAT_DTYPES)} weights are supported. "
                "Please use the unquantised repository of the model instead."
            )


def _resolve_head_key(
    tensors: Mapping[str, TensorInfo],
    prefix: str,
    config: PretrainedConfig,
) -> str | None:
    head_key = _head_key_candidate(prefix)

    # A head tensor in the checkpoint is used whatever the tie flags say.
    # Transformers refuses to tie two differing tensors, and Mistral3Config
    # defaults to tied embeddings although Mistral-Small-3.1 ships a distinct head.
    if head_key in tensors:
        return head_key

    if getattr(config, "tie_word_embeddings", False):
        return None

    raise UnsupportedCheckpointError(
        f"The checkpoint has no output projection ('{head_key}') and does not tie "
        "it to the embedding, so transformers would initialise a random head."
    )


def detect_biases(tensors: TensorIndex, model_type: str) -> tuple[str, ...]:
    """
    Returns the per-layer projections that have a bias (sorted parameter keys,
    see `ArchConfig.biases`), judged by the keys of the first layer.
    """

    keys = _layer_keys(model_type)
    layer_prefix = f"{tensors.prefix}layers.0."

    return tuple(
        sorted(
            name.removesuffix("_bias")
            for name, suffix in keys.items()
            if name.endswith("_bias") and layer_prefix + suffix in tensors.tensors
        )
    )


def _layer_keys(model_type: str) -> dict[str, str]:
    keys = dict(_LAYER_KEYS)
    if model_type not in _MOE_KEYS:
        keys.update(_DENSE_MLP_KEYS)
    if model_type == "gemma3_text":
        keys.update(_GEMMA3_LAYER_KEYS)
    elif model_type == "phi3":
        keys.update(_PHI3_FUSED_KEYS)
    return keys


def abliterable_components(arch: ArchConfig) -> list[str]:
    """The abliterable components of the architecture, sorted."""

    # Mixture-of-experts layers have no dense MLP to abliterate.
    if arch.is_moe:
        return ["attn.o_proj"]
    return sorted(COMPONENTS)


def component_shape(arch: ArchConfig, component: str) -> tuple[int, int]:
    """(d_out, d_in) of the modules of a component."""

    _check_component(arch, component)

    if component == "attn.o_proj":
        return arch.hidden_size, arch.num_attention_heads * arch.head_dim
    return arch.hidden_size, arch.intermediate_size


def _check_component(arch: ArchConfig, component: str) -> None:
    if component not in abliterable_components(arch):
        raise ValueError(f"Unknown component: {component}")


def param_layout(arch: ArchConfig) -> dict[str, Leaf]:
    """
    Every leaf of the parameter pytree, keyed by its "/"-separated path
    (see the module docstring).
    """

    layers = arch.num_hidden_layers
    hidden = arch.hidden_size
    q_size = arch.num_attention_heads * arch.head_dim
    kv_size = arch.num_key_value_heads * arch.head_dim

    layout = {"embed": Leaf((arch.vocab_size, hidden))}
    if not arch.tied_head:
        layout["head"] = Leaf((arch.vocab_size, hidden))
    layout["final_norm"] = Leaf((hidden,))

    layer: dict[str, tuple[int, ...]] = {
        "attn_norm": (hidden,),
        "q": (q_size, hidden),
        "k": (kv_size, hidden),
        "v": (kv_size, hidden),
    }
    for name, size in (("q", q_size), ("k", kv_size), ("v", kv_size)):
        if name in arch.biases:
            layer[f"{name}_bias"] = (size,)
    if arch.has_qk_norm:
        layer["q_norm"] = (arch.head_dim,)
        layer["k_norm"] = (arch.head_dim,)
    layer["o_proj"] = (1, hidden, q_size)
    if "o_proj" in arch.biases:
        layer["o_proj_bias"] = (1, hidden)
    if arch.has_sandwich_norms:
        layer["attn_out_norm"] = (hidden,)

    layer["mlp_norm"] = (hidden,)
    if arch.is_moe:
        experts = arch.num_experts
        expert_size = arch.moe_intermediate_size
        layer["router"] = (experts, hidden)
        layer["expert_gate"] = (experts, expert_size, hidden)
        layer["expert_up"] = (experts, expert_size, hidden)
        layer["expert_down"] = (experts, hidden, expert_size)
    else:
        layer["gate"] = (arch.intermediate_size, hidden)
        layer["up"] = (arch.intermediate_size, hidden)
        for name in ("gate", "up"):
            if name in arch.biases:
                layer[f"{name}_bias"] = (arch.intermediate_size,)
        layer["down_proj"] = (1, hidden, arch.intermediate_size)
        if "down_proj" in arch.biases:
            layer["down_proj_bias"] = (1, hidden)
    if arch.has_sandwich_norms:
        layer["mlp_out_norm"] = (hidden,)

    for name, shape in layer.items():
        layout[f"layers/{name}"] = Leaf((layers, *shape))

    rope_size = arch.rot // 2
    float32 = np.dtype(np.float32)
    layout["buffers/inv_freq"] = Leaf((layers, rope_size), float32)
    if arch.rope_switch == "long_factor":
        layout["buffers/long_inv_freq"] = Leaf((layers, rope_size), float32)
    layout["buffers/attention_scaling"] = Leaf((layers,), float32)
    layout["buffers/window"] = Leaf((layers,), np.dtype(np.int32))

    return layout


def unflatten(flat: Mapping[str, Any]) -> Params:
    """Turns a mapping from "/"-separated paths into nested dicts."""

    nested: Params = {}
    for path, value in flat.items():
        *parents, name = path.split("/")
        node = nested
        for parent in parents:
            node = node.setdefault(parent, {})
        node[name] = value
    return nested


def param_skeleton(
    arch: ArchConfig,
    dtype: Any,
    plan: ShardingPlan | None = None,
) -> Params:
    """
    The parameter pytree as jax.ShapeDtypeStruct leaves, for ahead-of-time lowering.
    With a plan, the leaves carry the shardings the parameters are placed with.
    """

    dtype = np.dtype(dtype)
    return unflatten(
        {
            path: jax.ShapeDtypeStruct(
                leaf.shape,
                leaf.dtype or dtype,
                sharding=None if plan is None else plan.param_sharding(path),
            )
            for path, leaf in param_layout(arch).items()
        }
    )


def device_footprint(arch: ArchConfig, dtype: Any, plan: ShardingPlan) -> int:
    """Bytes the placed parameters occupy on each device of the plan."""

    dtype = np.dtype(dtype)
    footprint = 0

    for path, leaf in param_layout(arch).items():
        size = math.prod(leaf.shape) * (leaf.dtype or dtype).itemsize
        if plan.is_sharded(path):
            size //= plan.model_axis_size
        footprint += size

    return footprint


def check_device_memory(arch: ArchConfig, dtype: Any, plan: ShardingPlan) -> None:
    """
    Raises DeviceMemoryError if the parameters would not fit into the memory of the
    plan's devices. Skipped for devices that report no memory statistics (CPU).
    """

    footprint = device_footprint(arch, dtype, plan)

    for device in plan.devices:
        stats = device.memory_stats()
        if stats is None:
            continue

        bytes_limit = stats["bytes_limit"]
        if footprint > bytes_limit:
            raise DeviceMemoryError(
                f"The model needs {footprint / 2**30:.2f} GiB per device in "
                f"{np.dtype(dtype).name}, but {device} has only "
                f"{bytes_limit / 2**30:.2f} GiB."
            )


# The tensors a parameter is assembled from, as (safetensors key, index into the
# parameter, stored shape).
_DiskTensors = list[tuple[str, tuple[int, ...], tuple[int, ...]]]


class _Source(NamedTuple):
    # Safetensors key, with "{i}" standing for the layer index and "{e}" for the
    # expert index (for a parameter stored as one tensor per expert).
    key: str

    # Range of the output axis (the second to last, or the only one) of a fused
    # tensor to read, or None for the whole tensor.
    rows: tuple[int, int] | None = None

    # Size of that axis in the fused tensor.
    fused_size: int | None = None


def _sources(arch: ArchConfig, tensors: TensorIndex) -> dict[str, _Source]:
    prefix = tensors.prefix
    sources = {
        "embed": _Source(f"{prefix}embed_tokens.weight"),
        "final_norm": _Source(f"{prefix}norm.weight"),
    }
    if tensors.head_key is not None:
        sources["head"] = _Source(tensors.head_key)

    q_size = arch.num_attention_heads * arch.head_dim
    kv_size = arch.num_key_value_heads * arch.head_dim
    intermediate_size = arch.intermediate_size
    qkv_size = q_size + 2 * kv_size
    fused_rows = {
        "q": ((0, q_size), qkv_size),
        "k": ((q_size, q_size + kv_size), qkv_size),
        "v": ((q_size + kv_size, qkv_size), qkv_size),
        "gate": ((0, intermediate_size), 2 * intermediate_size),
        "up": ((intermediate_size, 2 * intermediate_size), 2 * intermediate_size),
    }

    for name, suffix in _layer_keys(arch.model_type).items():
        key = f"{prefix}layers.{{i}}.{suffix}"
        if arch.model_type == "phi3" and name in _PHI3_FUSED_KEYS:
            rows, fused_size = fused_rows[name.removesuffix("_bias")]
            sources[f"layers/{name}"] = _Source(key, rows, fused_size)
        else:
            sources[f"layers/{name}"] = _Source(key)

    if arch.is_moe:
        expert_size = arch.moe_intermediate_size
        fused = f"{prefix}layers.0.{_FUSED_MOE_KEYS['expert_gate']}" in tensors.tensors
        moe_keys = _FUSED_MOE_KEYS if fused else _MOE_KEYS[arch.model_type]
        for name, suffix in moe_keys.items():
            key = f"{prefix}layers.{{i}}.{suffix}"
            if fused and name == "expert_gate":
                sources[f"layers/{name}"] = _Source(
                    key, (0, expert_size), 2 * expert_size
                )
            elif fused and name == "expert_up":
                sources[f"layers/{name}"] = _Source(
                    key, (expert_size, 2 * expert_size), 2 * expert_size
                )
            else:
                sources[f"layers/{name}"] = _Source(key)

    return sources


def _disk_tensors(
    arch: ArchConfig,
    path: str,
    leaf: Leaf,
    source: _Source,
) -> _DiskTensors:
    """The tensors a parameter is assembled from."""

    if not path.startswith("layers/"):
        return [(source.key, (), leaf.shape)]

    name = path.removeprefix("layers/")
    per_expert = "{e}" in source.key
    shape = (
        leaf.shape[2:] if per_expert or name in _MODULE_AXIS_KEYS else leaf.shape[1:]
    )
    if source.fused_size is not None:
        axis = max(len(shape) - 2, 0)
        shape = (*shape[:axis], source.fused_size, *shape[axis + 1 :])

    entries = []
    for i in range(arch.num_hidden_layers):
        if per_expert:
            for e in range(arch.num_experts):
                entries.append((source.key.format(i=i, e=e), (i, e), shape))
        elif name in _MODULE_AXIS_KEYS:
            entries.append((source.key.format(i=i), (i, 0), shape))
        else:
            entries.append((source.key.format(i=i), (i,), shape))
    return entries


def _check_tensors(
    tensors: TensorIndex,
    disk_tensors: dict[str, _DiskTensors],
) -> None:
    """Checks that every parameter exists with the expected shape, before any reading."""

    for entries in disk_tensors.values():
        for key, _, shape in entries:
            info = tensors.tensors.get(key)
            if info is None:
                raise UnsupportedCheckpointError(f"Tensor '{key}' is missing.")
            if info.shape != shape:
                raise UnsupportedCheckpointError(
                    f"Tensor '{key}' has shape {info.shape}, but {shape} was expected."
                )


def _host_buffers(arch: ArchConfig) -> dict[str, np.ndarray]:
    tables = build_tables(arch)
    buffers = {
        "buffers/inv_freq": tables.inv_freq,
        "buffers/attention_scaling": tables.attention_scaling,
        "buffers/window": np.array(arch.windows, dtype=np.int32),
    }
    if tables.long_inv_freq is not None:
        buffers["buffers/long_inv_freq"] = tables.long_inv_freq
    return buffers


class _ShardReader:
    """Reads tensors from lazily opened safetensors files."""

    def __init__(self, snapshot_dir: str, tensors: TensorIndex, stack: ExitStack):
        self.snapshot_dir = snapshot_dir
        self.tensors = tensors
        self.stack = stack
        self.files: dict[str, Any] = {}

    def read(self, key: str, rows: tuple[int, int] | None) -> np.ndarray:
        shard = self.tensors.tensors[key].shard
        if shard not in self.files:
            self.files[shard] = self.stack.enter_context(
                safe_open(os.path.join(self.snapshot_dir, shard), framework="np")
            )
        file = self.files[shard]

        if rows is None:
            return file.get_tensor(key)

        # The output axis is the second to last, or the only one.
        tensor = file.get_slice(key)
        start, end = rows
        if len(tensor.get_shape()) == 3:
            return tensor[:, start:end]
        return tensor[start:end]


def _read_leaf(
    reader: _ShardReader,
    path: str,
    leaf: Leaf,
    source: _Source,
    entries: _DiskTensors,
    dtype: np.dtype,
) -> np.ndarray:
    if not path.startswith("layers/"):
        ((key, _, _),) = entries
        return reader.read(key, source.rows).astype(dtype, copy=False)

    # One host buffer per stacked parameter, filled layer by layer (and expert by
    # expert), casting to the session dtype on the way.
    buffer = np.empty(leaf.shape, dtype=dtype)
    for key, index, _ in entries:
        np.copyto(buffer[index], reader.read(key, source.rows), casting="unsafe")
    return buffer


class _Prefetcher:
    """
    Reads the shard files sequentially on a background thread, which fills the page
    cache at the disk's sequential speed. Each stacked parameter reads one tensor per
    layer, spread over the files, which on its own reaches only part of that speed.
    """

    # Bytes per read, between which the thread checks whether to stop.
    CHUNK_SIZE = 16 * 2**20

    def __init__(self, paths: list[str]):
        self.paths = paths
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        buffer = bytearray(self.CHUNK_SIZE)
        for path in self.paths:
            with open(path, "rb", buffering=0) as file:
                while not self.stopped.is_set() and file.readinto(buffer):
                    pass

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stopped.set()
        if self.thread.is_alive():
            self.thread.join()


def _place(host: np.ndarray, sharding: jax.sharding.Sharding) -> jax.Array:
    array = jax.device_put(host, sharding)
    # Wait for the transfer, so that the host buffer is released before the next
    # one is filled.
    array.block_until_ready()
    return array


def load_params(
    ckpt: Checkpoint,
    tensors: TensorIndex,
    arch: ArchConfig,
    dtype: Any,
    plan: ShardingPlan,
) -> Params:
    """
    Reads the parameters in the session dtype and places them as the plan says.

    Each stacked parameter is assembled in one host buffer and placed with a single
    device_put, so no per-layer or unsharded device arrays are created. Raises
    DeviceMemoryError before reading any tensor data if the parameters would not fit.
    """

    dtype = np.dtype(dtype)
    layout = param_layout(arch)
    sources = _sources(arch, tensors)
    host_buffers = _host_buffers(arch)

    disk_tensors = {
        path: _disk_tensors(arch, path, leaf, sources[path])
        for path, leaf in layout.items()
        if path not in host_buffers
    }
    _check_tensors(tensors, disk_tensors)
    check_device_memory(arch, dtype, plan)

    # Only the files that hold text-model tensors are prefetched.
    shard_paths = sorted(
        {
            os.path.join(ckpt.snapshot_dir, tensors.tensors[key].shard)
            for entries in disk_tensors.values()
            for key, _, _ in entries
        }
    )

    placed = {}

    with ExitStack() as stack:
        stack.enter_context(_Prefetcher(shard_paths))
        reader = _ShardReader(ckpt.snapshot_dir, tensors, stack)

        for path, leaf in layout.items():
            if path in host_buffers:
                host = host_buffers.pop(path)
            else:
                host = _read_leaf(
                    reader, path, leaf, sources[path], disk_tensors[path], dtype
                )
            placed[path] = _place(host, plan.param_sharding(path))
            del host

            # jaxlib drops its reference to a transferred host buffer on a runtime
            # thread and defers the decref until Python next calls into jaxlib or
            # collects garbage. Collecting the young generation frees the buffer now,
            # before the next one is allocated.
            gc.collect(0)

    return unflatten(placed)


def disk_key(
    tensors: TensorIndex,
    component: str,
    layer_index: int,
    module_index: int,
) -> tuple[str, TensorInfo]:
    """
    The safetensors key of a module's weight, with its shard and storage dtype.
    Used by the merged export.
    """

    key = None
    if component in COMPONENTS and module_index == 0:
        key = f"{tensors.prefix}layers.{layer_index}.{_COMPONENT_MODULES[component]}.weight"
    if key not in tensors.tensors:
        raise ValueError(f"Unknown module: {component}[{layer_index}, {module_index}]")

    return key, tensors.tensors[key]


def module_path(
    arch: ArchConfig,
    component: str,
    layer_index: int,
    module_index: int,
) -> str:
    """
    The path of a module in the transformers model upstream loads, as named_modules()
    gives it (and as PEFT adapter keys use it).

    Never derived from the disk key: multimodal checkpoints store
    "language_model.model.layers.*", which transformers renames at load time.
    """

    _check_component(arch, component)
    if module_index != 0:
        raise ValueError(f"Unknown module: {component}[{layer_index}, {module_index}]")

    # Upstream loads multimodal checkpoints with AutoModelForImageTextToText.
    root = "model.language_model.layers" if arch.multimodal else "model.layers"
    return f"{root}.{layer_index}.{_COMPONENT_MODULES[component]}"
