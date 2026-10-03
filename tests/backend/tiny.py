# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
Tiny random-weight checkpoints for the backend tests.

Checkpoints are built from the resolved configurations of real models at pinned
commits (only config.json is downloaded, and cached by huggingface_hub), shrunk to a
few narrow layers, instantiated with transformers 5 and written with
`save_pretrained`. The files on disk are therefore exactly what transformers writes,
including the legacy "language_model.model.*" layout of the gemma3 and mistral3
wrappers. Building a checkpoint needs torch; tests that use this module should start
with `pytest.importorskip("torch")`.

Import with `from tests.backend.tiny import ...`.

API
---

`REAL_MODELS: dict[str, str]`
    Case name -> Hub id of the real model whose configuration a case is built from:
    "llama", "mistral", "qwen2", "qwen3", "gemma3_text", "gemma3" (multimodal
    wrapper), "mistral3" (multimodal wrapper), "ministral3" (Ministral 3 in the
    mistral3 wrapper: yarn RoPE and Llama 4 query scaling), "phi3" (shaped like
    Phi-4-mini: partial rotary, longrope, tied), "phi3_4k" (Phi-3-mini-4k: sliding
    window), "qwen3_moe" and "mixtral" (mixtures of experts).

`REVISIONS: dict[str, str]`
    Hub id -> the commit that the functions below read the model at, so that its
    configuration and tensor index cannot drift apart, and the tests do not change
    when the repository does. Every model they are called with must be pinned here.

`real_config(repo_id, **text_overrides) -> PretrainedConfig`
    The configuration of a Hub model as AutoConfig resolves it, optionally with
    `text_overrides` applied to the text config dict before resolution (for example
    a different `rope_parameters`).

`real_raw_config(repo_id) -> dict`
    The parsed config.json of a Hub model.

`real_tensor_index(repo_id) -> TensorIndex`
    The tensor index of a Hub model, built from the safetensors headers. The headers
    are fetched once with HTTP range requests (no weights are downloaded) and cached
    on disk under huggingface_hub's assets cache (`HF_ASSETS_CACHE`), so that, like
    the configurations, they need the Hub only on the first run (raising a
    RuntimeError if it cannot be reached then).

`real_arch(repo_id, **text_overrides) -> ArchConfig`
    The ArchConfig of a Hub model at full size, built from
    `real_config(repo_id, **text_overrides)`, its config.json and `real_tensor_index`.

`tiny_config(repo_id, *, num_hidden_layers=3, hidden_size=64, intermediate_size=128,
num_attention_heads=4, num_key_value_heads=2, vocab_size=256, num_experts=4,
**text_overrides) -> PretrainedConfig`
    The real configuration with the text model shrunk (head_dim becomes
    hidden_size / num_attention_heads where the config has the field; MoE models get
    `num_experts` experts of width intermediate_size / 4, two per token). Everything
    else (RoPE parameters, windows, activations, soft-capping, tie flags) is kept,
    except that `layer_types` is rebuilt for the new depth (alternating sliding and
    full attention, starting with sliding, when the real model mixes them), longrope
    factor lists are subsampled to rot/2 entries, and token ids that do not fit into a
    reduced vocabulary are replaced by 0 (pad), 1 (bos) and 2 (eos).
    `vocab_size=None` keeps the real vocabulary (and token ids), for use with the real
    tokenizer. Vision towers are shrunk too. `text_overrides` are applied to the text
    config dict last, for example `sliding_window=8` or
    `rope_parameters={"rope_type": "yarn", ...}`.

`model_class(config_or_dict) -> type`
    The Auto class upstream heretic loads a checkpoint with:
    AutoModelForImageTextToText when the config has "vision_config", else
    AutoModelForCausalLM.

`build_checkpoint(config, directory, *, dtype=None, seed=0,
tie_word_embeddings=None, max_shard_size=None, edit_config=None) -> Path`
    Instantiates the model with random weights (every parameter is redrawn so that
    activations stay of order one: weights N(0, 1/fan_in), norm weights 1 + N(0, 0.1),
    biases N(0, 0.1)), casts it to the torch `dtype` (float32 by default) and saves it
    into `directory` (created).
    `tie_word_embeddings` overrides the top-level tie flag before instantiation
    (False gives a separate, saved lm_head). `max_shard_size` (e.g. "100KB") gives a
    sharded checkpoint with model.safetensors.index.json. `edit_config(dict)` edits
    the saved config.json afterwards (for example to set or remove tie flags or to add
    a quantization_config).

`load_reference(directory, dtype=None) -> PreTrainedModel`
    The transformers model of a checkpoint, loaded with upstream's class choice and
    `attn_implementation="eager"` in the torch `dtype` (float32 by default), in eval
    mode.

`text_model(model) -> torch.nn.Module`
    The decoder (`.layers`, `.embed_tokens`, `.norm`) of a reference model.

`add_stray_files(directory) -> list[str]`
    Adds files that must never be downloaded, opened or exported:
    consolidated.safetensors, params.json and original/x.pth. Returns their names.

`FakeHub(root, repos, shas=None)`
    Serves local directories (`repos`: Hub id -> directory) as Hub repositories,
    downloading into snapshot directories under `root`, for code that calls
    `HfApi().model_info` and `snapshot_download`, and records every downloaded file
    in `downloads` (a list of (repo_id, filename)) and the revision every
    snapshot_download call asks for in `revisions` (a list of (repo_id, revision)).
    `install(monkeypatch, *modules)` replaces the `HfApi` and `snapshot_download`
    names in the given modules (by default heretic_tpu.backend.weights). Each
    repository has a single commit: `shas[repo_id]` if given, otherwise
    `FakeHub.SHA`, which the revisions None and "main" resolve to.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from functools import cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
from huggingface_hub import constants
from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled
from huggingface_hub.utils import filter_repo_objects
from transformers import AutoConfig, PretrainedConfig

from heretic_tpu.backend.arch import ArchConfig
from heretic_tpu.backend.weights import TensorIndex, TensorInfo

REAL_MODELS: dict[str, str] = {
    "llama": "unsloth/Llama-3.2-1B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3": "Qwen/Qwen3-0.6B",
    "gemma3_text": "unsloth/gemma-3-1b-it",
    "gemma3": "unsloth/gemma-3-4b-it",
    "mistral3": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "ministral3": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
    "phi3": "microsoft/Phi-4-mini-instruct",
    "phi3_4k": "microsoft/Phi-3-mini-4k-instruct",
    "qwen3_moe": "Qwen/Qwen3-30B-A3B",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

REVISIONS: dict[str, str] = {
    "unsloth/Llama-3.2-1B-Instruct": "5a8abab4a5d6f164389b1079fb721cfab8d7126c",
    "mistralai/Mistral-7B-Instruct-v0.3": "c170c708c41dac9275d15a8fff4eca08d52bab71",
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen2.5-32B-Instruct": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-4B-Instruct-2507": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "Qwen/Qwen3-30B-A3B": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
    "unsloth/gemma-3-1b-it": "5b11413a10db4e486ef16a20101fd028f8f2499c",
    "unsloth/gemma-3-4b-it": "bf46152c47f5dd20b896357cb51abc4c03b8ee8c",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": (
        "68faf511d618ef198fef186659617cfd2eb8e33a"
    ),
    "mistralai/Ministral-3-3B-Instruct-2512-BF16": (
        "b6d637bef2393152b3da2b2fde72eecdee30557e"
    ),
    "microsoft/Phi-3-mini-4k-instruct": "f39ac1d28e925b323eae81227eaba4464caced4e",
    "microsoft/Phi-3.5-mini-instruct": "2fe192450127e6a83f7441aef6e3ca586c338b77",
    "microsoft/Phi-4-mini-instruct": "cfbefacb99257ffa30c83adab238a50856ac3083",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "eba92302a2861cdc0098cc54bc9f17cb2c47eb61",
}

# Small replacements for token ids that do not fit into a reduced vocabulary.
_TOKEN_IDS = {"pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2}


def _revision(repo_id: str) -> str:
    try:
        return REVISIONS[repo_id]
    except KeyError:
        raise KeyError(
            f"Pin a commit of {repo_id} in tests.backend.tiny.REVISIONS."
        ) from None


@cache
def _real_config_dict(repo_id: str) -> dict[str, Any]:
    return AutoConfig.from_pretrained(repo_id, revision=_revision(repo_id)).to_dict()


def real_config(repo_id: str, **text_overrides: Any) -> PretrainedConfig:
    config = json.loads(json.dumps(_real_config_dict(repo_id)))
    (config.get("text_config") or config).update(text_overrides)
    return AutoConfig.for_model(**_without_model_type(config))


def _without_model_type(config: dict[str, Any]) -> dict[str, Any]:
    # AutoConfig.for_model takes the model type as its first argument.
    config = dict(config)
    return {"model_type": config.pop("model_type"), **config}


@cache
def real_raw_config(repo_id: str) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, "config.json", revision=_revision(repo_id))
    return json.loads(Path(path).read_text())


@cache
def real_tensor_index(repo_id: str) -> TensorIndex:
    from huggingface_hub import cached_assets_path

    revision = _revision(repo_id)
    # The assets directory is read when called, so that tests can redirect it.
    path = cached_assets_path(
        "heretic-tpu-tests",
        namespace=repo_id,
        subfolder="tensor-index",
        assets_dir=constants.HF_ASSETS_CACHE,
    ) / (revision + ".json")

    try:
        # Key -> [shard, shape, dtype].
        entries = json.loads(path.read_text())
    except FileNotFoundError:
        entries = _fetch_tensor_entries(repo_id, revision, path)
        # Atomically, as several test processes may share the cache.
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(entries))
        os.replace(temporary, path)

    tensors = {
        key: TensorInfo(shard=shard, shape=tuple(shape), dtype=dtype)
        for key, (shard, shape, dtype) in entries.items()
    }
    return TensorIndex.from_infos(repo_id, tensors, real_config(repo_id))


def _fetch_tensor_entries(repo_id: str, revision: str, path: Path) -> dict[str, Any]:
    from huggingface_hub import get_safetensors_metadata

    try:
        metadata = get_safetensors_metadata(repo_id, revision=revision)
    except (OfflineModeIsEnabled, httpx.HTTPError, HfHubHTTPError) as error:
        raise RuntimeError(
            f"The tensor index of {repo_id} at commit {revision} is not cached in "
            f"{path}, and reading the safetensors headers from the Hugging Face Hub "
            f"failed ({type(error).__name__}: {error}). Run the tests once with "
            "access to the Hub to cache it."
        ) from error

    return {
        key: [shard, list(info.shape), info.dtype]
        for shard, shard_metadata in metadata.files_metadata.items()
        for key, info in shard_metadata.tensors.items()
    }


def real_arch(repo_id: str, **text_overrides: Any) -> ArchConfig:
    return ArchConfig.from_hf(
        real_config(repo_id, **text_overrides),
        real_raw_config(repo_id),
        real_tensor_index(repo_id),
    )


def tiny_config(
    repo_id: str,
    *,
    num_hidden_layers: int = 3,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    vocab_size: int | None = 256,
    num_experts: int = 4,
    **text_overrides: Any,
) -> PretrainedConfig:
    config = json.loads(json.dumps(_real_config_dict(repo_id)))
    text = config.get("text_config") or config

    text["num_hidden_layers"] = num_hidden_layers
    text["hidden_size"] = hidden_size
    text["intermediate_size"] = intermediate_size
    text["num_attention_heads"] = num_attention_heads
    text["num_key_value_heads"] = num_key_value_heads
    if text.get("head_dim") is not None:
        text["head_dim"] = hidden_size // num_attention_heads

    if text["model_type"] in ("qwen3_moe", "mixtral"):
        # Configs may serialise the expert count under both names.
        for key in ("num_experts", "num_local_experts"):
            if key in text:
                text[key] = num_experts
        # Mixtral's experts are as wide as intermediate_size.
        if text["model_type"] == "qwen3_moe":
            text["moe_intermediate_size"] = intermediate_size // 4
        else:
            text["intermediate_size"] = intermediate_size // 4
        text["num_experts_per_tok"] = 2

    if vocab_size is not None:
        text["vocab_size"] = vocab_size
        for section in (config, text):
            for key, replacement in _TOKEN_IDS.items():
                if key in section:
                    section[key] = _fit_token_id(section[key], vocab_size, replacement)

    layer_types = text.get("layer_types")
    if layer_types is not None:
        if len(set(layer_types)) > 1:
            text["layer_types"] = [
                "sliding_attention" if i % 2 == 0 else "full_attention"
                for i in range(num_hidden_layers)
            ]
        else:
            text["layer_types"] = layer_types[:1] * num_hidden_layers

    rope_parameters = text.get("rope_parameters") or {}
    if rope_parameters.get("rope_type") == "longrope":
        head_dim = hidden_size // num_attention_heads
        rot = int(head_dim * rope_parameters.get("partial_rotary_factor", 1.0))
        for key in ("short_factor", "long_factor"):
            factors = rope_parameters[key]
            step = len(factors) // (rot // 2)
            rope_parameters[key] = factors[::step][: rot // 2]

    vision = config.get("vision_config")
    if vision is not None:
        vision.update(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=28,
            patch_size=14,
        )
        if "head_dim" in vision:
            vision["head_dim"] = 16
        if "mm_tokens_per_image" in config:
            # One image token per patch.
            config["mm_tokens_per_image"] = 4

    text.update(text_overrides)

    return AutoConfig.for_model(**_without_model_type(config))


def _fit_token_id(token_id: Any, vocab_size: int, replacement: int) -> Any:
    if isinstance(token_id, int) and token_id >= vocab_size:
        return replacement
    if isinstance(token_id, list):
        fitted = [_fit_token_id(item, vocab_size, replacement) for item in token_id]
        return list(dict.fromkeys(fitted))
    return token_id


def model_class(config: PretrainedConfig | dict[str, Any]) -> type:
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    if not isinstance(config, dict):
        config = config.to_dict()
    if "vision_config" in config:
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def build_checkpoint(
    config: PretrainedConfig,
    directory: str | Path,
    *,
    dtype: Any = None,
    seed: int = 0,
    tie_word_embeddings: bool | None = None,
    max_shard_size: str | None = None,
    edit_config: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # A copy, so that the caller's config is left unchanged.
    config = AutoConfig.for_model(**_without_model_type(config.to_dict()))
    if tie_word_embeddings is not None:
        config.tie_word_embeddings = tie_word_embeddings

    torch.manual_seed(seed)
    model = model_class(config).from_config(config, dtype=torch.float32)
    _randomise(model, seed)
    model.to(dtype or torch.float32)

    kwargs = {} if max_shard_size is None else {"max_shard_size": max_shard_size}
    model.save_pretrained(directory, **kwargs)

    if edit_config is not None:
        config_path = directory / "config.json"
        saved = json.loads(config_path.read_text())
        edit_config(saved)
        config_path.write_text(json.dumps(saved, indent=2))

    return directory


def _randomise(model: Any, seed: int) -> None:
    import torch

    generator = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            values = torch.randn(parameter.shape, generator=generator)
            if parameter.ndim >= 2:
                values *= parameter.shape[-1] ** -0.5
            elif "norm" in name:
                values = 1 + 0.1 * values
            else:
                values *= 0.1
            parameter.copy_(values)


def load_reference(directory: str | Path, dtype: Any = None) -> Any:
    import torch

    config = json.loads((Path(directory) / "config.json").read_text())
    model = model_class(config).from_pretrained(
        directory,
        dtype=dtype or torch.float32,
        attn_implementation="eager",
    )
    return model.eval()


def text_model(model: Any) -> Any:
    inner = model.model
    return getattr(inner, "language_model", inner)


def add_stray_files(directory: str | Path) -> list[str]:
    import torch
    from safetensors.torch import save_file

    directory = Path(directory)
    save_file(
        {"output.weight": torch.zeros(2, 2)},
        directory / "consolidated.safetensors",
    )
    (directory / "params.json").write_text(json.dumps({"dim": 2}))
    (directory / "original").mkdir(exist_ok=True)
    torch.save({"x": torch.zeros(1)}, directory / "original" / "x.pth")
    return ["consolidated.safetensors", "params.json", "original/x.pth"]


class FakeHub:
    SHA = "0123456789abcdef0123456789abcdef01234567"

    def __init__(
        self,
        root: str | Path,
        repos: dict[str, str | Path],
        shas: dict[str, str] | None = None,
    ):
        self.root = Path(root)
        self.repos = {repo_id: Path(path) for repo_id, path in repos.items()}
        self.shas = shas or {}
        self.downloads: list[tuple[str, str]] = []
        self.revisions: list[tuple[str, str | None]] = []

    def sha(self, repo_id: str) -> str:
        return self.shas.get(repo_id, self.SHA)

    def _resolve(self, repo_id: str, revision: str | None) -> str:
        if revision not in (None, "main", self.sha(repo_id)):
            raise ValueError(f"Unknown revision: {revision}")
        return self.sha(repo_id)

    def _files(self, repo_id: str) -> list[str]:
        repo = self.repos[repo_id]
        return sorted(
            path.relative_to(repo).as_posix()
            for path in repo.rglob("*")
            if path.is_file()
        )

    def model_info(self, repo_id: str, revision: str | None = None, **kwargs: Any):
        return SimpleNamespace(
            sha=self._resolve(repo_id, revision),
            siblings=[SimpleNamespace(rfilename=name) for name in self._files(repo_id)],
        )

    def snapshot_download(
        self,
        repo_id: str,
        revision: str | None = None,
        allow_patterns: list[str] | str | None = None,
        **kwargs: Any,
    ) -> str:
        self.revisions.append((repo_id, revision))
        # Like the Hub cache, the snapshot directory is named after the commit.
        snapshot = (
            self.root / repo_id.replace("/", "--") / self._resolve(repo_id, revision)
        )
        snapshot.mkdir(parents=True, exist_ok=True)

        for name in filter_repo_objects(
            self._files(repo_id),
            allow_patterns=allow_patterns,
        ):
            target = snapshot / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.repos[repo_id] / name, target)
                self.downloads.append((repo_id, name))

        return str(snapshot)

    def install(self, monkeypatch: Any, *modules: ModuleType) -> None:
        if not modules:
            from heretic_tpu.backend import weights

            modules = (weights,)

        for module in modules:
            monkeypatch.setattr(module, "HfApi", lambda *args, **kwargs: self)
            monkeypatch.setattr(module, "snapshot_download", self.snapshot_download)
