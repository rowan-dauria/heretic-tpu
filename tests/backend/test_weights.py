# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Checkpoint resolution, tensor index, loading and name mappings on tiny checkpoints."""

import json
import math
import os
import re
import shutil
import threading
import time
import tracemalloc
from contextlib import ExitStack, suppress
from pathlib import Path
from types import SimpleNamespace

import httpx
import jax
import ml_dtypes
import numpy as np
import pytest
from huggingface_hub import constants as hf_constants
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from heretic_tpu.backend import arch as arch_module
from heretic_tpu.backend import weights
from heretic_tpu.backend.arch import ArchConfig, check_config
from heretic_tpu.backend.errors import (
    DeviceMemoryError,
    UnsupportedCheckpointError,
)
from heretic_tpu.backend.rope import build_tables
from heretic_tpu.backend.sharding import choose_plan
from tests.backend.tiny import (
    REAL_MODELS,
    FakeHub,
    add_stray_files,
    build_checkpoint,
    load_reference,
    model_class,
    text_model,
    tiny_config,
)

torch = pytest.importorskip("torch")


def _build(name: str, directory: Path, **kwargs) -> Path:
    """Builds one of the checkpoint cases into a directory."""

    if name == "llama-biases":
        config = tiny_config(REAL_MODELS["llama"], attention_bias=True, mlp_bias=True)
        return build_checkpoint(config, directory, tie_word_embeddings=False, **kwargs)
    if name == "mistral3-untied":
        # An untied head and no top-level tie key (Mistral3Config defaults to tying).
        return build_checkpoint(
            tiny_config(REAL_MODELS["mistral3"]),
            directory,
            tie_word_embeddings=False,
            edit_config=lambda config: config.pop("tie_word_embeddings"),
            **kwargs,
        )
    if name == "qwen3-head-and-tie-flag":
        # Like Qwen3-0.6B: the tie flag is set, but lm_head.weight is shipped.
        return build_checkpoint(
            tiny_config(REAL_MODELS["qwen3"]),
            directory,
            tie_word_embeddings=False,
            edit_config=lambda config: config.update(tie_word_embeddings=True),
            **kwargs,
        )
    if name == "phi3-sharded":
        return build_checkpoint(
            tiny_config(REAL_MODELS["phi3"]),
            directory,
            max_shard_size="40KB",
            **kwargs,
        )
    if name == "qwen3_moe-fused":
        # Fused experts, as transformers 5 holds them in memory.
        from safetensors.torch import save_file

        build_checkpoint(tiny_config(REAL_MODELS["qwen3_moe"]), directory, **kwargs)
        state = load_reference(directory).state_dict()
        assert "model.layers.0.mlp.experts.gate_up_proj" in state
        save_file(
            {key: value.contiguous() for key, value in state.items()},
            str(directory / "model.safetensors"),
            metadata={"format": "pt"},
        )
        return directory
    return build_checkpoint(tiny_config(REAL_MODELS[name]), directory, **kwargs)


CASES = [
    "llama",
    "llama-biases",
    "mistral",
    "qwen2",
    "qwen3",
    "qwen3-head-and-tie-flag",
    "gemma3_text",
    "gemma3",
    "mistral3",
    "mistral3-untied",
    "phi3",
    "phi3_4k",
    "phi3-sharded",
    # Planned architectures (enabled by the fixture below).
    "qwen3_moe",
    "qwen3_moe-fused",
    "mixtral",
]


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("checkpoints")
    return {name: _build(name, root / name) for name in CASES}


@pytest.fixture
def planned_moe(monkeypatch) -> None:
    """Accepts the planned mixture-of-experts architectures."""

    for model_type in ("qwen3_moe", "mixtral"):
        monkeypatch.setitem(
            arch_module.SUPPORTED_ARCHITECTURES, model_type, (model_type,)
        )


def _load(directory: Path, dtype=np.float32, parallelism: str = "single"):
    ckpt = weights.resolve_checkpoint(str(directory), None)
    check_config(ckpt.config)
    weights.fetch_shards(ckpt)
    tensors = weights.build_tensor_index(ckpt)
    arch = ArchConfig.from_hf(ckpt.config, ckpt.raw_config, tensors)
    plan = choose_plan(arch, dtype, parallelism)
    params = weights.load_params(ckpt, tensors, arch, dtype, plan)
    return ckpt, tensors, arch, plan, params


def _flatten(params: dict) -> dict[str, np.ndarray]:
    flat = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        flat["/".join(key.key for key in path)] = leaf
    return flat


def _reference_params(model, arch: ArchConfig) -> dict[str, torch.Tensor]:
    """The parameters of a transformers model, in the port's layout."""

    decoder = text_model(model)
    q_size = arch.num_attention_heads * arch.head_dim
    kv_size = arch.num_key_value_heads * arch.head_dim

    def per_layer(layer) -> dict[str, torch.Tensor]:
        attention = layer.self_attn
        mlp = layer.mlp
        tensors = {"attn_norm": layer.input_layernorm.weight}

        if hasattr(attention, "qkv_proj"):
            q, k, v = attention.qkv_proj.weight.split([q_size, kv_size, kv_size])
            tensors.update(q=q, k=k, v=v)
        else:
            for name in ("q", "k", "v"):
                projection = getattr(attention, f"{name}_proj")
                tensors[name] = projection.weight
                if projection.bias is not None:
                    tensors[f"{name}_bias"] = projection.bias
        if hasattr(attention, "q_norm"):
            tensors.update(
                q_norm=attention.q_norm.weight, k_norm=attention.k_norm.weight
            )
        tensors["o_proj"] = attention.o_proj.weight[None]
        if attention.o_proj.bias is not None:
            tensors["o_proj_bias"] = attention.o_proj.bias[None]

        if hasattr(layer, "pre_feedforward_layernorm"):
            tensors["attn_out_norm"] = layer.post_attention_layernorm.weight
            tensors["mlp_norm"] = layer.pre_feedforward_layernorm.weight
            tensors["mlp_out_norm"] = layer.post_feedforward_layernorm.weight
        else:
            tensors["mlp_norm"] = layer.post_attention_layernorm.weight

        if hasattr(mlp, "experts"):
            gate, up = mlp.experts.gate_up_proj.chunk(2, dim=1)
            tensors.update(
                router=mlp.gate.weight,
                expert_gate=gate,
                expert_up=up,
                expert_down=mlp.experts.down_proj,
            )
            return tensors

        if hasattr(mlp, "gate_up_proj"):
            gate, up = mlp.gate_up_proj.weight.chunk(2)
            tensors.update(gate=gate, up=up)
        else:
            for name in ("gate", "up"):
                projection = getattr(mlp, f"{name}_proj")
                tensors[name] = projection.weight
                if projection.bias is not None:
                    tensors[f"{name}_bias"] = projection.bias
        tensors["down_proj"] = mlp.down_proj.weight[None]
        if mlp.down_proj.bias is not None:
            tensors["down_proj_bias"] = mlp.down_proj.bias[None]

        return tensors

    layers = [per_layer(layer) for layer in decoder.layers]
    reference = {
        "embed": decoder.embed_tokens.weight,
        "final_norm": decoder.norm.weight,
        **{
            f"layers/{name}": torch.stack([layer[name] for layer in layers])
            for name in layers[0]
        },
    }
    if not arch.tied_head:
        reference["head"] = model.lm_head.weight
    return reference


def _as_float32(array) -> np.ndarray:
    return np.asarray(array).astype(np.float32)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.usefixtures("planned_moe")
def test_loaded_params_equal_state_dict(checkpoints, name: str) -> None:
    directory = checkpoints[name]
    _, tensors, arch, _, params = _load(directory)
    model = load_reference(directory)

    flat = _flatten(params)
    layout = weights.param_layout(arch)
    assert set(flat) == set(layout)

    reference = _reference_params(model, arch)
    assert set(reference) == {
        path for path in layout if not path.startswith("buffers/")
    }

    for path, expected in reference.items():
        assert flat[path].dtype == np.float32, path
        assert flat[path].shape == tuple(expected.shape), path
        np.testing.assert_array_equal(np.asarray(flat[path]), expected.detach().numpy())

    tables = build_tables(arch)
    np.testing.assert_array_equal(flat["buffers/inv_freq"], tables.inv_freq)
    np.testing.assert_array_equal(
        flat["buffers/attention_scaling"], tables.attention_scaling
    )
    np.testing.assert_array_equal(flat["buffers/window"], np.array(arch.windows))
    assert flat["buffers/window"].dtype == np.int32
    if arch.rope_switch == "long_factor":
        np.testing.assert_array_equal(
            flat["buffers/long_inv_freq"], tables.long_inv_freq
        )

    # The tied head is the (unscaled) embedding matrix.
    if arch.tied_head:
        assert tensors.head_key is None
        torch.testing.assert_close(
            model.lm_head.weight, model.get_input_embeddings().weight
        )


def test_text_prefix_and_head_keys(checkpoints) -> None:
    expected = {
        "llama": ("model.", None),
        "phi3_4k": ("model.", "lm_head.weight"),
        "qwen3-head-and-tie-flag": ("model.", "lm_head.weight"),
        # Multimodal wrappers written by transformers 5 keep the legacy layout.
        "gemma3": ("language_model.model.", None),
        "mistral3": ("language_model.model.", None),
        "mistral3-untied": ("language_model.model.", "language_model.lm_head.weight"),
    }
    for name, (prefix, head_key) in expected.items():
        _, tensors, arch, _, params = _load(checkpoints[name])
        assert tensors.prefix == prefix, name
        assert tensors.head_key == head_key, name
        assert arch.tied_head == (head_key is None), name
        assert ("head" in params) == (head_key is not None), name


def test_missing_head_without_tie_flag_raises(tmp_path) -> None:
    directory = build_checkpoint(
        tiny_config(REAL_MODELS["llama"]),
        tmp_path / "ckpt",
        edit_config=lambda config: config.update(tie_word_embeddings=False),
    )
    ckpt = weights.resolve_checkpoint(str(directory), None)
    with pytest.raises(UnsupportedCheckpointError, match="random head"):
        weights.build_tensor_index(ckpt)


@pytest.mark.parametrize(
    ("storage", "session"),
    [
        (torch.float32, ml_dtypes.bfloat16),
        (torch.float16, ml_dtypes.bfloat16),
        (torch.bfloat16, np.float32),
        (torch.bfloat16, ml_dtypes.bfloat16),
    ],
)
def test_dtype_casts(tmp_path, storage, session) -> None:
    directory = _build("phi3_4k", tmp_path / "ckpt", dtype=storage)
    _, tensors, arch, _, params = _load(directory, dtype=session)

    assert {info.dtype for info in tensors.tensors.values()} == {
        {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16"}[storage]
    }

    # Casting on the host rounds as torch does.
    torch_session = torch.bfloat16 if session is ml_dtypes.bfloat16 else torch.float32
    reference = _reference_params(load_reference(directory, dtype=storage), arch)
    flat = _flatten(params)
    for path, expected in reference.items():
        assert flat[path].dtype == np.dtype(session), path
        np.testing.assert_array_equal(
            _as_float32(flat[path]),
            expected.detach().to(torch_session).float().numpy(),
        )
    assert flat["buffers/inv_freq"].dtype == np.float32


def test_sharded_checkpoint_uses_the_index(checkpoints, tmp_path) -> None:
    directory = checkpoints["phi3-sharded"]
    ckpt = weights.resolve_checkpoint(str(directory), None)

    index = json.loads((directory / "model.safetensors.index.json").read_text())
    assert ckpt.index == index
    assert ckpt.shard_files == tuple(sorted(set(index["weight_map"].values())))
    assert len(ckpt.shard_files) > 1

    tensors = weights.build_tensor_index(ckpt)
    assert {key: info.shard for key, info in tensors.tensors.items()} == index[
        "weight_map"
    ]

    arch = ArchConfig.from_hf(ckpt.config, ckpt.raw_config, tensors)
    for layer in range(arch.num_hidden_layers):
        for component, suffix in [
            ("attn.o_proj", "self_attn.o_proj.weight"),
            ("mlp.down_proj", "mlp.down_proj.weight"),
        ]:
            key, info = weights.disk_key(tensors, component, layer, 0)
            assert key == f"model.layers.{layer}.{suffix}"
            assert info.shard == index["weight_map"][key]
            assert info.dtype == "F32"
            assert info.shape == weights.component_shape(arch, component)

    with pytest.raises(ValueError):
        weights.disk_key(tensors, "attn.o_proj", 0, 1)


def test_disk_key_legacy_layout(checkpoints) -> None:
    _, tensors, _, _, _ = _load(checkpoints["gemma3"])
    key, info = weights.disk_key(tensors, "mlp.down_proj", 2, 0)
    assert key == "language_model.model.layers.2.mlp.down_proj.weight"
    assert info.shard == "model.safetensors"


def test_local_shard_set_never_globs(tmp_path) -> None:
    directory = _build("phi3-sharded", tmp_path / "ckpt")
    stray = add_stray_files(directory)
    # An unindexed shard-like file (for example left over from an earlier save).
    (directory / "model.safetensors").write_bytes(b"not a safetensors file")

    ckpt = weights.resolve_checkpoint(str(directory), None)
    assert "model.safetensors" not in ckpt.shard_files
    assert not set(stray) & set(ckpt.shard_files)
    tensors = weights.build_tensor_index(ckpt)
    assert "output.weight" not in tensors.tensors


def test_stray_files_are_never_downloaded(tmp_path, monkeypatch) -> None:
    repo = _build("phi3-sharded", tmp_path / "repo")
    stray = add_stray_files(repo)
    (repo / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [2, 3], "repetition_penalty": 1.25})
    )

    hub = FakeHub(tmp_path / "cache", {"org/model": repo})
    hub.install(monkeypatch)

    ckpt = weights.resolve_checkpoint("org/model", "main")
    assert ckpt.sha == FakeHub.SHA
    assert not ckpt.offline
    # snapshot_download resolves the branch, so that the cache records its commit.
    assert hub.revisions == [("org/model", "main")]
    metadata = {name for _, name in hub.downloads}
    assert metadata == {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
    }
    assert ckpt.generation_config.eos_token_id == [2, 3]
    assert ckpt.generation_config.repetition_penalty == 1.25

    check_config(ckpt.config)
    weights.fetch_shards(ckpt)
    # The shards come from the resolved commit, whatever the branch points to now.
    assert hub.revisions[1:] == [("org/model", FakeHub.SHA)]
    downloaded = {name for _, name in hub.downloads}
    assert downloaded - metadata == set(ckpt.shard_files)
    assert not set(stray) & downloaded

    tensors = weights.build_tensor_index(ckpt)
    arch = ArchConfig.from_hf(ckpt.config, ckpt.raw_config, tensors)
    weights.load_params(
        ckpt, tensors, arch, np.float32, choose_plan(arch, np.float32, "single")
    )


# The commit of the cached repository in the offline tests.
CACHED_SHA = "7a6a3128ee4137a248d6d1582824592b87a81647"


def _fill_hub_cache(
    repo: Path,
    cache: Path,
    names: tuple[str, ...] | None = None,
) -> None:
    """
    Puts a checkpoint (only the files in `names`, if given) into a Hugging Face cache
    as "org/model" at CACHED_SHA, as transformers and upstream fill the cache: blobs,
    a snapshot of symbolic links to them and refs/main, but no cached file listing
    (trees/), without which snapshot_download lists the files on the Hub.
    """

    root = cache / "models--org--model"
    blobs = root / "blobs"
    snapshot = root / "snapshots" / CACHED_SHA
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)

    for index, path in enumerate(sorted(repo.iterdir())):
        if names is None or path.name in names:
            blob = blobs / f"blob{index}"
            shutil.copyfile(path, blob)
            (snapshot / path.name).symlink_to(os.path.relpath(blob, snapshot))

    (root / "refs").mkdir()
    (root / "refs" / "main").write_text(CACHED_SHA)


@pytest.fixture
def hub_cache(tmp_path, monkeypatch) -> Path:
    """An empty Hugging Face cache, which huggingface_hub uses instead of the real one."""

    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(cache))
    return cache


def _load_resolved(ckpt: weights.Checkpoint):
    check_config(ckpt.config)
    weights.fetch_shards(ckpt)
    tensors = weights.build_tensor_index(ckpt)
    arch = ArchConfig.from_hf(ckpt.config, ckpt.raw_config, tensors)
    plan = choose_plan(arch, np.float32, "single")
    return arch, weights.load_params(ckpt, tensors, arch, np.float32, plan)


@pytest.mark.parametrize("model_commit", [None, CACHED_SHA])
@pytest.mark.parametrize("name", ["llama", "phi3-sharded"])
def test_cached_model_loads_offline(
    checkpoints,
    hub_cache,
    monkeypatch,
    name: str,
    model_commit: str | None,
) -> None:
    directory = checkpoints[name]
    _fill_hub_cache(directory, hub_cache)
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)

    with pytest.warns(UserWarning, match=f"local cache at commit {CACHED_SHA}"):
        ckpt = weights.resolve_checkpoint("org/model", model_commit)
    assert ckpt.offline
    assert ckpt.sha == CACHED_SHA
    local = weights.resolve_checkpoint(str(directory), None)
    assert ckpt.shard_files == local.shard_files
    assert ckpt.raw_config == local.raw_config

    arch, params = _load_resolved(ckpt)
    _, _, expected_arch, _, expected = _load(directory)
    assert arch == expected_arch
    flat = _flatten(params)
    for path, value in _flatten(expected).items():
        np.testing.assert_array_equal(np.asarray(flat[path]), np.asarray(value))


def test_cached_model_loads_during_hub_outage(
    checkpoints,
    hub_cache,
    monkeypatch,
) -> None:
    _fill_hub_cache(checkpoints["phi3-sharded"], hub_cache)
    # Nothing listens on the discard port, so connections are refused.
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(hf_constants, "ENDPOINT", "http://127.0.0.1:9")

    with pytest.warns(UserWarning, match="ConnectError|refused"):
        ckpt = weights.resolve_checkpoint("org/model", None)
    assert ckpt.offline
    assert ckpt.sha == CACHED_SHA
    _load_resolved(ckpt)


def _response(status_code: int) -> httpx.Response:
    request = httpx.Request("GET", "https://huggingface.co/api/models/org/model")
    return httpx.Response(status_code, request=request)


@pytest.mark.parametrize(
    ("error", "falls_back"),
    [
        (OfflineModeIsEnabled("offline"), True),
        (httpx.ConnectError("refused"), True),
        (httpx.ReadTimeout("timed out"), True),
        (HfHubHTTPError("unavailable", response=_response(503)), True),
        # The Hub rate-limits its API separately from file downloads.
        (HfHubHTTPError("rate limited", response=_response(429)), True),
        (HfHubHTTPError("request timeout", response=_response(408)), True),
        (HfHubHTTPError("unauthorised", response=_response(401)), False),
        (RepositoryNotFoundError("not found", response=_response(404)), False),
        (GatedRepoError("gated", response=_response(403)), False),
        (RevisionNotFoundError("no revision", response=_response(404)), False),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, Exception) else None,
)
def test_only_an_unreachable_hub_falls_back_to_the_cache(
    checkpoints,
    hub_cache,
    monkeypatch,
    error: Exception,
    falls_back: bool,
) -> None:
    _fill_hub_cache(checkpoints["llama"], hub_cache)
    # snapshot_download finds the cached commit by itself, and model_info fails.
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)

    def model_info(*args, **kwargs):
        raise error

    monkeypatch.setattr(
        weights, "HfApi", lambda: SimpleNamespace(model_info=model_info)
    )

    if falls_back:
        with pytest.warns(UserWarning, match="local cache"):
            assert weights.resolve_checkpoint("org/model", None).offline
    else:
        with pytest.raises(type(error)):
            weights.resolve_checkpoint("org/model", None)


def test_offline_errors_name_what_is_not_cached(
    checkpoints,
    hub_cache,
    monkeypatch,
) -> None:
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)

    # Nothing is cached.
    for model_commit in (None, CACHED_SHA):
        with pytest.raises(LocalEntryNotFoundError) as raised:
            weights.resolve_checkpoint("org/model", model_commit)
        assert isinstance(raised.value.__cause__, OfflineModeIsEnabled)

    # Only the metadata is cached, as after an interrupted download.
    directory = checkpoints["phi3-sharded"]
    _fill_hub_cache(directory, hub_cache, weights.METADATA_FILES)
    with pytest.warns(UserWarning, match="local cache"):
        ckpt = weights.resolve_checkpoint("org/model", None)
    with pytest.raises(LocalEntryNotFoundError, match=ckpt.shard_files[0]):
        weights.fetch_shards(ckpt)

    # A single-file checkpoint with only its configuration cached.
    shutil.rmtree(hub_cache / "models--org--model")
    _fill_hub_cache(checkpoints["llama"], hub_cache, (weights.CONFIG_FILE,))
    with pytest.raises(LocalEntryNotFoundError, match="lacks"):
        weights.resolve_checkpoint("org/model", None)


def test_resolve_checkpoint_local(checkpoints, tmp_path) -> None:
    directory = checkpoints["llama"]
    ckpt = weights.resolve_checkpoint(str(directory), None)
    assert ckpt.model == str(directory)
    assert ckpt.snapshot_dir == str(directory)
    assert ckpt.sha is None
    assert not ckpt.offline
    assert ckpt.index is None
    assert ckpt.shard_files == ("model.safetensors",)
    assert ckpt.raw_config == json.loads((directory / "config.json").read_text())
    assert ckpt.config.model_type == "llama"
    # From generation_config.json, which save_pretrained writes (without it, see
    # test_generation_config_without_its_file_comes_from_config_json).
    assert (directory / "generation_config.json").exists()
    assert ckpt.generation_config.eos_token_id == ckpt.config.eos_token_id

    # Only safetensors checkpoints are supported.
    bare = tmp_path / "bare"
    bare.mkdir()
    for name in ("config.json", "model.safetensors"):
        (bare / name).write_bytes((directory / name).read_bytes())
    weights.resolve_checkpoint(str(bare), None)
    (bare / "model.safetensors").unlink()
    with pytest.raises(UnsupportedCheckpointError, match="safetensors"):
        weights.resolve_checkpoint(str(bare), None)


def _without_token_ids(config: dict) -> None:
    for section in (config, config.get("text_config") or {}):
        section.pop("eos_token_id", None)
        section.pop("bos_token_id", None)


# Checkpoints without generation_config.json: case -> (REAL_MODELS key, config.json
# edit).
GENERATION_CASES = {
    # LlamaConfig and Gemma3TextConfig default to token ids that the file lacks, which
    # the resolved config reports.
    "llama-no-token-ids": ("llama", _without_token_ids),
    "gemma3-no-token-ids": ("gemma3", _without_token_ids),
    # Generation parameters in config.json, as older checkpoints ship them, which the
    # resolved config drops.
    "llama-legacy-parameters": (
        "llama",
        lambda config: config.update(
            repetition_penalty=1.3, do_sample=True, temperature=0.7, top_k=20
        ),
    ),
    # A multimodal wrapper, resolved from the top level and its text_config.
    "mistral3": ("mistral3", lambda config: None),
}


@pytest.mark.parametrize("case", GENERATION_CASES)
def test_generation_config_without_its_file_comes_from_config_json(
    tmp_path,
    case: str,
) -> None:
    from heretic_tpu.backend.engine import decode_spec

    name, edit = GENERATION_CASES[case]
    directory = build_checkpoint(
        tiny_config(REAL_MODELS[name]), tmp_path / "ckpt", edit_config=edit
    )
    (directory / "generation_config.json").unlink()
    raw_config = json.loads((directory / "config.json").read_text())

    ckpt = weights.resolve_checkpoint(str(directory), None)
    # As transformers resolves it for upstream's model.generate().
    assert ckpt.generation_config == load_reference(directory).generation_config
    assert ckpt.raw_config == raw_config

    # Upstream's greedy calls.
    spec = decode_spec(
        ckpt.generation_config,
        {"do_sample": False},
        max_new_tokens=4,
        chunk=1,
        pad_id=0,
        key=None,
    )
    if case.endswith("no-token-ids"):
        assert ckpt.generation_config.eos_token_id is None
        # Generation runs to max_new_tokens, as upstream's does.
        assert (spec.eos_ids == -1).all()
    elif case == "llama-legacy-parameters":
        config = ckpt.generation_config
        assert (config.do_sample, config.temperature, config.top_k) == (True, 0.7, 20)
        assert spec.repetition_penalty == 1.3


def _save_twice(directory: Path, sharded_first: bool) -> Path:
    """
    Saves a checkpoint into a folder that holds an earlier save of the other layout,
    as exporting to the same folder twice does. The second save (seed 1) is current,
    but save_pretrained leaves a file of the first behind: the index (whose shards it
    deletes) when the first save is sharded, else model.safetensors.
    """

    config = tiny_config(REAL_MODELS["qwen3"])
    for seed, sharded in enumerate((sharded_first, not sharded_first)):
        build_checkpoint(
            config,
            directory,
            seed=seed,
            max_shard_size="100KB" if sharded else None,
        )

    files = {path.name for path in directory.iterdir()}
    assert {weights.INDEX_FILE, weights.SINGLE_FILE} <= files
    return directory


@pytest.mark.parametrize(
    "sharded_first", [True, False], ids=["stale-index", "stale-single-file"]
)
def test_resaved_folder_loads_the_latest_save(
    tmp_path,
    recwarn,
    sharded_first: bool,
) -> None:
    directory = _save_twice(tmp_path / "ckpt", sharded_first)
    latest = build_checkpoint(
        tiny_config(REAL_MODELS["qwen3"]),
        tmp_path / "latest",
        seed=1,
        max_shard_size=None if sharded_first else "100KB",
    )

    ckpt, _, arch, _, params = _load(directory)
    flat = _flatten(params)
    stale = [w for w in recwarn if "cannot be used" in str(w.message)]

    if sharded_first:
        assert len(stale) == 1
        assert "stale" in str(stale[0].message)
        assert ckpt.index is None
        assert ckpt.shard_files == (weights.SINGLE_FILE,)
        # transformers loads model.safetensors too.
        reference = _reference_params(load_reference(directory), arch)
        for path, expected in reference.items():
            np.testing.assert_array_equal(
                np.asarray(flat[path]), expected.detach().numpy()
            )
    else:
        # A complete index is newer than model.safetensors, which transformers would
        # load (see Divergences from upstream in DESIGN.md).
        assert not stale
        assert ckpt.index == json.loads((directory / weights.INDEX_FILE).read_text())
        assert weights.SINGLE_FILE not in ckpt.shard_files

    _, _, _, _, expected = _load(latest)
    for path, value in _flatten(expected).items():
        np.testing.assert_array_equal(np.asarray(flat[path]), np.asarray(value))


def test_index_with_missing_shards_and_no_single_file_raises(tmp_path) -> None:
    directory = _build("phi3-sharded", tmp_path / "ckpt")
    index = json.loads((directory / weights.INDEX_FILE).read_text())
    missing = max(index["weight_map"].values())
    (directory / missing).unlink()

    with pytest.raises(UnsupportedCheckpointError, match=missing):
        weights.resolve_checkpoint(str(directory), None)


@pytest.mark.parametrize("offline", [False, True], ids=["online", "offline"])
def test_stale_index_on_the_hub_is_ignored(
    tmp_path,
    hub_cache,
    monkeypatch,
    offline: bool,
) -> None:
    repo = _save_twice(tmp_path / "repo", sharded_first=True)

    if offline:
        # The cached snapshot has the index but none of its shards.
        _fill_hub_cache(repo, hub_cache)
        monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
        expected = ["local cache", f"{weights.INDEX_FILE} .* are not cached"]
    else:
        hub = FakeHub(tmp_path / "cache", {"org/model": repo})
        hub.install(monkeypatch)
        expected = [f"{weights.INDEX_FILE} .* is stale"]

    with pytest.warns(UserWarning) as caught:
        ckpt = weights.resolve_checkpoint("org/model", None)
    messages = [str(warning.message) for warning in caught]
    for pattern in expected:
        assert any(re.search(pattern, message) for message in messages), pattern
    assert ckpt.offline == offline
    assert ckpt.index is None
    assert ckpt.shard_files == (weights.SINGLE_FILE,)

    _load_resolved(ckpt)
    if not offline:
        downloaded = {name for _, name in hub.downloads}
        assert downloaded - set(weights.METADATA_FILES) == {weights.SINGLE_FILE}


def test_missing_or_misshapen_tensors_raise_before_reading(
    tmp_path,
    monkeypatch,
) -> None:
    from safetensors.numpy import load_file, save_file

    directory = _build("qwen2", tmp_path / "ckpt")
    path = directory / "model.safetensors"
    original = load_file(str(path))

    def read_nothing(*args, **kwargs):
        raise AssertionError("tensor data was read")

    for edit, match in [
        (lambda t: t.pop("model.layers.2.self_attn.k_proj.bias"), "missing"),
        (
            lambda t: t.update(
                {"model.layers.1.mlp.up_proj.weight": np.zeros((64, 64), np.float32)}
            ),
            "shape",
        ),
    ]:
        tensors = dict(original)
        edit(tensors)
        save_file(tensors, str(path), metadata={"format": "pt"})

        ckpt = weights.resolve_checkpoint(str(directory), None)
        index = weights.build_tensor_index(ckpt)
        arch = ArchConfig.from_hf(ckpt.config, ckpt.raw_config, index)
        with monkeypatch.context() as patch:
            patch.setattr(weights._ShardReader, "read", read_nothing)
            with pytest.raises(UnsupportedCheckpointError, match=match):
                weights.load_params(
                    ckpt,
                    index,
                    arch,
                    np.float32,
                    choose_plan(arch, np.float32, "single"),
                )


def test_float8_linear_weight_rejected(tmp_path) -> None:
    from safetensors.torch import load_file, save_file

    directory = _build("llama", tmp_path / "ckpt")
    path = directory / "model.safetensors"
    tensors = load_file(str(path))
    key = "model.layers.1.mlp.down_proj.weight"
    tensors[key] = tensors[key].to(torch.float8_e4m3fn)
    save_file(tensors, str(path), metadata={"format": "pt"})

    ckpt = weights.resolve_checkpoint(str(directory), None)
    check_config(ckpt.config)
    with pytest.raises(UnsupportedCheckpointError, match=f"{key}.*F8_E4M3"):
        weights.build_tensor_index(ckpt)


class _FakeDevice:
    """A device with the given memory limit, or with no memory statistics (as CPUs)."""

    def __init__(self, device, bytes_limit: int | None):
        self.device = device
        self.id = device.id
        self.bytes_limit = bytes_limit

    def memory_stats(self) -> dict[str, int] | None:
        if self.bytes_limit is None:
            return None
        return {"bytes_limit": self.bytes_limit, "bytes_in_use": 0}

    def __str__(self) -> str:
        return "FakeDevice"


def test_preflight(checkpoints, monkeypatch) -> None:
    ckpt = weights.resolve_checkpoint(str(checkpoints["gemma3"]), None)
    tensors = weights.build_tensor_index(ckpt)
    arch = ArchConfig.from_hf(ckpt.config, ckpt.raw_config, tensors)
    plan = choose_plan(arch, ml_dtypes.bfloat16, "single")

    # The footprint is the placed size, buffers in their own dtypes included.
    params = weights.load_params(ckpt, tensors, arch, ml_dtypes.bfloat16, plan)
    footprint = weights.device_footprint(arch, ml_dtypes.bfloat16, plan)
    assert footprint == sum(leaf.nbytes for leaf in jax.tree.leaves(params))
    assert weights.device_footprint(arch, np.float32, plan) > footprint

    # Devices that report no memory statistics (CPUs) skip the check.
    device = plan.devices[0]
    for limit, fits in [(None, True), (footprint, True), (footprint - 1, False)]:
        fake_devices = [_FakeDevice(device, limit)]
        monkeypatch.setattr(
            type(plan), "devices", property(lambda self, devices=fake_devices: devices)
        )
        if fits:
            weights.check_device_memory(arch, ml_dtypes.bfloat16, plan)
            continue

        def read_nothing(*args, **kwargs):
            raise AssertionError("tensor data was read")

        monkeypatch.setattr(weights._ShardReader, "read", read_nothing)
        with pytest.raises(DeviceMemoryError, match="FakeDevice"):
            weights.load_params(ckpt, tensors, arch, ml_dtypes.bfloat16, plan)


def test_placement_creates_only_full_parameter_arrays(checkpoints, monkeypatch) -> None:
    placed_shapes = []
    device_put = jax.device_put

    def recording_device_put(x, *args, **kwargs):
        placed_shapes.append(np.shape(x))
        return device_put(x, *args, **kwargs)

    monkeypatch.setattr(jax, "device_put", recording_device_put)

    live_before = len(jax.live_arrays())
    _, _, arch, plan, params = _load(checkpoints["gemma3"])
    layout = weights.param_layout(arch)

    # One device_put per parameter, each of the whole stacked array.
    assert sorted(placed_shapes) == sorted(leaf.shape for leaf in layout.values())
    assert len(jax.live_arrays()) - live_before == len(layout)

    skeleton = weights.param_skeleton(arch, np.float32, plan)
    assert jax.tree.structure(skeleton) == jax.tree.structure(params)
    for expected, actual in zip(jax.tree.leaves(skeleton), jax.tree.leaves(params)):
        assert expected.shape == actual.shape
        assert expected.dtype == actual.dtype
        assert expected.sharding == actual.sharding


def _upstream_modules(model) -> dict[str, list[str]]:
    """Module names of upstream's get_layer_modules, per component."""

    names = {id(module): name for name, module in model.named_modules()}
    try:
        layers = model.model.language_model.layers
    except AttributeError:
        layers = model.model.layers

    modules: dict[str, list[str]] = {}
    for layer in layers:
        candidates = [("attn.o_proj", layer.self_attn.o_proj)]
        with suppress(AttributeError):
            candidates.append(("mlp.down_proj", layer.mlp.down_proj))
        with suppress(AttributeError, TypeError):
            for expert in layer.mlp.experts:
                candidates.append(("mlp.down_proj", expert.down_proj))
        for component, module in candidates:
            if isinstance(module, torch.nn.Module):
                modules.setdefault(component, []).append(names[id(module)])
    return modules


@pytest.mark.parametrize("name", CASES)
@pytest.mark.usefixtures("planned_moe")
def test_module_paths_match_named_modules(checkpoints, name: str) -> None:
    from transformers import AutoConfig

    directory = checkpoints[name]
    _, _, arch, _, _ = _load(directory)

    raw_config = json.loads((directory / "config.json").read_text())
    with torch.device("meta"):
        model = model_class(raw_config).from_config(
            AutoConfig.from_pretrained(directory)
        )

    upstream = _upstream_modules(model)
    assert sorted(upstream) == weights.abliterable_components(arch)

    for component, names in upstream.items():
        assert names == [
            weights.module_path(arch, component, layer, 0)
            for layer in range(arch.num_hidden_layers)
        ]
        # One module per layer (M = 1).
        assert len(names) == arch.num_hidden_layers

    root = "model.language_model.layers" if arch.multimodal else "model.layers"
    assert (
        weights.module_path(arch, "attn.o_proj", 1, 0) == f"{root}.1.self_attn.o_proj"
    )
    with pytest.raises(ValueError):
        weights.module_path(arch, "mlp.gate_proj", 0, 0)


@pytest.mark.usefixtures("planned_moe")
def test_moe_layouts(checkpoints) -> None:
    for name, fused in [
        ("qwen3_moe", False),
        ("qwen3_moe-fused", True),
        ("mixtral", False),
    ]:
        _, tensors, arch, _, params = _load(checkpoints[name])
        assert arch.is_moe
        assert (arch.num_experts, arch.num_experts_per_tok) == (4, 2)
        assert arch.moe_intermediate_size == 32
        assert weights.abliterable_components(arch) == ["attn.o_proj"]
        assert "down_proj" not in params["layers"]
        assert params["layers"]["expert_down"].shape == (3, 4, 64, 32)

        key = "model.layers.0.mlp.experts.gate_up_proj"
        assert (key in tensors.tensors) == fused, name


def test_moe_checkpoints_accepted(checkpoints) -> None:
    for name in ("qwen3_moe", "mixtral"):
        ckpt = weights.resolve_checkpoint(str(checkpoints[name]), None)
        check_config(ckpt.config)


TPU_MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3-4B-Instruct-2507"]

requires_tpu = pytest.mark.skipif(
    jax.default_backend() != "tpu", reason="requires a TPU"
)


class _HostMemorySampler:
    """Samples the anonymous and file-backed resident memory of this process (Linux)."""

    def __init__(self):
        self.peak = {"RssAnon": 0, "RssFile": 0, "VmRSS": 0}
        self.stopped = threading.Event()

    def _sample(self) -> dict[str, int]:
        values = {}
        with open("/proc/self/status") as file:
            for line in file:
                name, _, value = line.partition(":")
                if name in self.peak:
                    values[name] = int(value.split()[0]) * 1024
        return values

    def _run(self) -> None:
        while not self.stopped.wait(0.005):
            for name, value in self._sample().items():
                self.peak[name] = max(self.peak[name], value)

    def __enter__(self):
        self.baseline = self._sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stopped.set()
        self.thread.join()


def _evict_page_cache(ckpt: weights.Checkpoint) -> None:
    for shard in ckpt.shard_files:
        path = os.path.realpath(os.path.join(ckpt.snapshot_dir, shard))
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


@pytest.mark.tpu
@pytest.mark.slow
@requires_tpu
@pytest.mark.parametrize("repo_id", TPU_MODELS)
def test_load_real_model_on_tpu(repo_id: str) -> None:
    from safetensors import safe_open

    device = jax.local_devices()[0]
    assert device.platform == "tpu"
    dtype = np.dtype(ml_dtypes.bfloat16)

    ckpt = weights.resolve_checkpoint(repo_id, None)
    check_config(ckpt.config)
    weights.fetch_shards(ckpt)
    tensors = weights.build_tensor_index(ckpt)
    arch = ArchConfig.from_hf(ckpt.config, ckpt.raw_config, tensors)
    plan = choose_plan(arch, dtype, "auto")
    assert plan.kind == "single"

    footprint = weights.device_footprint(arch, dtype, plan)
    largest = max(
        math.prod(leaf.shape) * (leaf.dtype or dtype).itemsize
        for leaf in weights.param_layout(arch).values()
    )

    for cache in ("cold", "warm"):
        if cache == "cold":
            _evict_page_cache(ckpt)
        in_use_before = device.memory_stats()["bytes_in_use"]
        with _HostMemorySampler() as host:
            start = time.perf_counter()
            params = weights.load_params(ckpt, tensors, arch, dtype, plan)
            seconds = time.perf_counter() - start
        in_use = device.memory_stats()["bytes_in_use"] - in_use_before
        anonymous = host.peak["RssAnon"] - host.baseline["RssAnon"]

        print(
            f"\n{repo_id} ({cache} page cache): {seconds:.1f} s, "
            f"{footprint / seconds / 2**30:.2f} GiB/s; device bytes in use "
            f"{in_use / 2**30:.3f} GiB, expected {footprint / 2**30:.3f} GiB; "
            f"peak RSS {host.peak['VmRSS'] / 2**30:.2f} GiB "
            f"(anonymous +{anonymous / 2**30:.2f} GiB, "
            f"file-backed {host.peak['RssFile'] / 2**30:.2f} GiB); "
            f"largest parameter {largest / 2**30:.2f} GiB"
        )

        # The placed size is the footprint (up to allocation padding).
        assert footprint <= in_use <= 1.01 * footprint

        del params

    # The TPU runtime's allocator keeps freed host memory resident, so RSS is only
    # a high-water mark. The live host memory is about one stacked parameter (plus
    # one layer of it as read from the checkpoint, and the prefetcher's buffer).
    tracemalloc.start()
    params = weights.load_params(ckpt, tensors, arch, dtype, plan)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(f"{repo_id}: peak live host memory {peak / 2**30:.2f} GiB")
    assert peak < 1.1 * largest + weights._Prefetcher.CHUNK_SIZE

    # The values on the device are those of the checkpoint.
    prefix = tensors.prefix
    with ExitStack() as stack:

        def read(key: str) -> np.ndarray:
            info = tensors.tensors[key]
            file = stack.enter_context(
                safe_open(os.path.join(ckpt.snapshot_dir, info.shard), framework="np")
            )
            return file.get_tensor(key)

        np.testing.assert_array_equal(
            np.asarray(params["embed"]), read(f"{prefix}embed_tokens.weight")
        )
        for layer in (0, arch.num_hidden_layers - 1):
            for name, suffix in [
                ("q", "self_attn.q_proj.weight"),
                ("v", "self_attn.v_proj.weight"),
                ("gate", "mlp.gate_proj.weight"),
            ]:
                np.testing.assert_array_equal(
                    np.asarray(params["layers"][name][layer]),
                    read(f"{prefix}layers.{layer}.{suffix}"),
                )
            for component in weights.abliterable_components(arch):
                key, _ = weights.disk_key(tensors, component, layer, 0)
                np.testing.assert_array_equal(
                    np.asarray(
                        params["layers"][weights.COMPONENTS[component]][layer, 0]
                    ),
                    read(key),
                )


@pytest.mark.tpu
@requires_tpu
def test_preflight_rejects_oversized_model_on_tpu() -> None:
    from tests.backend.tiny import real_arch

    # About 61 GiB in bfloat16, which does not fit into one v6e chip (31.25 GiB).
    arch = real_arch("Qwen/Qwen2.5-32B-Instruct")
    dtype = ml_dtypes.bfloat16
    plan = choose_plan(arch, dtype, "auto")
    assert plan.kind == "single"
    assert plan.devices[0].memory_stats() is not None

    with pytest.raises(DeviceMemoryError, match="GiB per device"):
        weights.check_device_memory(arch, dtype, plan)

    # Qwen3-4B fits in bfloat16 and in float32.
    arch = real_arch("Qwen/Qwen3-4B-Instruct-2507")
    weights.check_device_memory(arch, dtype, choose_plan(arch, dtype, "auto"))
    weights.check_device_memory(arch, np.float32, choose_plan(arch, np.float32, "auto"))
