# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Merged and adapter export, and the upload helper, on tiny checkpoints."""

import filecmp
import json
import os
import shutil
import time
from contextlib import suppress
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from huggingface_hub import constants as hf_constants
from safetensors import safe_open

from heretic_tpu.backend import arch as arch_module
from heretic_tpu.backend import export, weights
from heretic_tpu.backend.arch import ArchConfig, check_config
from tests.backend.tiny import (
    REAL_MODELS,
    FakeHub,
    add_stray_files,
    build_checkpoint,
    load_reference,
    model_class,
    tiny_config,
)

torch = pytest.importorskip("torch")

# Checkpoints stored in float32, which the logit comparisons use.
FLOAT32_CASES = [
    "llama",
    "llama-untied-biases",
    "mistral",
    "qwen2",
    "qwen3",
    "gemma3_text",
    "gemma3",
    "mistral3",
    "mistral3-untied",
    "phi3",
    "phi3_4k",
    "phi3-sharded",
    # Planned architectures (enabled by the planned_moe fixture).
    "qwen3_moe",
    "mixtral",
]

CASES = [*FLOAT32_CASES, "qwen2-bf16", "gemma3-bf16", "llama-f16"]

MULTIMODAL_CASES = {"gemma3", "gemma3-bf16", "mistral3", "mistral3-untied"}

# Processor files of multimodal checkpoints, with arbitrary contents.
PROCESSOR_FILES = {
    "preprocessor_config.json": {"image_processor_type": "Gemma3ImageProcessor"},
    "processor_config.json": {"processor_class": "Gemma3Processor"},
    "video_preprocessor_config.json": {"do_resize": True},
    "chat_template.json": {"chat_template": "{{ messages }}"},
}

CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }} {% endfor %}"

SPECIAL_TOKENS = {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}

# Upstream module path suffixes, to compute expected safetensors keys independently.
SUFFIXES = {"attn.o_proj": "self_attn.o_proj", "mlp.down_proj": "mlp.down_proj"}


def _build(name: str, directory: Path) -> Path:
    """Builds one of the checkpoint cases into a directory, with a tokenizer."""

    family, _, variant = name.partition("-")
    biases = {"attention_bias": True, "mlp_bias": True}
    config = tiny_config(
        REAL_MODELS[family],
        **(biases if variant == "untied-biases" else {}),
    )

    if variant == "untied-biases":
        build_checkpoint(config, directory, tie_word_embeddings=False)
    elif variant == "untied":
        # An untied head and no top-level tie key (Mistral3Config defaults to tying).
        build_checkpoint(
            config,
            directory,
            tie_word_embeddings=False,
            edit_config=lambda saved: saved.pop("tie_word_embeddings"),
        )
    elif variant == "sharded":
        build_checkpoint(config, directory, max_shard_size="40KB")
    elif variant == "bf16":
        build_checkpoint(config, directory, dtype=torch.bfloat16)
    elif variant == "f16":
        build_checkpoint(config, directory, dtype=torch.float16)
    else:
        build_checkpoint(config, directory)

    _add_tokenizer(directory)
    if name in MULTIMODAL_CASES:
        for file_name, contents in PROCESSOR_FILES.items():
            (directory / file_name).write_text(json.dumps(contents))
    return directory


def _add_tokenizer(directory: Path, **kwargs) -> None:
    """Saves a small word-level tokenizer with a chat template into a directory."""

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokens = ["<pad>", "<s>", "</s>", "<unk>", *(f"w{i}" for i in range(16))]
    backend = Tokenizer(
        models.WordLevel({token: i for i, token in enumerate(tokens)}, "<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        **SPECIAL_TOKENS,
        **kwargs,
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(directory)


def _facade_tokenizer(model: str, **kwargs):
    """The tokenizer as upstream (and the facade) prepares it."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


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


def _resolve(model: str) -> tuple[weights.Checkpoint, weights.TensorIndex, ArchConfig]:
    """Runs the facade's call sequence up to the ArchConfig."""

    ckpt = weights.resolve_checkpoint(model, None)
    check_config(ckpt.config)
    weights.fetch_shards(ckpt)
    tensors = weights.build_tensor_index(ckpt)
    arch = ArchConfig.from_hf(ckpt.config, ckpt.raw_config, tensors)
    return ckpt, tensors, arch


def _random_adapters(
    arch: ArchConfig,
    *,
    rank: int = 2,
    seed: int = 0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Adapters with an effect on the model, except in layer 1 (B = 0)."""

    rng = np.random.default_rng(seed)
    adapters = {}

    for component in weights.abliterable_components(arch):
        d_out, d_in = weights.component_shape(arch, component)
        layers = arch.num_hidden_layers
        A = rng.standard_normal((layers, 1, rank, d_in), dtype=np.float32)
        A /= np.sqrt(d_in)
        B = 0.2 * rng.standard_normal((layers, 1, d_out, rank), dtype=np.float32)
        B[1] = 0
        adapters[component] = (A, B)

    return adapters


def _modified_keys(
    tensors: weights.TensorIndex,
    adapters: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Safetensors key -> (B, A) of every module with a non-zero B."""

    modified = {}
    for component, (A, B) in adapters.items():
        for layer in range(A.shape[0]):
            if np.any(B[layer, 0]):
                key = f"{tensors.prefix}layers.{layer}.{SUFFIXES[component]}.weight"
                modified[key] = (B[layer, 0], A[layer, 0])
    return modified


def _read(path: Path) -> tuple[dict[str, str] | None, dict[str, np.ndarray]]:
    with safe_open(str(path), framework="np") as file:
        keys = file.keys()
        return file.metadata(), {key: file.get_tensor(key) for key in keys}


def _header(path: Path) -> bytes:
    with open(path, "rb") as file:
        size = int.from_bytes(file.read(8), "little")
        return file.read(size)


def _files(directory: Path) -> set[str]:
    return {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _assert_loads_cleanly(directory: Path) -> None:
    """Loads with upstream's class choice, with every tensor accounted for."""

    raw_config = json.loads((directory / "config.json").read_text())
    _, info = model_class(raw_config).from_pretrained(
        directory,
        output_loading_info=True,
    )
    assert info["missing_keys"] == set()
    assert info["unexpected_keys"] == set()
    assert info["mismatched_keys"] == set()


@pytest.mark.parametrize("name", CASES)
@pytest.mark.usefixtures("planned_moe")
def test_merged_export(checkpoints, name: str, tmp_path) -> None:
    source = checkpoints[name]
    ckpt, tensors, arch = _resolve(str(source))
    adapters = _random_adapters(arch)
    target = tmp_path / "merged"

    export.save_merged(
        str(target),
        ckpt,
        tensors,
        arch,
        adapters,
        _facade_tokenizer(str(source)),
    )

    # Exactly the shard set, the metadata files and the tokenizer and processor files.
    shards = set(ckpt.shard_files)
    metadata = {"config.json", "generation_config.json"}
    if ckpt.index is not None:
        metadata.add(weights.INDEX_FILE)
    tokenizer_files = {
        file_name
        for file_name in export.TOKENIZER_FILES
        if (source / file_name).exists()
    }
    assert {"tokenizer.json", "tokenizer_config.json"} <= tokenizer_files
    if name in MULTIMODAL_CASES:
        assert set(PROCESSOR_FILES) <= tokenizer_files
    assert _files(target) == shards | metadata | tokenizer_files

    for file_name in metadata:
        assert filecmp.cmp(source / file_name, target / file_name, shallow=False)
    for file_name in PROCESSOR_FILES.keys() & tokenizer_files:
        assert filecmp.cmp(source / file_name, target / file_name, shallow=False)

    modified = _modified_keys(tensors, adapters)
    # Layers 0 and 2 of every component.
    assert len(modified) == 2 * len(adapters)

    for shard in shards:
        # Same __metadata__, tensor order and offsets.
        source_metadata, source_tensors = _read(source / shard)
        target_metadata, target_tensors = _read(target / shard)
        assert target_metadata == source_metadata == {"format": "pt"}
        assert _header(target / shard) == _header(source / shard)
        assert target_tensors.keys() == source_tensors.keys()

        for key, original in source_tensors.items():
            merged = target_tensors[key]
            assert (merged.dtype, merged.shape) == (original.dtype, original.shape)

            if key in modified:
                B, A = modified[key]
                expected = (original.astype(np.float32) + B @ A).astype(original.dtype)
                assert merged.tobytes() == expected.tobytes(), key
                assert merged.tobytes() != original.tobytes(), key
            else:
                # Including vision tensors, embeddings, the head and layer 1.
                assert merged.tobytes() == original.tobytes(), key

        if not any(tensors.tensors[key].shard == shard for key in modified):
            assert filecmp.cmp(source / shard, target / shard, shallow=False)

    if name == "phi3-sharded":
        # Some shards hold no modified weight and are byte copies.
        assert {tensors.tensors[key].shard for key in modified} < shards

    _assert_loads_cleanly(target)


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


def _save_upstream_adapter(
    source: Path,
    adapters: dict[str, tuple[np.ndarray, np.ndarray]],
    directory: Path,
) -> None:
    """Saves the adapters as upstream does: get_peft_model, then save_pretrained."""

    from peft import LoraConfig, get_peft_model

    model = load_reference(source)
    modules = _upstream_modules(model)
    rank = next(iter(adapters.values()))[0].shape[2]

    # Upstream's Model.apply_lora.
    peft_config = LoraConfig(
        r=rank,
        target_modules=sorted(name for names in modules.values() for name in names),
        lora_alpha=rank,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, peft_config)

    with torch.no_grad():
        for component, names in modules.items():
            A, B = adapters[component]
            for layer, name in enumerate(names):
                module = peft_model.base_model.model.get_submodule(name)
                module.lora_A["default"].weight.copy_(torch.from_numpy(A[layer, 0]))
                module.lora_B["default"].weight.copy_(torch.from_numpy(B[layer, 0]))

    peft_model.save_pretrained(directory)


def _loaded_lora(model, prefix: str = "") -> dict[str, np.ndarray]:
    """LoRA weights of a model with the adapter "default", under PEFT's file keys."""

    return {
        prefix + name.replace(".default.", "."): parameter.detach().numpy()
        for name, parameter in model.named_parameters()
        if ".lora_" in name
    }


@pytest.mark.parametrize("name", CASES)
@pytest.mark.usefixtures("planned_moe")
def test_adapter_export(checkpoints, name: str, tmp_path) -> None:
    pytest.importorskip("peft")
    from peft import PeftModel

    source = checkpoints[name]
    ckpt, _, arch = _resolve(str(source))
    adapters = _random_adapters(arch, rank=3)
    target = tmp_path / "adapter"

    export.save_adapter(
        str(target),
        ckpt,
        arch,
        adapters,
        base_model=str(source),
        model_commit=None,
    )
    assert _files(target) == {"adapter_config.json", "adapter_model.safetensors"}

    # The same file as upstream's PeftModel.save_pretrained writes.
    upstream = tmp_path / "upstream"
    _save_upstream_adapter(source, adapters, upstream)
    metadata, exported = _read(target / export.ADAPTER_WEIGHTS_FILE)
    upstream_metadata, upstream_tensors = _read(upstream / export.ADAPTER_WEIGHTS_FILE)
    assert metadata == upstream_metadata == {"format": "pt"}
    assert exported.keys() == upstream_tensors.keys()
    for key, tensor in exported.items():
        assert tensor.dtype == np.float32
        np.testing.assert_array_equal(tensor, upstream_tensors[key])

    if arch.multimodal:
        assert (
            "base_model.model.model.language_model.layers.0.self_attn.o_proj"
            ".lora_A.weight"
        ) in exported
    lora_keys = {key for key in exported if key.endswith(".lora_A.weight")}
    assert len(lora_keys) == len(adapters) * arch.num_hidden_layers

    config = json.loads((target / export.ADAPTER_CONFIG_FILE).read_text())
    upstream_config = json.loads((upstream / export.ADAPTER_CONFIG_FILE).read_text())
    assert config["r"] == config["lora_alpha"] == 3
    assert config["revision"] is None
    assert config["target_modules"] == sorted(upstream_config["target_modules"])
    for key, value in config.items():
        if key not in ("revision", "target_modules"):
            assert value == upstream_config[key], key

    # Loads with PEFT on upstream's class, with every LoRA weight as exported.
    peft_model = PeftModel.from_pretrained(load_reference(source), target)
    loaded = _loaded_lora(peft_model)
    assert loaded.keys() == exported.keys()
    for key, tensor in exported.items():
        np.testing.assert_array_equal(loaded[key], tensor)

    result = peft_model.load_adapter(str(target), adapter_name="check")
    assert not result.missing_keys
    assert not result.unexpected_keys

    # And with transformers' load_adapter.
    base = load_reference(source)
    info = base.load_adapter(str(target))
    assert not info.missing_keys
    assert not info.unexpected_keys
    loaded = _loaded_lora(base, prefix="base_model.model.")
    assert loaded.keys() == exported.keys()
    for key, tensor in exported.items():
        np.testing.assert_array_equal(loaded[key], tensor)


def test_adapter_export_of_strided_arrays(checkpoints, tmp_path) -> None:
    # Host copies of TPU arrays can come in the device's layout, for example with
    # the rank axis of B [L, M, d_out, r] major (non-C-contiguous strides), and
    # safetensors writes the buffer of a NumPy array as it is.
    ckpt, tensors, arch = _resolve(str(checkpoints["qwen2"]))
    adapters = _random_adapters(arch, rank=3)
    strided = {
        component: tuple(np.asfortranarray(array) for array in pair)
        for component, pair in adapters.items()
    }
    assert not strided["attn.o_proj"][1][0, 0].flags["C_CONTIGUOUS"]

    target = tmp_path / "adapter"
    export.save_adapter(
        str(target),
        ckpt,
        arch,
        strided,
        base_model=str(checkpoints["qwen2"]),
        model_commit=None,
    )

    _, exported = _read(target / export.ADAPTER_WEIGHTS_FILE)
    for component, (A, B) in adapters.items():
        for layer in range(arch.num_hidden_layers):
            path = weights.module_path(arch, component, layer, 0)
            prefix = f"base_model.model.{path}"
            np.testing.assert_array_equal(
                exported[f"{prefix}.lora_A.weight"], A[layer, 0]
            )
            np.testing.assert_array_equal(
                exported[f"{prefix}.lora_B.weight"], B[layer, 0]
            )

    # The merged export is unaffected by the strides.
    export.save_merged(
        str(tmp_path / "merged"),
        ckpt,
        tensors,
        arch,
        strided,
        _facade_tokenizer(str(checkpoints["qwen2"])),
    )
    _, merged = _read(tmp_path / "merged" / "model.safetensors")
    for key, (B, A) in _modified_keys(tensors, adapters).items():
        original = _read(checkpoints["qwen2"] / "model.safetensors")[1][key]
        expected = (original.astype(np.float32) + B @ A).astype(original.dtype)
        np.testing.assert_array_equal(merged[key], expected)


def _logits(model, input_ids) -> np.ndarray:
    with torch.no_grad():
        return model(input_ids=input_ids).logits.float().numpy()


@pytest.mark.parametrize("name", FLOAT32_CASES)
@pytest.mark.usefixtures("planned_moe")
def test_merged_logits_match_adapters(checkpoints, name: str, tmp_path) -> None:
    pytest.importorskip("peft")
    from peft import PeftModel

    source = checkpoints[name]
    ckpt, tensors, arch = _resolve(str(source))
    adapters = _random_adapters(arch)

    merged_dir = tmp_path / "merged"
    adapter_dir = tmp_path / "adapter"
    export.save_merged(
        str(merged_dir),
        ckpt,
        tensors,
        arch,
        adapters,
        _facade_tokenizer(str(source)),
    )
    export.save_adapter(
        str(adapter_dir),
        ckpt,
        arch,
        adapters,
        base_model=str(source),
        model_commit=None,
    )

    # Token ids from 16 on avoid the special and image tokens of the tiny configs.
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(16, arch.vocab_size, (2, 12), generator=generator)

    base = _logits(load_reference(source), input_ids)
    merged = _logits(load_reference(merged_dir), input_ids)

    # The base model with the merge applied in PyTorch.
    reference = load_reference(source)
    with torch.no_grad():
        for component, (A, B) in adapters.items():
            for layer in range(arch.num_hidden_layers):
                path = weights.module_path(arch, component, layer, 0)
                delta = torch.from_numpy(B[layer, 0]) @ torch.from_numpy(A[layer, 0])
                reference.get_submodule(path).weight += delta
    manual = _logits(reference, input_ids)

    # The base model running the exported adapter, through PEFT and transformers.
    with_peft = _logits(
        PeftModel.from_pretrained(load_reference(source), adapter_dir),
        input_ids,
    )
    with_adapter = load_reference(source)
    with_adapter.load_adapter(str(adapter_dir))
    with_adapter = _logits(with_adapter, input_ids)

    # The adapters change the model.
    assert np.abs(merged - base).max() > 0.1

    np.testing.assert_allclose(merged, manual, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(with_peft, manual, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(with_adapter, manual, rtol=1e-4, atol=1e-4)


def _cache_snapshot(repo: Path, root: Path) -> Path:
    """A copy of a checkpoint laid out like a Hugging Face cache snapshot."""

    blobs = root / "blobs"
    snapshot = root / "snapshots" / FakeHub.SHA
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)

    for index, path in enumerate(sorted(repo.iterdir())):
        blob = blobs / f"blob{index}"
        shutil.copyfile(path, blob)
        (snapshot / path.name).symlink_to(os.path.relpath(blob, snapshot))
    return snapshot


def test_untouched_shards_are_byte_copies(checkpoints, tmp_path) -> None:
    snapshot = _cache_snapshot(checkpoints["phi3-sharded"], tmp_path / "cache")
    ckpt, tensors, arch = _resolve(str(snapshot))

    # Only the attention output projection of layer 0 is modified.
    adapters = _random_adapters(arch)
    for component, (_, B) in adapters.items():
        B[1:] = 0
        if component != "attn.o_proj":
            B[:] = 0
    key, info = weights.disk_key(tensors, "attn.o_proj", 0, 0)

    # A target holding a symlink at a shard's name, which must be replaced rather
    # than written through.
    target = tmp_path / "merged"
    target.mkdir()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    (target / info.shard).symlink_to(outside)

    export.save_merged(
        str(target),
        ckpt,
        tensors,
        arch,
        adapters,
        _facade_tokenizer(str(snapshot)),
    )
    assert outside.read_bytes() == b"outside"

    untouched = set(ckpt.shard_files) - {info.shard}
    assert untouched
    for shard in ckpt.shard_files:
        path = target / shard
        assert path.is_file() and not path.is_symlink()
        assert filecmp.cmp(snapshot / shard, path, shallow=False) == (
            shard in untouched
        )
    assert not list(target.glob("*.tmp"))

    _, merged = _read(target / info.shard)
    _, original = _read(snapshot / info.shard)
    A, B = adapters["attn.o_proj"]
    np.testing.assert_array_equal(merged[key], original[key] + B[0, 0] @ A[0, 0])


def test_stale_weights_are_removed(checkpoints, tmp_path) -> None:
    target = tmp_path / "export"
    target.mkdir()
    # Leftovers of earlier exports, and unrelated files that are kept.
    (target / "model.safetensors").write_bytes(b"stale")
    (target / "model-00009-of-00009.safetensors").write_bytes(b"stale")
    (target / "model-extra.safetensors").write_bytes(b"kept")
    (target / "notes.txt").write_text("kept")
    kept = {"model-extra.safetensors", "notes.txt"}

    def save(name: str) -> weights.Checkpoint:
        ckpt, tensors, arch = _resolve(str(checkpoints[name]))
        export.save_merged(
            str(target),
            ckpt,
            tensors,
            arch,
            _random_adapters(arch),
            _facade_tokenizer(str(checkpoints[name])),
        )
        return ckpt

    sharded = save("phi3-sharded")
    assert len(sharded.shard_files) > 1
    weight_files = {path.name for path in target.glob("*.safetensors")}
    assert weight_files == set(sharded.shard_files) | {"model-extra.safetensors"}
    assert (target / weights.INDEX_FILE).exists()
    assert kept <= _files(target)
    _assert_loads_cleanly(target)

    # A single-file export over the sharded one leaves no index and no shard.
    save("phi3")
    weight_files = {path.name for path in target.glob("*.safetensors")}
    assert weight_files == {"model.safetensors", "model-extra.safetensors"}
    assert not (target / weights.INDEX_FILE).exists()
    assert kept <= _files(target)
    _assert_loads_cleanly(target)

    # And a sharded export over the single file removes it.
    save("phi3-sharded")
    assert not (target / "model.safetensors").exists()
    _assert_loads_cleanly(target)


def test_source_directory_is_refused(checkpoints, tmp_path) -> None:
    source = tmp_path / "source"
    shutil.copytree(checkpoints["llama"], source)
    ckpt, tensors, arch = _resolve(str(source))
    adapters = _random_adapters(arch)
    tokenizer = _facade_tokenizer(str(source))

    link = tmp_path / "link"
    link.symlink_to(source)
    (tmp_path / "inner_link").symlink_to(source / "sub")

    def contents() -> dict[Path, bytes | None]:
        return {
            path: path.read_bytes() if path.is_file() else None
            for path in source.rglob("*")
        }

    before = contents()

    for target in [
        source,
        source / "sub",
        source / "a" / "b",
        tmp_path / "source" / ".." / "source" / "sub",
        link,
        link / "sub",
        tmp_path / "inner_link",
    ]:
        with pytest.raises(ValueError, match="source checkpoint"):
            export.save_merged(str(target), ckpt, tensors, arch, adapters, tokenizer)

    # Nothing was written.
    assert contents() == before

    # A sibling directory with a similar name is not inside the source.
    sibling = tmp_path / "source-merged"
    export.save_merged(str(sibling), ckpt, tensors, arch, adapters, tokenizer)
    assert (sibling / "model.safetensors").exists()


def test_invalid_adapters_are_rejected(checkpoints, tmp_path) -> None:
    ckpt, tensors, arch = _resolve(str(checkpoints["llama"]))
    tokenizer = _facade_tokenizer(str(checkpoints["llama"]))
    adapters = _random_adapters(arch)
    A, B = adapters["mlp.down_proj"]

    for invalid in [
        {"attn.o_proj": adapters["attn.o_proj"]},
        {**adapters, "mlp.gate_proj": (A, B)},
        {**adapters, "mlp.down_proj": (A[:2], B[:2])},
        {**adapters, "mlp.down_proj": (A[:, :, :1], B)},
        {**adapters, "mlp.down_proj": (B, A)},
        {**adapters, "mlp.down_proj": (A[0], B[0])},
    ]:
        target = tmp_path / "target"
        with pytest.raises(ValueError):
            export.save_merged(str(target), ckpt, tensors, arch, invalid, tokenizer)
        with pytest.raises(ValueError):
            export.save_adapter(
                str(target),
                ckpt,
                arch,
                invalid,
                base_model="model",
                model_commit=None,
            )
        assert not target.exists()


def test_stray_files_are_never_downloaded_or_exported(tmp_path, monkeypatch) -> None:
    repo = _build("phi3-sharded", tmp_path / "repo")
    stray = add_stray_files(repo)

    hub = FakeHub(tmp_path / "cache", {"org/model": repo})
    hub.install(monkeypatch, weights, export)

    ckpt, tensors, arch = _resolve("org/model")
    assert ckpt.sha == FakeHub.SHA
    adapters = _random_adapters(arch)
    # The facade loads the tokenizer from the Hub at the resolved commit.
    tokenizer = _facade_tokenizer(str(repo))

    target = tmp_path / "merged"
    export.save_merged(str(target), ckpt, tensors, arch, adapters, tokenizer)

    downloaded = {file_name for _, file_name in hub.downloads}
    tokenizer_files = {"chat_template.jinja", "tokenizer.json", "tokenizer_config.json"}
    assert downloaded == {
        "config.json",
        "generation_config.json",
        weights.INDEX_FILE,
        *ckpt.shard_files,
        *tokenizer_files,
    }
    assert not set(stray) & downloaded

    exported = _files(target)
    assert not set(stray) & exported
    assert {
        file_name for file_name in exported if file_name.endswith(".safetensors")
    } == (set(ckpt.index["weight_map"].values()))
    assert exported == downloaded

    # The adapter records the base model and the resolved commit, or model_commit
    # when it is set.
    for model_commit, revision in [(None, FakeHub.SHA), ("main", "main")]:
        export.save_adapter(
            str(tmp_path / "adapter"),
            ckpt,
            arch,
            adapters,
            base_model="org/model",
            model_commit=model_commit,
        )
        config = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
        assert config["base_model_name_or_path"] == "org/model"
        assert config["revision"] == revision


def test_merged_export_of_a_checkpoint_resolved_offline(
    checkpoints,
    tmp_path,
    monkeypatch,
) -> None:
    # A cache as transformers fills it, without the file listing of the commit,
    # for which snapshot_download asks the Hub even when given a commit hash.
    source = checkpoints["phi3-sharded"]
    cache = tmp_path / "hub"
    snapshot = _cache_snapshot(source, cache / "models--org--model")
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)

    with pytest.warns(UserWarning, match="local cache"):
        ckpt = weights.resolve_checkpoint("org/model", FakeHub.SHA)
    assert ckpt.offline
    check_config(ckpt.config)
    weights.fetch_shards(ckpt)
    tensors = weights.build_tensor_index(ckpt)
    arch = ArchConfig.from_hf(ckpt.config, ckpt.raw_config, tensors)

    target = tmp_path / "merged"
    export.save_merged(
        str(target),
        ckpt,
        tensors,
        arch,
        _random_adapters(arch),
        _facade_tokenizer(str(snapshot)),
    )

    # Everything, the tokenizer files included, is taken from the cache.
    assert _files(target) == _files(source)
    _assert_loads_cleanly(target)


def test_tokenizer_and_processor_files(checkpoints, tmp_path) -> None:
    from transformers import AutoTokenizer

    source = tmp_path / "source"
    shutil.copytree(checkpoints["llama"], source)
    # The source tokenizer configuration sets the padding side, so transformers
    # persists the facade's.
    _add_tokenizer(source, padding_side="right")
    legacy_files = {
        "special_tokens_map.json": json.dumps(SPECIAL_TOKENS),
        "added_tokens.json": "{}",
        "tokenizer.model": "sentencepiece model",
        "vocab.json": json.dumps({"w0": 0}),
        "merges.txt": "#version: 0.2\n",
    }
    for file_name, contents in legacy_files.items():
        (source / file_name).write_text(contents)
    for file_name, contents in PROCESSOR_FILES.items():
        (source / file_name).write_text(json.dumps(contents))

    ckpt, tensors, arch = _resolve(str(source))
    tokenizer = _facade_tokenizer(str(source))
    assert tokenizer.pad_token == "</s>"
    target = tmp_path / "merged"
    export.save_merged(
        str(target),
        ckpt,
        tensors,
        arch,
        _random_adapters(arch),
        tokenizer,
    )

    # The tokenizer was saved over the copied files, as save_pretrained alone writes it.
    saved = tmp_path / "saved"
    saved_files = {Path(path).name for path in tokenizer.save_pretrained(saved)}
    for file_name in saved_files:
        assert filecmp.cmp(saved / file_name, target / file_name, shallow=False)

    # Every other listed file is a copy of the source's.
    assert set(export.TOKENIZER_FILES) <= _files(target)
    for file_name in set(export.TOKENIZER_FILES) - saved_files:
        assert filecmp.cmp(source / file_name, target / file_name, shallow=False)

    tokenizer_config = json.loads((target / "tokenizer_config.json").read_text())
    assert tokenizer_config["pad_token"] == "</s>"
    assert tokenizer_config["padding_side"] == "left"

    reloaded = AutoTokenizer.from_pretrained(target)
    assert reloaded.pad_token == "</s>"
    assert reloaded.padding_side == "left"
    assert reloaded("w1 w2").input_ids == tokenizer("w1 w2").input_ids
    chat = [{"role": "user", "content": "w3"}]
    assert reloaded.apply_chat_template(chat, tokenize=False) == (
        tokenizer.apply_chat_template(chat, tokenize=False)
    )


class _RecordingApi:
    """Records the HfApi calls of push_to_hub and the folder contents it uploads."""

    def __init__(self, calls: list, uploads: list, token: str | None = None):
        self.calls = calls
        self.uploads = uploads
        calls.append(("init", token))

    def create_repo(self, repo_id: str, *, private=None, exist_ok=False, **kwargs):
        self.calls.append(("create_repo", repo_id, private, exist_ok))
        return SimpleNamespace(repo_id=repo_id if "/" in repo_id else f"user/{repo_id}")

    def upload_folder(self, *, repo_id: str, folder_path: str, **kwargs):
        self.calls.append(("upload_folder", repo_id, folder_path))
        folder = Path(folder_path)
        self.uploads.append(
            {name: (folder / name).read_bytes() for name in _files(folder)}
        )


def test_push_to_hub(checkpoints, tmp_path, monkeypatch) -> None:
    calls: list = []
    uploads: list = []
    monkeypatch.setattr(
        export,
        "HfApi",
        lambda token=None: _RecordingApi(calls, uploads, token),
    )

    def write(directory: str) -> None:
        Path(directory, "weights.bin").write_bytes(b"weights")
        Path(directory, "sub").mkdir()
        Path(directory, "sub", "file.txt").write_text("text")

    export.push_to_hub("model-heretic", write, private=True, token="secret")
    ((_, _, folder),) = [call for call in calls if call[0] == "upload_folder"]
    assert calls == [
        ("init", "secret"),
        ("create_repo", "model-heretic", True, True),
        ("upload_folder", "user/model-heretic", folder),
    ]
    assert uploads == [{"weights.bin": b"weights", "sub/file.txt": b"text"}]
    # The temporary directory is removed.
    assert not Path(folder).exists()

    # A failed export creates no repository.
    calls.clear()

    def fail(directory: str) -> None:
        raise RuntimeError("export failed")

    with pytest.raises(RuntimeError, match="export failed"):
        export.push_to_hub("org/model-heretic", fail, private=False, token="secret")
    assert calls == []

    # Exactly the files of an export are uploaded.
    source = checkpoints["phi3-sharded"]
    ckpt, tensors, arch = _resolve(str(source))
    adapters = _random_adapters(arch)
    tokenizer = _facade_tokenizer(str(source))
    merged = tmp_path / "merged"
    export.save_merged(str(merged), ckpt, tensors, arch, adapters, tokenizer)

    uploads.clear()
    export.push_to_hub(
        "org/model-heretic",
        partial(
            export.save_merged,
            ckpt=ckpt,
            tensors=tensors,
            arch=arch,
            adapters=adapters,
            tokenizer=tokenizer,
        ),
        private=False,
        token="secret",
    )
    export.push_to_hub(
        "org/model-heretic",
        partial(
            export.save_adapter,
            ckpt=ckpt,
            arch=arch,
            adapters=adapters,
            base_model=str(source),
            model_commit=None,
        ),
        private=False,
        token="secret",
    )
    merged_upload, adapter_upload = uploads
    assert merged_upload == {
        name: (merged / name).read_bytes() for name in _files(merged)
    }
    assert set(adapter_upload) == {"adapter_config.json", "adapter_model.safetensors"}


@pytest.mark.slow
@pytest.mark.parametrize("repo_id", ["HuggingFaceTB/SmolLM2-135M-Instruct"])
def test_real_model_export(repo_id: str, tmp_path) -> None:
    """Exports a real checkpoint from the Hub, with rank-1 adapters as abliteration makes."""

    pytest.importorskip("peft")
    from peft import PeftModel

    ckpt, tensors, arch = _resolve(repo_id)
    tokenizer = _facade_tokenizer(repo_id, revision=ckpt.sha)
    adapters = _random_adapters(arch, rank=1)

    target = tmp_path / "merged"
    start = time.perf_counter()
    export.save_merged(str(target), ckpt, tensors, arch, adapters, tokenizer)
    print(f"Merged export of {repo_id}: {time.perf_counter() - start:.1f} s")

    assert {path.name for path in target.glob("*.safetensors")} == set(ckpt.shard_files)
    for file_name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
        assert (target / file_name).exists()

    modified = _modified_keys(tensors, adapters)
    snapshot = Path(ckpt.snapshot_dir)
    for shard in ckpt.shard_files:
        _, original = _read(snapshot / shard)
        _, merged = _read(target / shard)
        for key, tensor in original.items():
            if key in modified:
                B, A = modified[key]
                expected = (tensor.astype(np.float32) + B @ A).astype(tensor.dtype)
                assert merged[key].tobytes() == expected.tobytes(), key
            else:
                assert merged[key].tobytes() == tensor.tobytes(), key
    _assert_loads_cleanly(target)

    adapter = tmp_path / "adapter"
    export.save_adapter(
        str(adapter),
        ckpt,
        arch,
        adapters,
        base_model=repo_id,
        model_commit=None,
    )
    config = json.loads((adapter / export.ADAPTER_CONFIG_FILE).read_text())
    assert config["revision"] == ckpt.sha
    base = model_class(ckpt.raw_config).from_pretrained(repo_id, revision=ckpt.sha)
    peft_model = PeftModel.from_pretrained(base, adapter)
    assert len(_loaded_lora(peft_model)) == 2 * len(adapters) * arch.num_hidden_layers
