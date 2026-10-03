# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""ArchConfig and check_config against real configurations and transformers modules."""

import dataclasses
import importlib
import inspect
import json

import numpy as np
import pytest
from huggingface_hub import constants as hf_constants
from safetensors.numpy import save_file
from transformers import (
    AutoConfig,
    Gemma3TextConfig,
    LlamaConfig,
    LlavaConfig,
    MixtralConfig,
    Qwen2Config,
    Qwen3MoeConfig,
)

from heretic_tpu.backend import arch as arch_module
from heretic_tpu.backend import weights
from heretic_tpu.backend.arch import (
    NO_WINDOW,
    ArchConfig,
    check_config,
)
from heretic_tpu.backend.errors import (
    UnsupportedArchitectureError,
    UnsupportedCheckpointError,
)
from heretic_tpu.backend.weights import TensorIndex, TensorInfo
from tests.backend.tiny import (
    FakeHub,
    real_arch,
    real_config,
    real_raw_config,
    real_tensor_index,
)

# The models DESIGN.md lists (with an ungated mirror of google/gemma-3-4b-it),
# plus the text-only Gemma 3 and Mistral.
REAL_MODELS = [
    "unsloth/gemma-3-4b-it",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-4-mini-instruct",
    "unsloth/Llama-3.2-1B-Instruct",
    "unsloth/gemma-3-1b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

SYNTHETIC_CASES = {
    "qwen3-yarn": (
        "Qwen/Qwen3-0.6B",
        {
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
                "rope_theta": 1000000,
            }
        },
    ),
    "llama-dynamic": (
        "unsloth/Llama-3.2-1B-Instruct",
        {
            "rope_parameters": {
                "rope_type": "dynamic",
                "factor": 2.0,
                "rope_theta": 500000.0,
            },
            "max_position_embeddings": 64,
        },
    ),
}

CASES = [(repo_id, {}) for repo_id in REAL_MODELS] + list(SYNTHETIC_CASES.values())
CASE_IDS = REAL_MODELS + list(SYNTHETIC_CASES)


def _modelling_class(text_config, suffix: str) -> type:
    from transformers.models.auto.modeling_auto import MODEL_MAPPING

    module = importlib.import_module(MODEL_MAPPING[type(text_config)].__module__)
    (cls,) = [
        cls
        for name, cls in inspect.getmembers(module, inspect.isclass)
        if name.endswith(suffix) and cls.__module__ == module.__name__
    ]
    return cls


@pytest.mark.parametrize(("repo_id", "overrides"), CASES, ids=CASE_IDS)
def test_fields_match_transformers_modules(repo_id: str, overrides: dict) -> None:
    torch = pytest.importorskip("torch")

    text_config = real_config(repo_id, **overrides).get_text_config()
    arch = real_arch(repo_id, **overrides)

    assert arch.model_type == text_config.model_type
    assert arch.num_hidden_layers == text_config.num_hidden_layers
    assert len(arch.layer_types) == len(arch.windows) == len(arch.rope)

    attention_class = _modelling_class(text_config, "Attention")
    rotary = _modelling_class(text_config, "RotaryEmbedding")(config=text_config)

    for layer in range(arch.num_hidden_layers):
        with torch.device("meta"):
            attention = attention_class(text_config, layer_idx=layer)

        assert arch.head_dim == attention.head_dim
        assert arch.attention_scale == attention.scaling
        assert (
            arch.num_attention_heads // arch.num_key_value_heads
            == attention.num_key_value_groups
        )

        if hasattr(attention, "qkv_proj"):
            assert attention.qkv_proj.out_features == (
                (arch.num_attention_heads + 2 * arch.num_key_value_heads)
                * arch.head_dim
            )
        else:
            assert attention.q_proj.out_features == (
                arch.num_attention_heads * arch.head_dim
            )
            assert attention.k_proj.out_features == (
                arch.num_key_value_heads * arch.head_dim
            )
        assert attention.o_proj.out_features == arch.hidden_size

        # Llama, Mistral and Phi-3 pass the config's window (if any) for every layer.
        if hasattr(attention, "sliding_window"):
            window = attention.sliding_window
        else:
            window = getattr(text_config, "sliding_window", None)
        assert arch.windows[layer] == (NO_WINDOW if window is None else window)
        assert (arch.layer_types[layer] == "sliding_attention") == (window is not None)

        if isinstance(rotary.rope_type, dict):
            layer_type = arch.layer_types[layer]
            rope_type = rotary.rope_type[layer_type]
            inv_freq = getattr(rotary, f"{layer_type}_inv_freq")
        else:
            rope_type = rotary.rope_type
            inv_freq = rotary.inv_freq
        assert arch.rope[layer].rope_type == rope_type
        assert arch.rope[layer].rot == 2 * inv_freq.numel()


@pytest.mark.parametrize(("repo_id", "overrides"), CASES, ids=CASE_IDS)
def test_round_trip_through_save_pretrained(
    repo_id: str,
    overrides: dict,
    tmp_path,
) -> None:
    arch = real_arch(repo_id, **overrides)

    real_config(repo_id, **overrides).save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert "rope_scaling" not in json.dumps(saved)

    round_tripped = ArchConfig.from_hf(
        AutoConfig.from_pretrained(tmp_path),
        saved,
        real_tensor_index(repo_id),
    )
    assert round_tripped == arch
    assert hash(round_tripped) == hash(arch)


def test_hash_and_equality() -> None:
    for repo_id in REAL_MODELS:
        first = real_arch(repo_id)
        second = real_arch(repo_id)
        assert first is not second
        assert first == second
        assert hash(first) == hash(second)
        assert {first: 1}[second] == 1

    assert real_arch(REAL_MODELS[0]) != real_arch(REAL_MODELS[1])


def test_tied_head() -> None:
    repo_id = "Qwen/Qwen3-0.6B"
    tensors = real_tensor_index(repo_id)

    # Qwen3-0.6B sets tie_word_embeddings but ships lm_head.weight, which is used.
    assert real_config(repo_id).tie_word_embeddings
    assert tensors.head_key == "lm_head.weight"
    arch = real_arch(repo_id)
    assert not arch.tied_head

    # A re-saved copy without the head ties it to the embedding.
    without_head = TensorIndex.from_infos(
        tensors.snapshot_dir,
        {k: v for k, v in tensors.tensors.items() if k != "lm_head.weight"},
        real_config(repo_id),
    )
    assert without_head.head_key is None
    tied = ArchConfig.from_hf(
        real_config(repo_id), real_raw_config(repo_id), without_head
    )
    assert tied.tied_head
    assert tied != arch
    assert dataclasses.replace(arch, tied_head=True) == tied


def test_real_tensor_index_is_cached_on_disk(tmp_path, monkeypatch) -> None:
    repo_id = "Qwen/Qwen3-0.6B"
    expected = real_tensor_index(repo_id)

    # Once cached (by the call above, if not before), reading the tensor index needs
    # no Hub access, so that the default test selection can run offline.
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    assert real_tensor_index.__wrapped__(repo_id) == expected

    # Otherwise, the error says how to fill the cache.
    monkeypatch.setattr(hf_constants, "HF_ASSETS_CACHE", str(tmp_path))
    with pytest.raises(RuntimeError, match="OfflineModeIsEnabled.*Run the tests once"):
        real_tensor_index.__wrapped__(repo_id)


def test_multimodal() -> None:
    repo_id = "unsloth/Llama-3.2-1B-Instruct"
    config = real_config(repo_id)
    raw_config = real_raw_config(repo_id)
    tensors = real_tensor_index(repo_id)

    assert not ArchConfig.from_hf(config, raw_config, tensors).multimodal
    # The key decides, whatever its value (upstream's get_model_class rule).
    assert ArchConfig.from_hf(
        config, {**raw_config, "vision_config": None}, tensors
    ).multimodal

    assert real_arch("unsloth/gemma-3-4b-it").multimodal
    assert real_arch("mistralai/Mistral-Small-3.1-24B-Instruct-2503").multimodal
    assert not real_arch("unsloth/gemma-3-1b-it").multimodal


def test_known_fields() -> None:
    gemma3 = real_arch("unsloth/gemma-3-4b-it")
    assert gemma3.model_type == "gemma3_text"
    assert gemma3.tied_head
    assert gemma3.activation == "gelu_pytorch_tanh"
    assert gemma3.attention_scale == 256**-0.5
    assert gemma3.has_qk_norm and gemma3.has_sandwich_norms
    assert gemma3.rope_per_layer_type
    assert set(zip(gemma3.layer_types, gemma3.windows)) == {
        ("sliding_attention", 1024),
        ("full_attention", NO_WINDOW),
    }
    assert gemma3.rope_switch is None

    qwen2 = real_arch("Qwen/Qwen2.5-0.5B-Instruct")
    assert qwen2.biases == ("k", "q", "v")
    assert qwen2.head_dim == 64
    assert set(qwen2.windows) == {NO_WINDOW}

    mistral3_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    mistral3 = real_arch(mistral3_id)
    assert mistral3.model_type == "mistral"
    # Mistral3Config ties by default, but the checkpoint ships a distinct head.
    assert real_config(mistral3_id).tie_word_embeddings
    assert not mistral3.tied_head
    assert real_tensor_index(mistral3_id).prefix == "language_model.model."
    assert real_tensor_index(mistral3_id).head_key == "language_model.lm_head.weight"

    phi4 = real_arch("microsoft/Phi-4-mini-instruct")
    assert phi4.head_dim == 128
    assert phi4.rot == 96
    assert phi4.tied_head
    assert phi4.rope_switch == "long_factor"
    assert phi4.rope_switch_len == 4096
    assert phi4.rope[0].get("original_max_position_embeddings") == 4096
    assert isinstance(phi4.rope[0].get("long_factor"), tuple)

    phi3 = real_arch("microsoft/Phi-3-mini-4k-instruct")
    assert set(phi3.windows) == {2047}
    assert not phi3.tied_head
    assert phi3.rope_switch is None

    llama = real_arch("unsloth/Llama-3.2-1B-Instruct")
    assert {spec.rope_type for spec in llama.rope} == {"llama3"}
    assert llama.biases == ()
    assert set(llama.layer_types) == {"full_attention"}

    repo_id, overrides = SYNTHETIC_CASES["qwen3-yarn"]
    yarn = real_arch(repo_id, **overrides)
    assert yarn.rope_switch is None
    assert yarn.rope[0].get("factor") == 4.0

    repo_id, overrides = SYNTHETIC_CASES["llama-dynamic"]
    dynamic = real_arch(repo_id, **overrides)
    assert dynamic.rope_switch == "raise"
    assert dynamic.rope_switch_len == 64


@pytest.mark.parametrize(
    ("repo_id", "error", "match"),
    [
        ("unsloth/gemma-2b-it", UnsupportedArchitectureError, "'gemma'.*system role"),
        (
            "unsloth/gemma-2-2b-it",
            UnsupportedArchitectureError,
            "'gemma2'.*system role",
        ),
        (
            "mistralai/Ministral-3-3B-Instruct-2512",
            UnsupportedCheckpointError,
            "pre-quantised.*fp8",
        ),
        ("Qwen/Qwen3-0.6B-FP8", UnsupportedCheckpointError, "fp8"),
    ],
)
def test_rejects_real_checkpoints(repo_id: str, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        check_config(AutoConfig.from_pretrained(repo_id))


def test_accepts_moe_and_ministral3() -> None:
    # Enabled after passing the forward and generation parity tests.
    for config in (Qwen3MoeConfig(), MixtralConfig()):
        check_config(config)
    check_config(
        AutoConfig.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512-BF16")
    )


def _llama(**kwargs) -> LlamaConfig:
    return LlamaConfig(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("config", "error", "match"),
    [
        (
            LlavaConfig(text_config=_llama().to_dict()),
            UnsupportedArchitectureError,
            "'llama'.*inside a 'llava' model",
        ),
        (
            _llama(
                rope_parameters={
                    "rope_type": "proportional",
                    "rope_theta": 1e4,
                    "partial_rotary_factor": 1.0,
                }
            ),
            UnsupportedArchitectureError,
            "RoPE type 'proportional'",
        ),
        (
            Gemma3TextConfig(
                num_hidden_layers=2,
                layer_types=["sliding_attention", "full_attention"],
                rope_parameters={
                    "full_attention": {
                        "rope_type": "longrope",
                        "rope_theta": 1e6,
                        "short_factor": [1.0] * 128,
                        "long_factor": [2.0] * 128,
                        "original_max_position_embeddings": 4096,
                    },
                    "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
                },
            ),
            UnsupportedArchitectureError,
            "'longrope' on some layer types",
        ),
        (
            Gemma3TextConfig(num_hidden_layers=2, use_bidirectional_attention=True),
            UnsupportedArchitectureError,
            "Bidirectional",
        ),
        (
            Qwen2Config(
                num_hidden_layers=2,
                layer_types=["chunked_attention", "full_attention"],
            ),
            UnsupportedArchitectureError,
            "chunked_attention",
        ),
        (_llama(hidden_act="relu"), UnsupportedArchitectureError, "'relu'"),
        (
            _llama(partial_rotary_factor=0.5),
            UnsupportedArchitectureError,
            "Partial rotary",
        ),
        (
            _llama(per_layer_config={0: {"intermediate_size": 64}}),
            UnsupportedArchitectureError,
            "per-layer overrides",
        ),
        # The quantisation check comes first, whatever else is unsupported.
        (
            Qwen3MoeConfig(quantization_config={"quant_method": "awq", "bits": 4}),
            UnsupportedCheckpointError,
            "awq",
        ),
    ],
    ids=[
        "llava",
        "proportional",
        "partial-longrope",
        "bidirectional",
        "chunked",
        "relu",
        "partial-rotary-llama",
        "heterogeneous",
        "quantised-first",
    ],
)
def test_rejects_configs(config, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        check_config(config)


def test_accepts_dynamic_and_yarn() -> None:
    check_config(_llama(rope_parameters={"rope_type": "dynamic", "factor": 2.0}))
    check_config(
        _llama(
            rope_parameters={
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 1024,
            }
        )
    )


def _write_checkpoint(directory, config: dict, tensors: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})


def test_quantised_checkpoint_rejected_before_shard_download(
    tmp_path,
    monkeypatch,
) -> None:
    # A text prefix could not be detected in these tensors, so the error must come
    # from the configuration check, before any shard is downloaded.
    config = _llama().to_dict()
    config["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3"}
    _write_checkpoint(
        tmp_path / "repo",
        config,
        {"weight_scale": np.ones(4, dtype=np.float32)},
    )
    hub = FakeHub(tmp_path / "cache", {"org/quantised": tmp_path / "repo"})
    hub.install(monkeypatch)

    ckpt = weights.resolve_checkpoint("org/quantised", None)
    assert ckpt.shard_files == ("model.safetensors",)
    with pytest.raises(UnsupportedCheckpointError, match="fp8"):
        check_config(ckpt.config)
    assert [name for _, name in hub.downloads] == ["config.json"]


def test_quantised_tensors_rejected() -> None:
    config = _llama(tie_word_embeddings=True)
    base = {
        "model.layers.0.self_attn.o_proj.weight": TensorInfo("a", (64, 64), "BF16"),
        "model.embed_tokens.weight": TensorInfo("a", (256, 64), "BF16"),
    }

    for extra, match in [
        (
            {
                "model.layers.0.mlp.down_proj.weight": TensorInfo(
                    "a", (64, 128), "F8_E4M3"
                )
            },
            "F8_E4M3",
        ),
        ({"lm_head.weight": TensorInfo("a", (256, 64), "I8")}, "I8"),
        (
            {
                "model.layers.0.mlp.down_proj.weight_scale": TensorInfo(
                    "a", (64,), "F32"
                )
            },
            "weight_scale",
        ),
        (
            {
                "model.layers.0.self_attn.q_proj.qweight": TensorInfo(
                    "a", (8, 64), "I32"
                )
            },
            "qweight",
        ),
        (
            {"model.layers.0.self_attn.q_proj.g_idx": TensorInfo("a", (64,), "I32")},
            "g_idx",
        ),
    ]:
        with pytest.raises(UnsupportedCheckpointError, match=match):
            TensorIndex.from_infos("", {**base, **extra}, config)

    # Tensors outside the text model (vision tower, projector) are not checked.
    vision = {"vision_tower.patch.weight": TensorInfo("a", (4, 4), "F8_E4M3")}
    assert TensorIndex.from_infos("", {**base, **vision}, config).prefix == "model."


def test_text_prefix_must_be_unique() -> None:
    config = _llama()
    o_proj = TensorInfo("a", (64, 64), "BF16")

    with pytest.raises(UnsupportedCheckpointError, match="found 0"):
        TensorIndex.from_infos("", {"transformer.h.0.attn.weight": o_proj}, config)

    with pytest.raises(UnsupportedCheckpointError, match="found 2"):
        TensorIndex.from_infos(
            "",
            {
                "model.layers.0.self_attn.o_proj.weight": o_proj,
                "language_model.model.layers.0.self_attn.o_proj.weight": o_proj,
            },
            config,
        )


def test_unknown_and_remote_code_configs(tmp_path) -> None:
    unknown = tmp_path / "unknown"
    _write_checkpoint(unknown, {"model_type": "no-such-model"}, {})
    with pytest.raises(UnsupportedArchitectureError):
        weights.resolve_checkpoint(str(unknown), None)

    remote = tmp_path / "remote"
    _write_checkpoint(
        remote,
        {
            "model_type": "custom-model",
            "auto_map": {"AutoConfig": "configuration_custom.CustomConfig"},
        },
        {},
    )
    with pytest.raises(UnsupportedArchitectureError):
        weights.resolve_checkpoint(str(remote), None)


@pytest.fixture
def planned_moe(monkeypatch) -> None:
    """Accepts the planned mixture-of-experts architectures."""

    for model_type in ("qwen3_moe", "mixtral"):
        monkeypatch.setitem(
            arch_module.SUPPORTED_ARCHITECTURES, model_type, (model_type,)
        )


@pytest.mark.usefixtures("planned_moe")
def test_moe_fields() -> None:
    qwen3_moe = real_arch("Qwen/Qwen3-30B-A3B")
    assert qwen3_moe.is_moe
    assert (qwen3_moe.num_experts, qwen3_moe.moe_intermediate_size) == (128, 768)
    assert qwen3_moe.num_experts_per_tok == 8
    assert qwen3_moe.norm_topk_prob is True
    assert qwen3_moe.has_qk_norm

    mixtral = real_arch("mistralai/Mixtral-8x7B-Instruct-v0.1")
    assert mixtral.is_moe
    assert (mixtral.num_experts, mixtral.moe_intermediate_size) == (8, 14336)
    assert mixtral.num_experts_per_tok == 2
    assert set(mixtral.windows) == {NO_WINDOW}

    assert not real_arch("Qwen/Qwen3-0.6B").is_moe


@pytest.mark.usefixtures("planned_moe")
@pytest.mark.parametrize(
    "overrides",
    [{"mlp_only_layers": [0]}, {"decoder_sparse_step": 2}, {"num_experts": 0}],
    ids=["mlp_only_layers", "decoder_sparse_step", "no-experts"],
)
def test_rejects_qwen3_moe_with_dense_layers(overrides: dict) -> None:
    check_config(Qwen3MoeConfig(num_hidden_layers=4))
    with pytest.raises(UnsupportedArchitectureError, match="dense layers"):
        check_config(Qwen3MoeConfig(num_hidden_layers=4, **overrides))
