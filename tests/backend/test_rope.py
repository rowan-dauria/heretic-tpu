# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""RoPE tables against the rotary embedding modules of transformers."""

import importlib
import inspect

import numpy as np
import pytest

from heretic_tpu.backend.rope import build_tables
from tests.backend.tiny import real_arch, real_config

torch = pytest.importorskip("torch")

REAL_MODELS = [
    "unsloth/gemma-3-4b-it",
    "unsloth/gemma-3-1b-it",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-4-mini-instruct",
    "unsloth/Llama-3.2-1B-Instruct",
]

# RoPE variants that no released checkpoint of a supported family uses.
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
    "qwen3-yarn-options": (
        "Qwen/Qwen3-0.6B",
        {
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 10240,
                "rope_theta": 1000000,
                "beta_fast": 16,
                "beta_slow": 2,
                "truncate": False,
                "attention_factor": 1.25,
            }
        },
    ),
    "qwen3-yarn-mscale": (
        "Qwen/Qwen3-0.6B",
        {
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": 40.0,
                "original_max_position_embeddings": 4096,
                "rope_theta": 10000.0,
                "mscale": 1.0,
                "mscale_all_dim": 0.707,
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
    "llama-linear": (
        "unsloth/Llama-3.2-1B-Instruct",
        {"rope_parameters": {"rope_type": "linear", "factor": 4.0, "rope_theta": 1e4}},
    ),
    "gemma3-yarn-per-layer-type": (
        "unsloth/gemma-3-4b-it",
        {
            "rope_parameters": {
                "full_attention": {
                    "rope_type": "yarn",
                    "factor": 8.0,
                    "original_max_position_embeddings": 16384,
                    "rope_theta": 1000000.0,
                    # Ignored by transformers for per-layer-type parameters.
                    "truncate": False,
                },
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            }
        },
    ),
    "phi4-longrope-attention-factor": (
        "microsoft/Phi-4-mini-instruct",
        {"rope_parameters": None},
    ),
    "phi3-partial-default": (
        "microsoft/Phi-3-mini-4k-instruct",
        {
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
            }
        },
    ),
}


def _overrides(case: str) -> tuple[str, dict]:
    repo_id, overrides = SYNTHETIC_CASES[case]
    if case == "phi4-longrope-attention-factor":
        parameters = dict(real_config(repo_id).rope_parameters)
        parameters.pop("type")
        parameters["attention_factor"] = 1.3
        parameters["factor"] = 16.0
        overrides = {"rope_parameters": parameters}
    return repo_id, overrides


def _rotary_class(text_config) -> type:
    from transformers.models.auto.modeling_auto import MODEL_MAPPING

    module = importlib.import_module(MODEL_MAPPING[type(text_config)].__module__)
    (rotary_class,) = [
        cls
        for name, cls in inspect.getmembers(module, inspect.isclass)
        if name.endswith("RotaryEmbedding") and cls.__module__ == module.__name__
    ]
    return rotary_class


def _reference(text_config, layer_type: str | None):
    """(rotary module, inv_freq, attention scaling) for a layer type."""

    rotary = _rotary_class(text_config)(config=text_config)
    if isinstance(rotary.rope_type, dict):
        return (
            rotary,
            getattr(rotary, f"{layer_type}_inv_freq").numpy(),
            getattr(rotary, f"{layer_type}_attention_scaling"),
        )
    return rotary, rotary.inv_freq.numpy(), rotary.attention_scaling


def _check_tables(repo_id: str, overrides: dict) -> None:
    config = real_config(repo_id, **overrides)
    text_config = config.get_text_config()
    arch = real_arch(repo_id, **overrides)
    tables = build_tables(arch)

    assert tables.inv_freq.dtype == np.float32
    assert tables.inv_freq.shape == (arch.num_hidden_layers, arch.rot // 2)
    assert tables.attention_scaling.dtype == np.float32
    assert tables.attention_scaling.shape == (arch.num_hidden_layers,)

    per_layer_type = isinstance(text_config.rope_parameters.get("full_attention"), dict)

    for layer, layer_type in enumerate(arch.layer_types):
        rotary, inv_freq, attention_scaling = _reference(
            text_config, layer_type if per_layer_type else None
        )

        # Both sides are float32. PyTorch's power function is accurate to about one
        # ulp, while the port's is correctly rounded.
        np.testing.assert_array_max_ulp(tables.inv_freq[layer], inv_freq, maxulp=1)
        assert tables.attention_scaling[layer] == np.float32(attention_scaling)

    if arch.rope_switch == "long_factor":
        # Positions beyond the original context length switch transformers to the
        # long factors.
        x = torch.zeros(1, 1, 1)
        positions = torch.arange(arch.rope_switch_len + 1)[None]
        rotary(x, positions)
        for layer in range(arch.num_hidden_layers):
            np.testing.assert_array_max_ulp(
                tables.long_inv_freq[layer],
                rotary.inv_freq.numpy(),
                maxulp=1,
            )
    else:
        assert tables.long_inv_freq is None


def _check_cos_sin(repo_id: str, overrides: dict) -> None:
    """cos/sin computed from the tables equal the rotary module's output."""

    config = real_config(repo_id, **overrides)
    text_config = config.get_text_config()
    arch = real_arch(repo_id, **overrides)
    tables = build_tables(arch)

    # Transformers switches when max(position) + 1 exceeds the switch length,
    # so positions up to the switch length minus one use the short tables.
    cases = [(arch.rope_switch_len - 1 if arch.rope_switch else 8191, False)]
    if arch.rope_switch == "long_factor":
        cases.append((2 * arch.rope_switch_len, True))

    for limit, long in cases:
        positions = np.unique(np.linspace(0, limit, 97).astype(np.int64))

        for layer in sorted({0, arch.num_hidden_layers - 1}):
            layer_type = arch.layer_types[layer]
            rotary = _rotary_class(text_config)(config=text_config)
            kwargs = {}
            if isinstance(rotary.rope_type, dict):
                kwargs["layer_type"] = layer_type
            cos, sin = rotary(
                torch.zeros(1, 1, 1), torch.tensor(positions)[None], **kwargs
            )

            inv_freq = (tables.long_inv_freq if long else tables.inv_freq)[layer]
            freqs = positions[:, None].astype(np.float32) * inv_freq
            emb = np.concatenate([freqs, freqs], axis=-1)
            scaling = tables.attention_scaling[layer]

            # An inverse frequency that differs by an ulp (see _check_tables) shifts
            # the angle by up to position * ulp.
            spacing = np.spacing(np.concatenate([inv_freq, inv_freq]))
            tolerance = 2e-6 + 2 * scaling * positions[:, None] * spacing
            assert np.all(np.abs(np.cos(emb) * scaling - cos[0].numpy()) <= tolerance)
            assert np.all(np.abs(np.sin(emb) * scaling - sin[0].numpy()) <= tolerance)


@pytest.mark.parametrize("repo_id", REAL_MODELS)
def test_tables_match_transformers(repo_id: str) -> None:
    _check_tables(repo_id, {})


@pytest.mark.parametrize("case", SYNTHETIC_CASES)
def test_synthetic_tables_match_transformers(case: str) -> None:
    _check_tables(*_overrides(case))


@pytest.mark.parametrize(
    "repo_id",
    [
        "unsloth/gemma-3-4b-it",
        "microsoft/Phi-4-mini-instruct",
        "unsloth/Llama-3.2-1B-Instruct",
    ],
)
def test_cos_sin_match_transformers(repo_id: str) -> None:
    _check_cos_sin(repo_id, {})


@pytest.mark.parametrize("case", ["qwen3-yarn", "llama-dynamic"])
def test_synthetic_cos_sin_match_transformers(case: str) -> None:
    _check_cos_sin(*_overrides(case))


def test_gemma3_scales_full_attention_layers_only() -> None:
    arch = real_arch("unsloth/gemma-3-4b-it")
    tables = build_tables(arch)

    full = arch.layer_types.index("full_attention")
    sliding = arch.layer_types.index("sliding_attention")
    assert arch.rope[full].rope_type == "linear"
    assert arch.rope[sliding].rope_type == "default"

    # Linear factor 8 on base 1e6 for full attention, plain base 1e4 for sliding.
    exponents = np.arange(0, arch.rot, 2, dtype=np.float64) / arch.rot
    np.testing.assert_allclose(tables.inv_freq[full], 1e6**-exponents / 8, rtol=1e-6)
    np.testing.assert_allclose(tables.inv_freq[sliding], 1e4**-exponents, rtol=1e-6)


def test_longrope_attention_factor() -> None:
    # sqrt(1 + ln(131072 / 4096) / ln(4096)) for Phi-3.5-mini and Phi-4-mini.
    for repo_id in ("microsoft/Phi-3.5-mini-instruct", "microsoft/Phi-4-mini-instruct"):
        tables = build_tables(real_arch(repo_id))
        np.testing.assert_allclose(tables.attention_scaling, 1.1902381, rtol=1e-7)
