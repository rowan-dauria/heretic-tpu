# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
Parity of the JAX decoder with transformers (float32 on CPU, eager attention and
eager experts) on tiny random-weight checkpoints built from the configurations of
real models: logits at every position, hidden states and module I/O of left-padded
mixed-length batches, the LoRA path against PEFT, prefill plus decode steps against
the full forward, RoPE switching, blocked attention, the mixture-of-experts
combination order in bfloat16, tensor parallelism on a forced 4-device CPU mesh, and
the memory behaviour of the KV cache. Marked `tpu`: real models in bfloat16 against
the float32 CPU reference.
"""

import dataclasses
import functools
import math
import os
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from heretic_tpu.backend import layers, transformer, weights
from heretic_tpu.backend.arch import NO_WINDOW, ArchConfig, check_config
from heretic_tpu.backend.sharding import choose_plan
from tests.backend.tiny import (
    REAL_MODELS,
    build_checkpoint,
    load_reference,
    tiny_config,
)

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]

requires_tpu = pytest.mark.skipif(
    jax.default_backend() != "tpu", reason="requires a TPU"
)

# A sliding window shorter than the test prompts.
WINDOW = 8

# The Phi-3 case switches to the long RoPE factors above this sequence length.
PHI3_SWITCH = 20

# Float32 tolerances. Differences come from summation order and from the elementary
# functions (exp, rsqrt, cos, ...), which differ by an ulp or so between libraries.
RTOL = 1e-4
ATOL = 1e-4


def _case(name: str) -> tuple[Any, dict[str, Any]]:
    """The tiny configuration of a test case, and the build_checkpoint arguments."""

    llama = REAL_MODELS["llama"]

    if name in ("qwen3_moe", "mixtral"):
        # 4 experts of width 32, 2 per token.
        return tiny_config(REAL_MODELS[name]), {}
    if name == "ministral3":
        # A short original context, so that the Llama 4 query scaling (which grows
        # every original_max_position_embeddings positions) is active in the tests.
        repo_id = REAL_MODELS["ministral3"]
        rope = dict(tiny_config(repo_id).get_text_config().rope_parameters)
        rope["original_max_position_embeddings"] = 6
        return tiny_config(repo_id, rope_parameters=rope), {}

    if name == "llama":
        # llama3 RoPE; tied, without lm_head.
        return tiny_config(llama), {}
    if name == "llama-biases":
        # Biases on every projection, including attn.o_proj and mlp.down_proj.
        config = tiny_config(llama, attention_bias=True, mlp_bias=True)
        return config, {"tie_word_embeddings": False}
    if name == "llama-gelu":
        return tiny_config(llama, hidden_act="gelu"), {}
    if name == "llama-dynamic":
        # Tested below the threshold (max_position_embeddings) only.
        rope = {"rope_type": "dynamic", "factor": 2.0, "rope_theta": 500000.0}
        return tiny_config(llama, rope_parameters=rope, max_position_embeddings=64), {}
    if name in ("mistral", "phi3_4k"):
        return tiny_config(REAL_MODELS[name], sliding_window=WINDOW), {}
    if name == "qwen2":
        # use_sliding_window is false, so the window is ignored.
        return tiny_config(REAL_MODELS["qwen2"], sliding_window=WINDOW), {}
    if name == "qwen3":
        return tiny_config(REAL_MODELS["qwen3"]), {}
    if name == "qwen3-yarn":
        rope = {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
        }
        return tiny_config(REAL_MODELS["qwen3"], rope_parameters=rope), {}
    if name == "gemma3_text":
        # Linear scaling on full-attention layers only, as in the released 4B to 27B
        # checkpoints, and soft-capping small enough to be active.
        rope = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "linear",
                "factor": 8.0,
                "rope_theta": 1000000.0,
            },
        }
        config = tiny_config(
            REAL_MODELS["gemma3_text"],
            sliding_window=WINDOW,
            rope_parameters=rope,
            attn_logit_softcapping=1.0,
            final_logit_softcapping=3.0,
        )
        return config, {}
    if name == "gemma3":
        # The wrapper ignores final_logit_softcapping, as transformers does.
        config = tiny_config(
            REAL_MODELS["gemma3"],
            sliding_window=WINDOW,
            final_logit_softcapping=3.0,
        )
        return config, {}
    if name == "mistral3":
        return tiny_config(REAL_MODELS["mistral3"]), {}
    if name == "mistral3-untied":
        # An untied head and no top-level tie key (Mistral3Config defaults to tying).
        return tiny_config(REAL_MODELS["mistral3"]), {
            "tie_word_embeddings": False,
            "edit_config": lambda config: config.pop("tie_word_embeddings"),
        }
    if name == "phi3":
        # Shaped like Phi-4-mini (partial rotary, longrope, tied), with a short
        # original context so that tests cross the long-factor switch.
        repo_id = REAL_MODELS["phi3"]
        rope = dict(tiny_config(repo_id).get_text_config().rope_parameters)
        rope["original_max_position_embeddings"] = PHI3_SWITCH
        config = tiny_config(
            repo_id,
            rope_parameters=rope,
            max_position_embeddings=4 * PHI3_SWITCH,
            original_max_position_embeddings=PHI3_SWITCH,
        )
        return config, {}
    raise ValueError(name)


def _build(name: str, directory: Path) -> Path:
    """Builds the checkpoint of a test case."""

    if name == "qwen3_moe-fused":
        # Fused experts, as transformers 5 holds them in memory.
        from safetensors.torch import save_file

        build_checkpoint(tiny_config(REAL_MODELS["qwen3_moe"]), directory)
        state = load_reference(directory).state_dict()
        assert "model.layers.0.mlp.experts.gate_up_proj" in state
        save_file(
            {key: value.contiguous() for key, value in state.items()},
            str(directory / "model.safetensors"),
            metadata={"format": "pt"},
        )
        return directory

    config, kwargs = _case(name)
    return build_checkpoint(config, directory, **kwargs)


CASES = [
    "llama",
    "llama-biases",
    "llama-gelu",
    "llama-dynamic",
    "mistral",
    "qwen2",
    "qwen3",
    "qwen3-yarn",
    "gemma3_text",
    "gemma3",
    "mistral3",
    "mistral3-untied",
    "phi3",
    "phi3_4k",
    "qwen3_moe",
    "qwen3_moe-fused",
    "mixtral",
    "ministral3",
]


# Transformers' Gemma 3 never passes attn_logit_softcapping to its attention function
# (Gemma 2 does). The port applies it (see "Divergences from upstream" in
# docs/DESIGN.md), so references for soft-capped configurations run transformers' own
# Gemma 3 eager attention with the cap passed through.
SOFTCAPPED_EAGER = "eager_softcap"


def _softcapped_eager_attention(module, query, key, value, attention_mask, **kwargs):
    from transformers.models.gemma3 import modeling_gemma3

    return modeling_gemma3.eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        softcap=module.attn_logit_softcapping,
        **kwargs,
    )


def _load_reference(directory: Path, arch: ArchConfig) -> Any:
    model = load_reference(directory)
    if arch.is_moe:
        # The numerics the port follows (transformers defaults to grouped matmuls).
        model.set_experts_implementation("eager")
    if arch.attn_logit_softcapping is not None:
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface, eager_mask

        AttentionInterface.register(SOFTCAPPED_EAGER, _softcapped_eager_attention)
        AttentionMaskInterface.register(SOFTCAPPED_EAGER, eager_mask)
        model.set_attn_implementation(SOFTCAPPED_EAGER)
    return model


class Model(NamedTuple):
    directory: Path
    arch: ArchConfig
    params: weights.Params
    reference: Any


def _load(directory: Path, dtype: Any = np.float32) -> tuple[ArchConfig, Any]:
    ckpt = weights.resolve_checkpoint(str(directory), None)
    check_config(ckpt.config)
    tensors = weights.build_tensor_index(ckpt)
    arch = ArchConfig.from_hf(ckpt.config, ckpt.raw_config, tensors)
    plan = choose_plan(arch, dtype, "single")
    return arch, weights.load_params(ckpt, tensors, arch, dtype, plan)


@pytest.fixture(scope="module")
def models(tmp_path_factory) -> Callable[[str], Model]:
    """Builds and loads each case on first use."""

    root = tmp_path_factory.mktemp("forward")
    cache: dict[str, Model] = {}

    def get(name: str) -> Model:
        if name not in cache:
            directory = _build(name, root / name)
            arch, params = _load(directory)
            reference = _load_reference(directory, arch)
            cache[name] = Model(directory, arch, params, reference)
        return cache[name]

    return get


@functools.cache
def _jitted(fn: Callable, arch: ArchConfig, **static: Any) -> Callable:
    return jax.jit(functools.partial(fn, arch=arch, **static))


def _batch(
    arch: ArchConfig,
    lengths: list[int],
    length: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """A left-padded batch of random tokens (padding id 0)."""

    rng = np.random.default_rng(seed)
    tokens = rng.integers(3, arch.vocab_size, (len(lengths), length), dtype=np.int32)
    mask = np.arange(length)[None, :] >= length - np.array(lengths)[:, None]
    return np.where(mask, tokens, 0).astype(np.int32), mask


def _long_factor(arch: ArchConfig, mask: np.ndarray) -> np.ndarray | None:
    """
    The per-row long-factor flags that reproduce one transformers call on the whole
    batch, which decides from the longest row.
    """

    if arch.rope_switch != "long_factor":
        return None
    long = mask.sum(axis=1).max() > arch.rope_switch_len
    return np.full(mask.shape[0], long)


def _random_lora(arch: ArchConfig, rank: int, seed: int = 1) -> transformer.Lora:
    """Random non-zero adapters, large enough to change the outputs noticeably."""

    rng = np.random.default_rng(seed)
    lora = {}
    for component in weights.abliterable_components(arch):
        d_out, d_in = weights.component_shape(arch, component)
        depth = arch.num_hidden_layers
        a = rng.standard_normal((depth, 1, rank, d_in)) / math.sqrt(d_in)
        b = rng.standard_normal((depth, 1, d_out, rank)) / math.sqrt(rank)
        lora[component] = (a.astype(np.float32), b.astype(np.float32))
    return lora


class Reference(NamedTuple):
    # [B, T, V]
    logits: np.ndarray

    # [B, L + 1, D] at the last column.
    hidden: np.ndarray

    # Component -> (inputs [B, L, 1, d_in], outputs [B, L, 1, d_out]) at the last
    # column, from forward hooks as upstream registers them.
    module_io: dict[str, tuple[np.ndarray, np.ndarray]]


def _reference(
    model: Any,
    arch: ArchConfig,
    tokens: np.ndarray,
    mask: np.ndarray,
    root: Any = None,
) -> Reference:
    """
    Runs the transformers model on a left-padded batch with the position ids of
    generate. `root` is the module that module paths are relative to (the base model
    of a PeftModel).
    """

    root = model if root is None else root
    captured: dict[str, tuple[list, list]] = {}
    handles = []

    for component in weights.abliterable_components(arch):
        captured[component] = ([], [])
        for layer_index in range(arch.num_hidden_layers):
            path = weights.module_path(arch, component, layer_index, 0)

            def hook(module, inputs, output, component=component):
                captured[component][0].append(inputs[0][:, -1].detach().numpy())
                captured[component][1].append(output[:, -1].detach().numpy())

            handles.append(root.get_submodule(path).register_forward_hook(hook))

    positions = np.where(mask, np.cumsum(mask, axis=1) - 1, 0)
    try:
        with torch.no_grad():
            output = model(
                input_ids=torch.from_numpy(tokens).long(),
                attention_mask=torch.from_numpy(mask).long(),
                position_ids=torch.from_numpy(positions).long(),
                output_hidden_states=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    hidden = torch.stack([h[:, -1] for h in output.hidden_states], dim=1)
    return Reference(
        logits=output.logits.float().numpy(),
        hidden=hidden.float().numpy(),
        module_io={
            component: tuple(np.stack(arrays, axis=1)[:, :, None] for arrays in pair)
            for component, pair in captured.items()
        },
    )


def _assert_close(actual: Any, expected: np.ndarray, **kwargs: Any) -> None:
    actual = np.asarray(actual)
    assert actual.shape == expected.shape
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(
        actual, expected, **{"rtol": RTOL, "atol": ATOL, **kwargs}
    )


def _assert_captures(
    captures: transformer.Captures,
    reference: Reference,
) -> None:
    _assert_close(captures.logits, reference.logits[:, -1])
    _assert_close(captures.hidden, reference.hidden)
    assert set(captures.module_io) == set(reference.module_io)
    for component, (inputs, outputs) in reference.module_io.items():
        _assert_close(captures.module_io[component][0], inputs)
        _assert_close(captures.module_io[component][1], outputs)


@pytest.mark.parametrize("name", CASES)
def test_forward_matches_transformers(models, name: str) -> None:
    model = models(name)
    arch = model.arch
    tokens, mask = _batch(arch, [24, 19, 9, 1], 24)
    long_factor = _long_factor(arch, mask)
    reference = _reference(model.reference, arch, tokens, mask)

    # Every position, padding included: fully masked rows attend uniformly to every
    # key in both implementations.
    logits = _jitted(transformer.logits_all, arch)(
        model.params, None, tokens, mask, long_factor
    )
    _assert_close(logits, reference.logits)

    captures, cache = _jitted(transformer.prefill, arch, want=transformer.CAPTURES)(
        model.params, None, tokens, mask, long_factor
    )
    assert cache is None
    assert captures.logits.dtype == jnp.float32
    assert captures.hidden.shape == (4, arch.num_hidden_layers + 1, arch.hidden_size)
    _assert_captures(captures, reference)


def test_case_features(models) -> None:
    """The cases exercise what they are meant to."""

    assert models("llama").arch.tied_head
    assert models("llama").arch.rope[0].rope_type == "llama3"
    assert models("llama-biases").arch.biases == (
        "down_proj",
        "gate",
        "k",
        "o_proj",
        "q",
        "up",
        "v",
    )
    assert models("llama-gelu").arch.activation == "gelu"
    assert models("llama-dynamic").arch.rope_switch == "raise"
    assert set(models("mistral").arch.windows) == {WINDOW}
    assert models("qwen2").arch.biases == ("k", "q", "v")
    assert set(models("qwen2").arch.windows) == {NO_WINDOW}
    assert models("qwen3-yarn").arch.rope[0].rope_type == "yarn"
    assert models("qwen3-yarn").arch.attention_scale == 16**-0.5

    gemma3_text = models("gemma3_text").arch
    assert gemma3_text.windows == (WINDOW, NO_WINDOW, WINDOW)
    assert [spec.rope_type for spec in gemma3_text.rope] == [
        "default",
        "linear",
        "default",
    ]
    assert gemma3_text.final_logit_softcapping == 3.0

    gemma3 = models("gemma3").arch
    assert gemma3.multimodal
    assert gemma3.rope[1].get("factor") == 8.0
    assert gemma3.final_logit_softcapping is None

    assert models("mistral3").arch.tied_head
    assert not models("mistral3-untied").arch.tied_head

    ministral3 = models("ministral3").arch
    assert ministral3.model_type == "ministral3"
    assert ministral3.multimodal and ministral3.tied_head
    assert ministral3.rope[0].rope_type == "yarn"
    assert ministral3.rope[0].get("llama_4_scaling_beta") == 0.1

    moe = models("qwen3_moe").arch
    assert moe.is_moe and moe.num_experts_per_tok == 2 and moe.norm_topk_prob
    assert weights.abliterable_components(moe) == ["attn.o_proj"]
    assert models("mixtral").arch.norm_topk_prob is None

    phi3 = models("phi3").arch
    assert phi3.rot == 12 and phi3.head_dim == 16
    assert phi3.rope_switch == "long_factor"
    assert phi3.rope_switch_len == PHI3_SWITCH
    assert phi3.tied_head
    assert set(models("phi3_4k").arch.windows) == {WINDOW}


def test_soft_capping_is_active(models) -> None:
    model = models("gemma3_text")
    tokens, mask = _batch(model.arch, [24, 13], 24)
    uncapped = dataclasses.replace(
        model.arch, attn_logit_softcapping=None, final_logit_softcapping=None
    )
    attention_uncapped = dataclasses.replace(model.arch, attn_logit_softcapping=None)

    capped_logits = _jitted(transformer.logits_all, model.arch)(
        model.params, None, tokens, mask
    )
    for arch in (uncapped, attention_uncapped):
        logits = _jitted(transformer.logits_all, arch)(model.params, None, tokens, mask)
        assert np.abs(np.asarray(logits) - np.asarray(capped_logits))[mask].max() > 0.1


def test_llama_4_query_scaling_is_active(models) -> None:
    model = models("ministral3")
    tokens, mask = _batch(model.arch, [24, 13], 24)
    unscaled = dataclasses.replace(model.arch, model_type="mistral")

    scaled_logits, unscaled_logits = (
        np.asarray(
            _jitted(transformer.logits_all, arch)(model.params, None, tokens, mask)
        )
        for arch in (model.arch, unscaled)
    )
    assert np.abs(scaled_logits - unscaled_logits)[mask].max() > 0.01


@pytest.mark.parametrize("name", CASES)
def test_lora_matches_peft(models, name: str) -> None:
    peft = pytest.importorskip("peft")

    model = models(name)
    arch = model.arch
    rank = 3
    lora = _random_lora(arch, rank)

    # get_peft_model modifies the model it wraps, so wrap a fresh copy.
    paths = [
        weights.module_path(arch, component, layer_index, 0)
        for component in weights.abliterable_components(arch)
        for layer_index in range(arch.num_hidden_layers)
    ]
    peft_model = peft.get_peft_model(
        _load_reference(model.directory, arch),
        peft.LoraConfig(
            r=rank,
            lora_alpha=rank,
            lora_dropout=0.0,
            bias="none",
            target_modules=paths,
        ),
    )
    base = peft_model.base_model.model
    with torch.no_grad():
        for component, (a, b) in lora.items():
            for layer_index in range(arch.num_hidden_layers):
                module = base.get_submodule(
                    weights.module_path(arch, component, layer_index, 0)
                )
                module.lora_A["default"].weight.copy_(
                    torch.from_numpy(a[layer_index, 0])
                )
                module.lora_B["default"].weight.copy_(
                    torch.from_numpy(b[layer_index, 0])
                )

    tokens, mask = _batch(arch, [24, 16, 5], 24, seed=2)
    long_factor = _long_factor(arch, mask)
    reference = _reference(peft_model, arch, tokens, mask, root=base)

    logits = _jitted(transformer.logits_all, arch)(
        model.params, lora, tokens, mask, long_factor
    )
    _assert_close(logits, reference.logits)

    # The adapters change the model noticeably.
    plain = _jitted(transformer.logits_all, arch)(
        model.params, None, tokens, mask, long_factor
    )
    assert np.abs(np.asarray(plain) - reference.logits)[mask].max() > 0.1

    captures, _ = _jitted(transformer.prefill, arch, want=transformer.CAPTURES)(
        model.params, lora, tokens, mask, long_factor
    )
    _assert_captures(captures, reference)


@pytest.mark.parametrize("name", CASES)
def test_prefill_and_decode_match_full_forward(models, name: str) -> None:
    model = models(name)
    arch = model.arch
    prompt_len, steps = 16, 10
    cache_len = prompt_len + steps

    tokens, mask = _batch(arch, [16, 11, 5, 1], prompt_len, seed=3)
    continuation, _ = _batch(arch, [steps] * 4, steps, seed=4)
    full_tokens = np.concatenate([tokens, continuation], axis=1)
    full_mask = np.concatenate([mask, np.ones_like(continuation, dtype=bool)], axis=1)

    # The decode steps reproduce one forward call over the whole sequence,
    # so they use its RoPE factors throughout.
    long_factor = _long_factor(arch, full_mask)

    plain = _reference(model.reference, arch, full_tokens, full_mask).logits
    lora = _random_lora(arch, rank=2)
    adapted = np.asarray(
        _jitted(transformer.logits_all, arch)(
            model.params, lora, full_tokens, full_mask, long_factor
        )
    )

    prefill = _jitted(transformer.prefill, arch, cache_len=cache_len)
    decode = _jitted(transformer.decode_step, arch)

    for adapters, expected in ((None, plain), (lora, adapted)):
        captures, cache = prefill(model.params, adapters, tokens, mask, long_factor)
        shape = (
            arch.num_hidden_layers,
            4,
            cache_len,
            arch.num_key_value_heads,
            arch.head_dim,
        )
        assert cache[0].shape == shape and cache[1].shape == shape
        _assert_close(captures.logits, expected[:, prompt_len - 1])

        positions = mask.sum(axis=1).astype(np.int32)
        for step in range(steps):
            slot = prompt_len + step
            logits, cache = decode(
                model.params,
                adapters,
                cache,
                continuation[:, step],
                positions + step,
                np.int32(slot),
                full_mask,
                long_factor,
            )
            _assert_close(logits, expected[:, slot])

    # The re-prefill of long-RoPE generation: all cache slots, the unwritten ones
    # masked, with the logits at the newest slot. It rebuilds the cache that the
    # decode steps wrote (at the slots of real tokens; padding slots hold values
    # of fully masked rows, which attend to every slot of the call).
    newest = cache_len - 4
    written = full_mask & (np.arange(cache_len) <= newest)
    captures, rebuilt = _jitted(transformer.prefill, arch, cache_len=cache_len)(
        model.params,
        lora,
        np.where(written, full_tokens, 0),
        written,
        long_factor,
        column=np.int32(newest),
    )
    _assert_close(captures.logits, adapted[:, newest])
    for rebuilt_part, part in zip(rebuilt, cache):
        _assert_close(
            np.asarray(rebuilt_part)[:, written], np.asarray(part)[:, written]
        )


def test_long_factor_per_row(models) -> None:
    """
    Rows of two upstream batches share an engine batch: one batch's longest row is
    just above the switch length (long factors), the other's exactly at it (short).
    Each row equals transformers run on its own upstream batch.
    """

    model = models("phi3")
    arch = model.arch
    long_rows = [PHI3_SWITCH + 1, 7]
    short_rows = [PHI3_SWITCH, 12]
    length = 24

    tokens, mask = _batch(arch, long_rows + short_rows, length, seed=5)
    long_factor = np.array([True, True, False, False])
    logits = np.asarray(
        _jitted(transformer.logits_all, arch)(
            model.params, None, tokens, mask, long_factor
        )
    )

    for rows, longest in ((slice(0, 2), long_rows[0]), (slice(2, 4), short_rows[0])):
        # The upstream batch, left-padded to its own longest row.
        expected = _reference(
            model.reference,
            arch,
            tokens[rows, length - longest :],
            mask[rows, length - longest :],
        ).logits
        actual = logits[rows, length - longest :]
        real = mask[rows, length - longest :]
        _assert_close(actual[real], expected[real])

    # Both sides of the switch differ noticeably.
    short_everywhere = np.asarray(
        _jitted(transformer.logits_all, arch)(model.params, None, tokens, mask)
    )
    assert np.abs(short_everywhere[:2] - logits[:2])[mask[:2]].max() > 0.01


def test_long_prompt_matches_transformers(models) -> None:
    """Several query blocks, the last one padded, with windows and soft-capping."""

    model = models("gemma3_text")
    arch = model.arch
    length = 1100
    assert layers.query_block_size(length) == 512

    tokens, mask = _batch(arch, [length, 700], length, seed=6)
    reference = _reference(model.reference, arch, tokens, mask)
    logits = _jitted(transformer.logits_all, arch)(model.params, None, tokens, mask)
    _assert_close(logits, reference.logits)


def _eager_attention(q, k, v, kv_mask, window, scale, cap) -> np.ndarray:
    """
    Float64 eager attention with a materialised mask, written as transformers does
    (repeated key/value heads, additive masking). Query rows are independent, so the
    rows are processed in chunks only to bound memory.
    """

    q, k, v = (np.asarray(x, dtype=np.float64) for x in (q, k, v))
    groups = q.shape[2] // k.shape[2]
    k = np.repeat(k, groups, axis=2)
    v = np.repeat(v, groups, axis=2)
    key_slots = np.arange(q.shape[1])

    outputs = []
    for start in range(0, q.shape[1], 1024):
        query_slots = key_slots[start : start + 1024]
        scores = np.einsum("bqhd,bkhd->bhqk", q[:, query_slots], k) * scale
        if cap is not None:
            scores = np.tanh(scores / cap) * cap
        visible = (
            kv_mask[:, None, None, :]
            & (key_slots[None, :] <= query_slots[:, None])
            & (query_slots[:, None] - key_slots[None, :] < window)
        )
        scores = np.where(visible, scores, -1e300)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        outputs.append(np.einsum("bhqk,bkhd->bqhd", probs, v))

    out = np.concatenate(outputs, axis=1)
    return out.reshape(*out.shape[:2], -1)


@pytest.mark.parametrize(
    ("length", "window", "cap"),
    [(1100, NO_WINDOW, None), (1100, 37, 2.0), (9000, 300, None)],
)
def test_blocked_attention_matches_eager(length: int, window: int, cap) -> None:
    rng = np.random.default_rng(7)
    batch, heads, kv_heads, head_dim = 2, 4, 2, 8
    q = rng.standard_normal((batch, length, heads, head_dim)).astype(np.float32)
    k = rng.standard_normal((batch, length, kv_heads, head_dim)).astype(np.float32)
    v = rng.standard_normal((batch, length, kv_heads, head_dim)).astype(np.float32)
    kv_mask = np.arange(length)[None, :] >= np.array([[0], [length // 3]])

    attention = jax.jit(
        functools.partial(layers.attention, scale=head_dim**-0.5, cap=cap)
    )
    actual = np.asarray(attention(q, k, v, kv_mask, np.int32(window)))
    expected = _eager_attention(q, k, v, kv_mask, window, head_dim**-0.5, cap)

    # Fully masked padding rows give finite values; real rows match.
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual[kv_mask], expected[kv_mask], atol=2e-5)


def test_query_block_size() -> None:
    assert layers.query_block_size(32) == 32
    assert layers.query_block_size(512) == 512
    assert layers.query_block_size(8192) == 512
    assert layers.query_block_size(8193) == 256


@pytest.mark.parametrize("kind", ["qwen3_moe", "mixtral"])
def test_moe_combination_order_in_bfloat16(kind: str) -> None:
    """
    The routed MLP in bfloat16 equals transformers' eager expert loop bit for bit on
    inputs for which every matmul is exact in float32, so that only the roundings to
    bfloat16 remain, and on which combining the experts in top-k rank order instead
    of ascending expert index gives a different result.
    """

    from transformers import MixtralConfig, Qwen3MoeConfig
    from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeSparseMoeBlock,
    )

    hidden, experts, top_k, width, tokens = 32, 8, 3, 16, 256
    if kind == "qwen3_moe":
        config = Qwen3MoeConfig(
            hidden_size=hidden,
            num_experts=experts,
            num_experts_per_tok=top_k,
            moe_intermediate_size=width,
            norm_topk_prob=True,
        )
        block_class = Qwen3MoeSparseMoeBlock
    else:
        config = MixtralConfig(
            hidden_size=hidden,
            num_local_experts=experts,
            num_experts_per_tok=top_k,
            intermediate_size=width,
        )
        block_class = MixtralSparseMoeBlock
    config._experts_implementation = "eager"
    block = block_class(config).to(torch.bfloat16)

    # Coarse grids keep every product and sum of the matmuls exact in float32.
    # A constant input feature with a large gate weight keeps the gate outputs above
    # 8, where SiLU rounds to the identity in bfloat16 whatever its implementation.
    rng = np.random.default_rng(8)

    def grid(shape: tuple[int, ...], step: float, limit: int) -> np.ndarray:
        return (rng.integers(-limit, limit + 1, shape) * step).astype(np.float32)

    x = grid((tokens, hidden), 1 / 8, 8)
    x[:, 0] = 1
    router = grid((experts, hidden), 1 / 4, 8)
    gate = grid((experts, width, hidden), 1 / 32, 8)
    gate[:, :, 0] = 16
    up = grid((experts, width, hidden), 1 / 16, 16)
    down = grid((experts, hidden, width), 1 / 16, 8)

    # Tokens with a tie at the top-k boundary are dropped: their choice of expert is
    # an implementation detail of torch.topk (the port takes the lower index).
    logits = np.sort((x @ router.T).astype(ml_dtypes.bfloat16), axis=-1)[:, ::-1]
    x = x[logits[:, top_k - 1] != logits[:, top_k]]
    assert len(x) > tokens // 2

    inputs = torch.from_numpy(x).bfloat16()
    with torch.no_grad():
        block.gate.weight.copy_(torch.from_numpy(router))
        block.experts.gate_up_proj.copy_(
            torch.from_numpy(np.concatenate([gate, up], 1))
        )
        block.experts.down_proj.copy_(torch.from_numpy(down))
        expected = block(inputs[None])[0]

        # The same contributions, combined in rank order.
        _, rank_weights, rank_experts = block.gate(inputs)
        rank_order = torch.zeros_like(inputs)
        for rank in range(top_k):
            rank_order = rank_order + block.experts(
                inputs,
                rank_experts[:, rank : rank + 1],
                rank_weights[:, rank : rank + 1],
            )
    assert not torch.equal(rank_order, expected)

    moe = jax.jit(
        functools.partial(
            layers.routed_moe,
            top_k=top_k,
            normalise=True,
            float32_weights=kind == "mixtral",
            activation_name="silu",
        )
    )
    bfloat16 = ml_dtypes.bfloat16
    with jax.default_device(jax.devices("cpu")[0]):
        actual = moe(
            x.astype(bfloat16),
            router.astype(bfloat16),
            gate.astype(bfloat16),
            up.astype(bfloat16),
            down.astype(bfloat16),
        )
    np.testing.assert_array_equal(
        np.asarray(actual).astype(np.float32), expected.float().numpy()
    )


def _float32_dot_precisions(text: str) -> list[str]:
    """The precision of every dot_general with a float32 operand in StableHLO text."""

    import re

    precisions = []
    for line in text.splitlines():
        if "stablehlo.dot_general" not in line:
            continue
        operands = re.search(r": \((tensor<[^>]*>), (tensor<[^>]*>)\)", line)
        assert operands, line
        if not any(re.search(r"[<x]f32>$", t) for t in operands.groups()):
            continue
        precision = re.search(r"precision = \[(\w+), (\w+)\]", line)
        precisions.append(precision.group(0) if precision else "DEFAULT")
    return precisions


@pytest.mark.parametrize("dtype", [np.float32, ml_dtypes.bfloat16])
def test_lora_path_runs_at_highest_precision(models, dtype) -> None:
    arch = models("llama").arch
    params = weights.param_skeleton(arch, dtype)
    lora = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), _random_lora(arch, 4)
    )
    tokens = jax.ShapeDtypeStruct((2, 32), jnp.int32)
    mask = jax.ShapeDtypeStruct((2, 32), jnp.bool_)

    text = (
        _jitted(transformer.prefill, arch, want=transformer.CAPTURES, cache_len=48)
        .lower(params, lora, tokens, mask)
        .as_text()
    )
    precisions = _float32_dot_precisions(text)
    # The two LoRA products of each component (and, for float32 models, every
    # projection) have float32 operands.
    assert len(precisions) >= 4
    assert all(p == "precision = [HIGHEST, HIGHEST]" for p in precisions), precisions


def _memory(compiled: Any) -> Any:
    analysis = compiled.memory_analysis()
    if analysis is None:
        pytest.skip("no memory analysis on this backend")
    return analysis


def test_kv_cache_is_not_copied(models) -> None:
    """
    On a configuration whose cache dwarfs its activations, the cache costs one copy
    in prefill (built in its final shape from the scan outputs) and is updated in place
    by a decode step with a donated cache. A decode step may materialise the cache
    slice of the layer it runs (1/L of the cache), so the configuration has a
    realistic depth.
    """

    tiny = models("llama").arch
    depth = 16
    arch = dataclasses.replace(
        tiny,
        num_hidden_layers=depth,
        layer_types=tiny.layer_types[:1] * depth,
        windows=tiny.windows[:1] * depth,
        rope=tiny.rope[:1] * depth,
    )
    params = weights.param_skeleton(arch, np.float32)
    batch, prompt_len = 8, 32
    tokens = jax.ShapeDtypeStruct((batch, prompt_len), jnp.int32)
    mask = jax.ShapeDtypeStruct((batch, prompt_len), jnp.bool_)

    def cache_bytes(cache_len: int) -> int:
        return (
            2
            * arch.num_hidden_layers
            * batch
            * cache_len
            * arch.num_key_value_heads
            * arch.head_dim
            * 4
        )

    prefill_cost = {}
    decode_temp = {}
    for cache_len in (4096, 8192):
        compiled = (
            _jitted(transformer.prefill, arch, cache_len=cache_len)
            .lower(params, None, tokens, mask)
            .compile()
        )
        memory = _memory(compiled)
        prefill_cost[cache_len] = (
            memory.temp_size_in_bytes
            + memory.output_size_in_bytes
            - memory.alias_size_in_bytes
        )

        shape = (
            arch.num_hidden_layers,
            batch,
            cache_len,
            arch.num_key_value_heads,
            arch.head_dim,
        )
        cache = (jax.ShapeDtypeStruct(shape, jnp.float32),) * 2
        compiled = (
            jax.jit(
                functools.partial(transformer.decode_step, arch=arch),
                donate_argnums=(2,),
            )
            .lower(
                params,
                None,
                cache,
                jax.ShapeDtypeStruct((batch,), jnp.int32),
                jax.ShapeDtypeStruct((batch,), jnp.int32),
                jax.ShapeDtypeStruct((), jnp.int32),
                jax.ShapeDtypeStruct((batch, cache_len), jnp.bool_),
            )
            .compile()
        )
        memory = _memory(compiled)
        assert memory.alias_size_in_bytes == cache_bytes(cache_len)
        decode_temp[cache_len] = memory.temp_size_in_bytes

    cache_delta = cache_bytes(8192) - cache_bytes(4096)
    assert prefill_cost[8192] - prefill_cost[4096] <= 1.25 * cache_delta
    assert decode_temp[8192] - decode_temp[4096] <= 0.25 * cache_delta


# Runs the forward pass, captures, adapters and prefill plus decode steps under the
# "tensor" plan of a forced 4-device CPU mesh and compares them with the "single" plan.
_TENSOR_PARALLEL_SCRIPT = textwrap.dedent(
    """
    import functools
    import sys

    import jax
    import numpy as np

    from heretic_tpu.backend import transformer, weights
    from heretic_tpu.backend.arch import ArchConfig
    from heretic_tpu.backend.sharding import choose_plan

    assert len(jax.local_devices()) == 4

    def run(arch, plan, params, lora, tokens, mask):
        replicated = plan.replicated
        kv = plan.kv_cache_sharding
        if lora is not None:
            lora = {
                component: tuple(
                    jax.device_put(array, sharding)
                    for array, sharding in zip(lora[component], plan.lora_shardings(component))
                )
                for component in lora
            }

        outputs = {}
        logits = jax.jit(
            functools.partial(transformer.logits_all, arch=arch),
            out_shardings=replicated,
        )(params, lora, tokens, mask)
        outputs["logits"] = logits

        captures, _ = jax.jit(
            functools.partial(transformer.prefill, arch=arch, want=transformer.CAPTURES)
        )(params, lora, tokens, mask)
        outputs["hidden"] = captures.hidden
        for component, (inputs, module_outputs) in captures.module_io.items():
            outputs[component + "/in"] = inputs
            outputs[component + "/out"] = module_outputs

        prompt_len = tokens.shape[1] - 4
        prefill = jax.jit(
            functools.partial(transformer.prefill, arch=arch, cache_len=tokens.shape[1]),
            out_shardings=(transformer.Captures(replicated, None, None), (kv, kv)),
        )
        decode = jax.jit(
            functools.partial(transformer.decode_step, arch=arch),
            out_shardings=(replicated, (kv, kv)),
            donate_argnums=(2,),
        )
        captures, cache = prefill(params, lora, tokens[:, :prompt_len], mask[:, :prompt_len])
        assert cache[0].sharding == kv
        steps = [captures.logits]
        positions = mask[:, :prompt_len].sum(axis=1).astype(np.int32)
        for step in range(4):
            slot = prompt_len + step
            step_logits, cache = decode(
                params, lora, cache, tokens[:, slot], positions + step,
                np.int32(slot), mask,
            )
            steps.append(step_logits)
        outputs["decode"] = np.stack([np.asarray(x) for x in steps], axis=1)

        return {name: np.asarray(value) for name, value in outputs.items()}

    for directory in sys.argv[1:]:
        ckpt = weights.resolve_checkpoint(directory, None)
        tensors = weights.build_tensor_index(ckpt)
        arch = ArchConfig.from_hf(ckpt.config, ckpt.raw_config, tensors)

        rng = np.random.default_rng(0)
        length = 20
        tokens = rng.integers(3, arch.vocab_size, (3, length)).astype(np.int32)
        mask = np.arange(length)[None, :] >= np.array([[0], [5], [11]])
        tokens = np.where(mask, tokens, 0).astype(np.int32)
        lora = {}
        for component in weights.abliterable_components(arch):
            d_out, d_in = weights.component_shape(arch, component)
            lora[component] = (
                rng.standard_normal((arch.num_hidden_layers, 1, 2, d_in)).astype(np.float32) / d_in**0.5,
                rng.standard_normal((arch.num_hidden_layers, 1, d_out, 2)).astype(np.float32),
            )

        tensor_plan = choose_plan(arch, np.float32, "tensor")
        single_plan = choose_plan(arch, np.float32, "single")
        print(directory, tensor_plan.attention_sharded, tensor_plan.mlp_sharded)

        for adapters in (None, lora):
            results = []
            for plan in (single_plan, tensor_plan):
                params = weights.load_params(ckpt, tensors, arch, np.float32, plan)
                results.append(run(arch, plan, params, adapters, tokens, mask))
                del params
            # Sharded contractions sum partial results in another order.
            for name, expected in results[0].items():
                np.testing.assert_allclose(
                    results[1][name], expected, rtol=1e-4, atol=1e-4, err_msg=name
                )
    """
)


def test_tensor_parallel_forward_on_four_cpu_devices(tmp_path) -> None:
    checkpoints = [
        # 8 query and 4 key/value heads: attention and the MLP are sharded.
        build_checkpoint(
            tiny_config(
                REAL_MODELS["llama"],
                num_attention_heads=8,
                num_key_value_heads=4,
                attention_bias=True,
                mlp_bias=True,
            ),
            tmp_path / "attention",
            tie_word_embeddings=False,
        ),
        # 2 key/value heads: attention is replicated, the MLP is sharded.
        build_checkpoint(
            tiny_config(REAL_MODELS["gemma3"], sliding_window=WINDOW),
            tmp_path / "gemma3",
        ),
        # Experts of width 32, sharded along it.
        build_checkpoint(tiny_config(REAL_MODELS["qwen3_moe"]), tmp_path / "moe"),
    ]

    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        "PYTHONPATH": os.pathsep.join(
            [str(REPO_ROOT / "src"), str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]
        ),
    }
    result = subprocess.run(
        [sys.executable, "-c", _TENSOR_PARALLEL_SCRIPT, *map(str, checkpoints)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert [line.split()[1:] for line in result.stdout.splitlines()] == [
        ["True", "True"],
        ["False", "True"],
        ["False", "True"],
    ]


# A real model on the TPU in bfloat16, against the float32 CPU reference.
# Real models in bfloat16 on the TPU, against the float32 CPU reference. The first is
# the primary case; the others cover q/k norms (Qwen3), sandwich norms, windows and
# embedding scaling (Gemma 3) and llama3 RoPE with a tied head (Llama 3.2).
TPU_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "unsloth/gemma-3-1b-it",
    "unsloth/Llama-3.2-1B-Instruct",
]

CHAT_PROMPTS = [
    "What is 1+1?",
    "Write a haiku about the sea.",
    "Explain the difference between a list and a tuple in Python.",
    "Who wrote Pride and Prejudice?",
    "Give me three tips for staying focused while studying.",
    "Translate 'good morning' into French, German and Spanish.",
    "What causes the seasons on Earth?",
    "Summarise the plot of Hamlet in two sentences.",
]


@pytest.mark.tpu
@pytest.mark.slow
@requires_tpu
@pytest.mark.parametrize("repo_id", TPU_MODELS)
def test_bfloat16_on_tpu_matches_float32_reference(repo_id: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = weights.resolve_checkpoint(repo_id, None)
    check_config(ckpt.config)
    weights.fetch_shards(ckpt)
    tensors = weights.build_tensor_index(ckpt)
    arch = ArchConfig.from_hf(ckpt.config, ckpt.raw_config, tensors)
    dtype = ml_dtypes.bfloat16
    plan = choose_plan(arch, dtype, "single")
    params = weights.load_params(ckpt, tensors, arch, dtype, plan)

    # Padding as upstream configures the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=ckpt.sha)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in CHAT_PROMPTS
    ]
    encoded = tokenizer(texts, padding=True, return_tensors="np")
    tokens = encoded["input_ids"].astype(np.int32)
    mask = encoded["attention_mask"].astype(bool)

    # The engine's prompt bucket: a power of two of at least 32.
    bucket = max(32, 1 << (tokens.shape[1] - 1).bit_length())
    padding = ((0, 0), (bucket - tokens.shape[1], 0))
    bucket_tokens = np.pad(tokens, padding, constant_values=tokenizer.pad_token_id)
    bucket_mask = np.pad(mask, padding)

    prefill = jax.jit(functools.partial(transformer.prefill, arch=arch))
    start = time.perf_counter()
    compiled = prefill.lower(params, None, bucket_tokens, bucket_mask).compile()
    compile_seconds = time.perf_counter() - start
    captures, _ = compiled(params, None, bucket_tokens, bucket_mask)
    logits = np.asarray(captures.logits)

    start = time.perf_counter()
    for _ in range(5):
        jax.block_until_ready(compiled(params, None, bucket_tokens, bucket_mask))
    run_seconds = (time.perf_counter() - start) / 5

    def reference_logits(torch_dtype: Any) -> np.ndarray:
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            revision=ckpt.sha,
            dtype=torch_dtype,
            attn_implementation="eager",
        ).eval()
        positions = np.where(mask, np.cumsum(mask, axis=1) - 1, 0)
        with torch.no_grad():
            output = model(
                input_ids=torch.from_numpy(tokens).long(),
                attention_mask=torch.from_numpy(mask).long(),
                position_ids=torch.from_numpy(positions).long(),
            )
        return output.logits[:, -1].float().numpy()

    def kl_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> np.ndarray:
        log_p = jax.nn.log_softmax(p_logits, axis=-1)
        log_q = jax.nn.log_softmax(q_logits, axis=-1)
        return np.asarray(jnp.sum(jnp.exp(log_p) * (log_p - log_q), axis=-1))

    expected = reference_logits(torch.float32)
    # Transformers' own bfloat16 run (on the CPU) shows the error bfloat16 itself
    # causes, with the same rounding recipe.
    torch_bfloat16 = reference_logits(torch.bfloat16)

    kl = kl_divergence(expected, logits)
    torch_kl = kl_divergence(expected, torch_bfloat16)
    top1 = np.mean(logits.argmax(axis=-1) == expected.argmax(axis=-1))
    max_diff = np.abs(logits - expected).max()

    print(
        f"\n{repo_id} bfloat16 on {jax.devices()[0].device_kind}, "
        f"B = {len(CHAT_PROMPTS)}, T = {bucket}, against float32 transformers: "
        f"top-1 agreement {top1:.3f}, max |logit difference| {max_diff:.4f} "
        f"(logit range {np.abs(expected).max():.2f}), KL mean {kl.mean():.2e}, "
        f"max {kl.max():.2e} (transformers bfloat16: top-1 agreement "
        f"{np.mean(torch_bfloat16.argmax(axis=-1) == expected.argmax(axis=-1)):.3f}, "
        f"max |logit difference| {np.abs(torch_bfloat16 - expected).max():.4f}, "
        f"KL mean {torch_kl.mean():.2e}, max {torch_kl.max():.2e}); "
        f"compile {compile_seconds:.1f} s, run {run_seconds * 1000:.1f} ms"
    )

    assert top1 >= 0.75
    # No worse than transformers' own bfloat16 forward, up to a small margin.
    assert kl.mean() <= 2 * torch_kl.mean() + 1e-3
