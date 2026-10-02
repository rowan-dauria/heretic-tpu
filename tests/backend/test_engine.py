# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
The engine's entry points and shape policy: batch construction, `decode_spec`, the
repetition penalty and sampling warpers against transformers' own logits processors
on fixed logits, `stop_check` semantics, streaming, sampling, captures, scoring
against transformers' log-probabilities, and the batch-size policy (`batch_limit`,
`lower_limit`, `fits`) with simulated memory limits.
"""

import functools
import math
import os
import re
import subprocess
import sys
import textwrap
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from transformers import GenerationConfig

from heretic_tpu.backend import engine, transformer, weights
from heretic_tpu.backend.arch import NO_WINDOW, ArchConfig, RopeSpec
from heretic_tpu.backend.engine import (
    DeviceMemoryError,
    Engine,
    ShapeKey,
    decode_spec,
    is_out_of_memory,
    score_batch,
    split_rows,
    token_batch,
)
from heretic_tpu.backend.sharding import choose_plan
from tests.backend.tiny import (
    REAL_MODELS,
    build_checkpoint,
    load_reference,
    tiny_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def synthetic_arch(**changes: Any) -> ArchConfig:
    """A small Llama-like architecture, for tests that need no weights."""

    layers = changes.pop("num_hidden_layers", 2)
    head_dim = 16
    rope = RopeSpec(
        rope_type="default",
        rot=head_dim,
        params=(("rope_theta", 10000.0), ("rope_type", "default")),
    )
    fields = {
        "model_type": "llama",
        "num_hidden_layers": layers,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 256,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 2**16,
        "head_dim": head_dim,
        "activation": "silu",
        "attn_logit_softcapping": None,
        "final_logit_softcapping": None,
        "query_pre_attn_scalar": None,
        "norm_topk_prob": None,
        "num_experts_per_tok": None,
        "num_experts": None,
        "moe_intermediate_size": None,
        "biases": (),
        "tied_head": True,
        "multimodal": False,
        "layer_types": ("full_attention",) * layers,
        "windows": (NO_WINDOW,) * layers,
        "rope": (rope,) * layers,
        "rope_per_layer_type": False,
        "rope_switch": None,
        "rope_switch_len": None,
    }
    fields.update(changes)
    return ArchConfig(**fields)


# Batch construction.


def test_buckets_and_splits() -> None:
    assert engine.bucket_length(1) == 32
    assert engine.bucket_length(32) == 32
    assert engine.bucket_length(33) == 64
    assert engine.bucket_length(1000) == 1024

    assert split_rows(10, 4) == [(0, 4, 4), (4, 8, 4), (8, 10, 2)]
    assert split_rows(5, 8) == [(0, 5, 8)]
    assert split_rows(3, 64) == [(0, 3, 4)]
    assert split_rows(7, 1) == [(i, i + 1, 1) for i in range(7)]


def test_token_batch() -> None:
    batch = token_batch([[5, 6, 7], [8]], [3, 3], pad_id=1, batch_size=4)
    assert batch.tokens.shape == (4, 32) and batch.tokens.dtype == np.int32
    assert batch.mask.dtype == np.bool_
    assert batch.n_real == 2
    np.testing.assert_array_equal(batch.tokens[0, -3:], [5, 6, 7])
    np.testing.assert_array_equal(batch.tokens[1, -2:], [1, 8])
    assert (batch.tokens[0, :-3] == 1).all()
    np.testing.assert_array_equal(batch.mask.sum(axis=1), [3, 1, 1, 1])
    # Filler rows repeat the last real row, including its reference length.
    np.testing.assert_array_equal(batch.tokens[3], batch.tokens[1])
    np.testing.assert_array_equal(batch.ref_len, [3, 3, 3, 3])

    with pytest.raises(ValueError):
        token_batch([[5]] * 3, [1] * 3, pad_id=0, batch_size=2)
    with pytest.raises(ValueError):
        token_batch([[5] * 40], [40], pad_id=0, length=32)


def test_score_batch() -> None:
    batch = score_batch(
        [[5, 6, 7], [8, 9]],
        [3, 3],
        [[1, 2, 3], [4]],
        pad_id=0,
        C_score=2,
        batch_size=4,
    )
    assert batch.G == 4 and batch.C_score == 2
    np.testing.assert_array_equal(batch.cand[0], [1, 2, 3, 0])
    np.testing.assert_array_equal(batch.cand_mask[1], [True, False, False, False])
    np.testing.assert_array_equal(batch.cand[3], batch.cand[1])


# decode_spec.


def test_decode_spec_resolves_like_generate(monkeypatch) -> None:
    monkeypatch.setattr(engine, "_warned", set())

    config = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        eos_token_id=[3, 4],
        repetition_penalty=1.1,
    )
    key = jax.random.key(0)

    spec = decode_spec(config, {}, max_new_tokens=10, chunk=32, pad_id=3, key=key)
    assert spec.sampling and spec.C_chunk == 32 and spec.max_new_tokens == 10
    np.testing.assert_array_equal(spec.eos_ids, [3, 4, -1, -1, -1, -1, -1, -1])
    assert (spec.temperature, spec.top_p, spec.repetition_penalty) == (0.7, 0.8, 1.1)
    # Unset fields take transformers' global defaults (top_k = 50 when sampling).
    assert spec.top_k == 50 and spec.min_p == 0.0
    assert spec.key is key

    # The facade's greedy override: warpers are neutral and no key is kept.
    greedy = decode_spec(
        config, {"do_sample": False}, max_new_tokens=10, chunk=1, pad_id=3, key=key
    )
    assert not greedy.sampling and greedy.key is None
    assert (greedy.temperature, greedy.top_k, greedy.top_p, greedy.min_p) == (
        1.0,
        0,
        1.0,
        0.0,
    )
    assert greedy.repetition_penalty == 1.1

    # The caller's config is unchanged, and an int EOS id is a one-element list.
    assert config.top_k is None
    plain = decode_spec(
        GenerationConfig(eos_token_id=7),
        {"do_sample": False},
        max_new_tokens=1,
        chunk=32,
        pad_id=0,
        key=None,
    )
    assert plain.eos_ids[0] == 7 and (plain.eos_ids[1:] == -1).all()
    assert plain.repetition_penalty == 1.0

    with pytest.raises(ValueError):
        decode_spec(
            GenerationConfig(eos_token_id=list(range(9))),
            {},
            max_new_tokens=1,
            chunk=1,
            pad_id=0,
            key=None,
        )
    with pytest.raises(ValueError):
        decode_spec(
            config, {"temperature": 0.0}, max_new_tokens=1, chunk=1, pad_id=0, key=key
        )
    with pytest.raises(ValueError):
        decode_spec(config, {}, max_new_tokens=1, chunk=1, pad_id=0, key=None)


def test_decode_spec_warns_once_about_unsupported_processors(monkeypatch) -> None:
    monkeypatch.setattr(engine, "_warned", set())
    config = GenerationConfig(no_repeat_ngram_size=3, typical_p=0.5)

    def spec(sampling: bool) -> None:
        decode_spec(
            config,
            {"do_sample": sampling},
            max_new_tokens=1,
            chunk=1,
            pad_id=0,
            key=jax.random.key(0),
        )

    with pytest.warns(UserWarning) as record:
        spec(False)
    assert ["no_repeat_ngram_size" in str(w.message) for w in record] == [True]

    # typical_p is a sampling warper; the processor warning is not repeated.
    with pytest.warns(UserWarning) as record:
        spec(True)
    assert ["typical_p" in str(w.message) for w in record] == [True]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec(True)


# Logits processing against transformers.


class Model(NamedTuple):
    directory: Path
    arch: ArchConfig
    params: weights.Params
    engine: Engine
    reference: Any


@pytest.fixture(scope="module")
def llama(tmp_path_factory) -> Model:
    pytest.importorskip("torch")
    directory = build_checkpoint(
        tiny_config(REAL_MODELS["llama"]), tmp_path_factory.mktemp("llama")
    )
    ckpt = weights.resolve_checkpoint(str(directory), None)
    tensors = weights.build_tensor_index(ckpt)
    arch = ArchConfig.from_hf(ckpt.config, ckpt.raw_config, tensors)
    plan = choose_plan(arch, np.float32, "single")
    params = weights.load_params(ckpt, tensors, arch, np.float32, plan)
    return Model(
        directory,
        arch,
        params,
        Engine(arch, plan, np.float32),
        load_reference(directory),
    )


WARPER_SETTINGS = [
    {},
    {"temperature": 0.7},
    {"top_k": 0, "top_p": 0.9},
    {"top_k": 0, "top_p": 0.0},
    {"top_k": 5, "temperature": 1.5},
    {"top_k": 1},
    {"top_k": 0, "min_p": 0.1},
    {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.02},
    {"top_k": 5000, "top_p": 0.5, "repetition_penalty": 1.3},
]


@pytest.mark.parametrize("settings", WARPER_SETTINGS)
@pytest.mark.parametrize("ties", [False, True])
def test_processors_match_transformers(
    llama, settings: dict[str, Any], ties: bool
) -> None:
    """
    The repetition penalty and the sampling warpers on fixed logits equal the logits
    processor list transformers builds for `generate` from the same configuration.
    With `ties`, the logits are rounded to bfloat16, as a bfloat16 model's are, so
    that many are equal (top-p, which transformers applies to tied tokens in an
    arbitrary sort order, is then left out).
    """

    import torch

    if ties and "top_p" in settings:
        pytest.skip("transformers breaks top-p ties in an arbitrary order")

    config = GenerationConfig(
        **{"do_sample": True, "repetition_penalty": 1.2, **settings}
    )
    spec = decode_spec(
        config, {}, max_new_tokens=1, chunk=1, pad_id=0, key=jax.random.key(0)
    )

    resolved, _ = llama.reference._prepare_generation_config(config)
    processors = llama.reference._get_logits_processor(
        resolved, input_ids_seq_length=12, device="cpu"
    )

    rng = np.random.default_rng(0)
    vocab_size = 1000
    logits = (3 * rng.standard_normal((4, vocab_size))).astype(np.float32)
    if ties:
        logits = logits.astype(ml_dtypes.bfloat16).astype(np.float32)
        logits[:, :10] = 0.0
        logits[:, 10:20] = -0.0
    previous = rng.integers(0, vocab_size, (4, 12))
    expected = processors(torch.from_numpy(previous), torch.from_numpy(logits)).numpy()

    penalised = np.zeros((4, vocab_size), dtype=bool)
    np.put_along_axis(penalised, previous, True, axis=1)
    params = engine.sampling_params(spec)

    @jax.jit
    def process(logits, penalised, params):
        scores = engine.apply_repetition_penalty(
            logits, penalised, params.repetition_penalty
        )
        return engine.apply_warpers(scores, params)

    actual = np.asarray(process(logits, penalised, params))

    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    kept = ~np.isneginf(expected)
    np.testing.assert_allclose(actual[kept], expected[kept], rtol=1e-6)
    if settings:
        # Something was filtered or changed.
        assert np.isneginf(expected).any() or not np.allclose(expected, logits)


def test_order_keys_order_like_the_scores() -> None:
    rng = np.random.default_rng(1)
    scores = np.concatenate(
        [
            rng.standard_normal(1000).astype(np.float32)
            * 10.0 ** rng.integers(-30, 30, 1000),
            np.array([0.0, -0.0, np.inf, -np.inf, 1e-45, -1e-45], dtype=np.float32),
        ]
    )
    keys = np.asarray(jax.jit(engine._order_keys)(scores[None]))[0]
    for i, j in rng.integers(0, len(scores), (5000, 2)):
        assert (keys[i] < keys[j]) == (scores[i] < scores[j])
        assert (keys[i] == keys[j]) == (scores[i] == scores[j])


# Entry points.


def _rows(arch: ArchConfig, lengths: list[int], seed: int = 0) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [rng.integers(3, arch.vocab_size, length).tolist() for length in lengths]


def _greedy(pad_id: int = 0, eos: list[int] | None = None, **kwargs: Any):
    return decode_spec(
        GenerationConfig(eos_token_id=eos or [], repetition_penalty=1.3),
        {"do_sample": False},
        pad_id=pad_id,
        key=None,
        **{"max_new_tokens": 16, "chunk": 4, **kwargs},
    )


def test_stop_check(llama) -> None:
    rows = _rows(llama.arch, [20, 9, 14])
    batch = token_batch(rows, [20] * 3, pad_id=0, batch_size=4)
    plain = llama.engine.generate(llama.params, None, batch, _greedy())

    # An EOS token that only row 0 generates, by its fifth token.
    for index in range(1, 5):
        eos = int(plain.tokens[0, index])
        if eos not in plain.tokens[0, :index] and eos not in plain.tokens[1:]:
            break
    else:
        pytest.fail("Row 0 generates no suitable EOS token.")
    eos_finish = index + 1
    spec = _greedy(eos=[eos])
    reference = llama.engine.generate(llama.params, None, batch, spec)
    np.testing.assert_array_equal(reference.finish, [eos_finish, 16, 16])

    calls = []

    def stop_check(tokens: np.ndarray, done: np.ndarray) -> np.ndarray:
        calls.append((tokens.copy(), done.copy()))
        # Row 1 is marked once five tokens exist. Row 0 is proposed as well, but it
        # has finished by then, so it is ignored.
        marked = tokens.shape[1] >= 5
        return np.array([marked, marked, False])

    result = llama.engine.generate(llama.params, None, batch, spec, stop_check)

    # Called between chunks with the real rows' tokens so far (filler rows are never
    # passed), until row 1 is marked.
    assert [tokens.shape for tokens, _ in calls[:3]] == [(3, 1), (3, 5), (3, 9)]
    np.testing.assert_array_equal(calls[1][1], [True, False, False])
    np.testing.assert_array_equal(calls[2][1], [True, True, False])

    # Row 0 is finished at its EOS token, as without stop_check.
    assert result.finish[0] == eos_finish
    np.testing.assert_array_equal(result.tokens[0], reference.tokens[0])

    # Row 1 counts the tokens generated when it was marked and continues with padding.
    assert result.finish[1] == 5
    np.testing.assert_array_equal(result.tokens[1, :5], reference.tokens[1, :5])
    assert (result.tokens[1, 5:] == 0).all()

    # Row 2 is unaffected.
    np.testing.assert_array_equal(result.tokens[2], reference.tokens[2])
    assert result.finish[2] == reference.finish[2]

    # Marking every unfinished row ends the generation at once.
    calls.clear()

    def stop_all(tokens: np.ndarray, done: np.ndarray) -> np.ndarray:
        calls.append(tokens.shape[1])
        return np.full(done.shape, tokens.shape[1] >= 5)

    result = llama.engine.generate(llama.params, None, batch, spec, stop_all)
    assert calls == [1, 5]
    np.testing.assert_array_equal(result.finish, [eos_finish, 5, 5])
    assert (result.tokens[1:, 5:] == 0).all()


def test_stream_yields_the_generated_tokens(llama) -> None:
    rows = _rows(llama.arch, [11], seed=1)
    batch = token_batch(rows, [11], pad_id=0)
    plain = llama.engine.generate(llama.params, None, batch, _greedy())

    eos = int(plain.tokens[0, 6])
    spec = _greedy(eos=[eos])
    generated = llama.engine.generate(llama.params, None, batch, spec)
    streamed = list(llama.engine.stream(llama.params, None, batch, spec))

    assert all(token.shape == (1,) for token in streamed)
    finish = generated.finish[0]
    assert finish <= 7
    # Up to and including the EOS token.
    np.testing.assert_array_equal(
        np.concatenate(streamed), generated.tokens[0, :finish]
    )


def test_sampling(llama) -> None:
    rows = _rows(llama.arch, [17, 6], seed=2)
    batch = token_batch(rows, [17, 17], pad_id=0)
    greedy = llama.engine.generate(llama.params, None, batch, _greedy(chunk=8))

    def sample(key: jax.Array, **settings: Any) -> np.ndarray:
        spec = decode_spec(
            GenerationConfig(repetition_penalty=1.3, **settings),
            {"do_sample": True},
            max_new_tokens=16,
            chunk=8,
            pad_id=0,
            key=key,
        )
        return llama.engine.generate(llama.params, None, batch, spec).tokens

    # Top-k with k = 1 keeps only the greedy token.
    np.testing.assert_array_equal(sample(jax.random.key(0), top_k=1), greedy.tokens)

    # One program serves every warper setting, and draws depend only on the key.
    first = sample(jax.random.key(1), temperature=2.0)
    np.testing.assert_array_equal(sample(jax.random.key(1), temperature=2.0), first)
    assert not np.array_equal(sample(jax.random.key(2), temperature=2.0), first)
    assert not np.array_equal(first, greedy.tokens)
    sampling_keys = [key for key, _ in llama.engine._executables if key.sampling]
    assert len(sampling_keys) == 1


def test_capture_matches_prefill(llama) -> None:
    rows = _rows(llama.arch, [30, 12, 3], seed=3)
    batch = token_batch(rows, [30] * 3, pad_id=0, batch_size=4)
    want = frozenset({"logits", "hidden", "module_io"})

    captures = llama.engine.capture(llama.params, None, batch, want)
    expected, _ = jax.jit(
        functools.partial(transformer.prefill, arch=llama.arch, want=want)
    )(llama.params, None, batch.tokens, batch.mask)

    for actual, reference in zip(
        jax.tree.leaves(captures), jax.tree.leaves(expected), strict=True
    ):
        assert actual.shape[0] == 4
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))

    logits_only = llama.engine.capture(llama.params, None, batch, {"logits"})
    assert logits_only.hidden is None and logits_only.module_io is None
    with pytest.raises(ValueError):
        llama.engine.capture(llama.params, None, batch, frozenset({"attentions"}))


def test_score_matches_transformers_log_probabilities(llama) -> None:
    import torch

    arch = llama.arch
    rows = _rows(arch, [25, 10, 7], seed=4)
    candidates = [[5, 9, 200], [17], [4, 4]]
    batch = score_batch(rows, [25] * 3, candidates, pad_id=0, C_score=8, batch_size=4)

    outputs = llama.engine.score(llama.params, None, batch)
    assert outputs.next_lp.shape == (3, 8) and outputs.last_lp.shape == (3, 4)

    # transformers on the rows left-padded to the longest row.
    longest = 25
    tokens = np.zeros((3, longest), dtype=np.int64)
    mask = np.zeros((3, longest), dtype=np.int64)
    for index, row in enumerate(rows):
        tokens[index, longest - len(row) :] = row
        mask[index, longest - len(row) :] = 1
    positions = np.where(mask, np.cumsum(mask, axis=1) - 1, 0)
    with torch.no_grad():
        logits = llama.reference(
            input_ids=torch.from_numpy(tokens),
            attention_mask=torch.from_numpy(mask),
            position_ids=torch.from_numpy(positions),
        ).logits
    log_probs = torch.log_softmax(logits, dim=-1).numpy()
    greedy = logits.argmax(dim=-1).numpy()

    columns = np.arange(longest - 8, longest - 1)
    targets = tokens[:, columns + 1]
    expected_lp = np.take_along_axis(log_probs[:, columns], targets[..., None], -1)
    # Columns of real tokens (padding columns attend to a different number of
    # padding slots in the bucket).
    real = mask[:, columns].astype(bool)
    assert not real.all()
    np.testing.assert_allclose(
        outputs.next_lp[:, :-1][real], expected_lp[..., 0][real], atol=1e-4
    )
    np.testing.assert_array_equal(
        outputs.next_greedy[:, :-1][real], (greedy[:, columns] == targets)[real]
    )
    assert (outputs.next_lp[:, -1] == 0).all() and outputs.next_greedy[:, -1].all()

    for index, row_candidates in enumerate(candidates):
        count = len(row_candidates)
        expected = log_probs[index, -1, row_candidates]
        np.testing.assert_allclose(outputs.last_lp[index, :count], expected, atol=1e-4)
        np.testing.assert_array_equal(
            outputs.last_greedy[index, :count],
            greedy[index, -1] == np.array(row_candidates),
        )
        assert (outputs.last_lp[index, count:] == 0).all()
        assert not outputs.last_greedy[index, count:].any()

    # The argmax of some row is among its candidates, so greedy flags are exercised.
    batch = score_batch(
        rows, [25] * 3, [[int(g)] for g in greedy[:, -1]], pad_id=0, C_score=8
    )
    assert llama.engine.score(llama.params, None, batch).last_greedy[:, 0].all()


# Shape policy.


def test_is_out_of_memory() -> None:
    assert is_out_of_memory(DeviceMemoryError("too big"))
    assert is_out_of_memory(jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: oom"))
    assert not is_out_of_memory(jax.errors.JaxRuntimeError("INTERNAL: oops"))
    assert not is_out_of_memory(MemoryError())


def _fake_engine(
    monkeypatch,
    fits: Callable[[int], bool],
    compile_fails: Callable[[int], bool] = lambda size: False,
) -> tuple[Engine, list[int]]:
    """An engine whose compilation and memory check are simulated."""

    arch = synthetic_arch()
    eng = Engine(arch, choose_plan(arch, np.float32, "single"), np.float32)
    compiled = []

    def lower_all(key: ShapeKey, size: int) -> dict:
        compiled.append(size)
        if compile_fails(size):
            raise jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: HBM")
        return {}

    monkeypatch.setattr(eng, "_lower_all", lower_all)
    monkeypatch.setattr(eng, "fits", lambda key, size: fits(size))
    return eng, compiled


KEY = ShapeKey("generate", 64, None, max_new_tokens=100, C_chunk=32)


def test_batch_limit_halves_and_records(monkeypatch) -> None:
    eng, compiled = _fake_engine(monkeypatch, lambda size: size <= 4)

    assert eng.batch_limit(KEY, 16) == 4
    assert compiled == [16, 8, 4]

    # Sizes at or above the smallest failing one are skipped, and sizes up to the
    # largest fitting one are returned at once.
    compiled.clear()
    assert eng.batch_limit(KEY, 64) == 4
    assert eng.batch_limit(KEY, 3) == 3
    assert eng.batch_limit(KEY, 4) == 4
    assert compiled == []

    # Other keys have their own records.
    other = KEY._replace(T=128)
    assert eng.batch_limit(other, 2) == 2
    assert compiled == [2]

    # A failure at run time lowers the limit.
    compiled.clear()
    assert eng.lower_limit(KEY, 4) == 2
    assert compiled == [2]
    assert eng.batch_limit(KEY, 64) == 2
    assert eng.lower_limit(KEY, 2) == 1
    with pytest.raises(DeviceMemoryError):
        eng.lower_limit(KEY, 1)

    eng.release()
    compiled.clear()
    assert eng.batch_limit(KEY, 8) == 4
    assert compiled == [8, 4]


def test_batch_limit_treats_compile_oom_as_failure(monkeypatch) -> None:
    eng, compiled = _fake_engine(
        monkeypatch, fits=lambda size: True, compile_fails=lambda size: size > 2
    )
    assert eng.batch_limit(KEY, 8) == 2
    assert compiled == [8, 4, 2]

    eng, _ = _fake_engine(monkeypatch, fits=lambda size: False)
    with pytest.raises(DeviceMemoryError):
        eng.batch_limit(KEY, 4)

    # Errors other than out-of-memory propagate.
    eng, _ = _fake_engine(monkeypatch, fits=lambda size: True)

    def broken(key: ShapeKey, size: int) -> dict:
        raise ValueError("broken")

    monkeypatch.setattr(eng, "_lower_all", broken)
    with pytest.raises(ValueError):
        eng.batch_limit(KEY, 4)


def test_entry_points_check_fits_before_the_first_run(llama, monkeypatch) -> None:
    eng = Engine(llama.arch, choose_plan(llama.arch, np.float32, "single"), np.float32)
    batch = token_batch(_rows(llama.arch, [9, 4]), [9, 9], pad_id=0)

    monkeypatch.setattr(eng, "fits", lambda key, size: False)
    with pytest.raises(DeviceMemoryError) as error:
        eng.capture(llama.params, None, batch, {"logits"})
    assert is_out_of_memory(error.value)
    with pytest.raises(DeviceMemoryError):
        eng.generate(llama.params, None, batch, _greedy())

    # Checked once per (key, B).
    checks = []
    monkeypatch.setattr(eng, "fits", lambda key, size: checks.append(size) or True)
    for _ in range(2):
        eng.capture(llama.params, None, batch, {"logits"})
    assert checks == [2]


def test_batch_limit_on_cpu_compiles_at_the_cap(llama) -> None:
    eng = Engine(llama.arch, choose_plan(llama.arch, np.float32, "single"), np.float32)
    key = ShapeKey("capture", 32, None, want=frozenset({"logits"}))

    # Without memory statistics everything fits.
    assert eng.fits(key, 8)
    assert eng.batch_limit(key, 8) == 8
    assert (key, 8) in eng._executables

    batch = token_batch(_rows(llama.arch, [5] * 8), [5] * 8, pad_id=0)
    assert eng.capture(llama.params, None, batch, {"logits"}).logits.shape == (
        8,
        llama.arch.vocab_size,
    )


def test_fits_reserves_memory(monkeypatch) -> None:
    arch = synthetic_arch()
    eng = Engine(arch, choose_plan(arch, np.float32, "single"), np.float32)
    key = ShapeKey("generate", 32, 4, max_new_tokens=64, C_chunk=32)
    need = eng.memory_need(key, 8)

    # The prefill output (cache and decoding state) is part of the need.
    cache_bytes = 2 * arch.num_hidden_layers * 8 * 96 * 2 * 16 * 4
    assert need >= cache_bytes

    limit = 10 * 2**20
    reserve = engine.MEMORY_RESERVE * limit

    def stats(in_use: int):
        return lambda devices: [{"bytes_limit": limit, "bytes_in_use": in_use}]

    # Adapters exist (rank 4): only the 5 % reserve.
    monkeypatch.setattr(engine, "_memory_stats", stats(int(limit - reserve - need)))
    assert eng.fits(key, 8)
    monkeypatch.setattr(engine, "_memory_stats", stats(int(limit - reserve - need) + 2))
    assert not eng.fits(key, 8)

    # Without adapters, room for rank-50 adapters is kept as well.
    no_lora = key._replace(rank=None)
    adapter_bytes = sum(
        arch.num_hidden_layers * 50 * (d_in + d_out) * 4
        for d_out, d_in in (
            weights.component_shape(arch, component)
            for component in weights.abliterable_components(arch)
        )
    )
    in_use = int(limit - reserve - adapter_bytes - eng.memory_need(no_lora, 8))
    monkeypatch.setattr(engine, "_memory_stats", stats(in_use))
    assert eng.fits(no_lora, 8)
    monkeypatch.setattr(engine, "_memory_stats", stats(in_use + 2))
    assert not eng.fits(no_lora, 8)

    # Outputs kept on the device need another 10 %.
    on_device = Engine(
        arch,
        choose_plan(arch, np.float32, "single"),
        np.float32,
        offload_outputs_to_cpu=False,
    )
    monkeypatch.setattr(engine, "_memory_stats", stats(int(limit - reserve - need)))
    assert not on_device.fits(key, 8)
    output_reserve = engine.OUTPUT_RESERVE * limit
    monkeypatch.setattr(
        engine, "_memory_stats", stats(int(limit - reserve - output_reserve - need))
    )
    assert on_device.fits(key, 8)


def test_adapter_values_are_arguments(llama) -> None:
    """
    Parameters and adapters are arguments, never compile-time constants: new adapter
    values reuse the executables, which hold no constant as large as a parameter.
    """

    arch = llama.arch
    rank = 3
    lora = {}
    for component in weights.abliterable_components(arch):
        d_out, d_in = weights.component_shape(arch, component)
        lora[component] = (
            jnp.full((arch.num_hidden_layers, 1, rank, d_in), 0.1),
            jnp.full((arch.num_hidden_layers, 1, d_out, rank), 0.1),
        )
    batch = token_batch(_rows(arch, [9, 4]), [9, 9], pad_id=0)
    eng = Engine(arch, choose_plan(arch, np.float32, "single"), np.float32)

    first = eng.generate(llama.params, lora, batch, _greedy())
    lora = {component: (a, -b) for component, (a, b) in lora.items()}
    second = eng.generate(llama.params, lora, batch, _greedy())
    assert len(eng._executables) == 1
    assert not np.array_equal(first.tokens, second.tokens)

    smallest_parameter = min(
        math.prod(leaf.shape) for leaf in jax.tree.leaves(llama.params)
    )
    for executables in eng._executables.values():
        for program in executables.programs.values():
            for shape in re.findall(
                r"\w+\[([\d,]*)\]\S* constant\(", program.compiled.as_text()
            ):
                size = math.prod(int(d) for d in shape.split(",") if d)
                assert size < max(smallest_parameter, arch.vocab_size)


# Runs the entry points under the "tensor" plan of a forced 4-device CPU mesh and
# compares them with the "single" plan.
_TENSOR_PARALLEL_SCRIPT = textwrap.dedent(
    """
    import sys

    import jax
    import numpy as np
    from transformers import GenerationConfig

    from heretic_tpu.backend import engine, weights
    from heretic_tpu.backend.arch import ArchConfig
    from heretic_tpu.backend.sharding import choose_plan

    assert len(jax.local_devices()) == 4

    ckpt = weights.resolve_checkpoint(sys.argv[1], None)
    tensors = weights.build_tensor_index(ckpt)
    arch = ArchConfig.from_hf(ckpt.config, ckpt.raw_config, tensors)

    rng = np.random.default_rng(0)
    rows = [rng.integers(3, arch.vocab_size, n).tolist() for n in (21, 13, 5)]
    lora = {}
    for component in weights.abliterable_components(arch):
        d_out, d_in = weights.component_shape(arch, component)
        lora[component] = (
            rng.standard_normal((arch.num_hidden_layers, 1, 2, d_in)) / d_in**0.5,
            rng.standard_normal((arch.num_hidden_layers, 1, d_out, 2)) / 2**0.5,
        )

    def run(plan):
        params = weights.load_params(ckpt, tensors, arch, np.float32, plan)
        adapters = {
            component: tuple(
                jax.device_put(np.float32(array), sharding)
                for array, sharding in zip(lora[component], plan.lora_shardings(component))
            )
            for component in lora
        }
        eng = engine.Engine(arch, plan, np.float32)
        batch = engine.token_batch(rows, [21] * 3, pad_id=0, batch_size=4)
        outputs = {}

        captures = eng.capture(params, adapters, batch, engine.CAPTURES)
        for name, value in jax.tree_util.tree_leaves_with_path(captures):
            outputs["capture" + jax.tree_util.keystr(name)] = np.asarray(value)

        config = GenerationConfig(eos_token_id=[0, 7], repetition_penalty=1.3)
        for sampling in (False, True):
            spec = engine.decode_spec(
                config,
                {"do_sample": sampling, "temperature": 0.8},
                max_new_tokens=12,
                chunk=4,
                pad_id=0,
                key=jax.random.key(0),
            )
            generated = eng.generate(params, adapters, batch, spec)
            outputs[f"tokens/{sampling}"] = generated.tokens
            outputs[f"finish/{sampling}"] = generated.finish

        scores = eng.score(
            params,
            adapters,
            engine.score_batch(rows, [21] * 3, [[5, 6], [7], [8]], pad_id=0, C_score=4),
        )
        for name, value in scores._asdict().items():
            outputs["score/" + name] = value
        return outputs

    tensor_plan = choose_plan(arch, np.float32, "tensor")
    assert tensor_plan.attention_sharded and tensor_plan.mlp_sharded
    single = run(choose_plan(arch, np.float32, "single"))
    tensor = run(tensor_plan)
    for name, expected in single.items():
        if expected.dtype.kind in "biu":
            np.testing.assert_array_equal(tensor[name], expected, err_msg=name)
        else:
            # Sharded contractions sum partial results in another order.
            np.testing.assert_allclose(
                tensor[name], expected, rtol=1e-4, atol=1e-4, err_msg=name
            )
    print(sorted(single))
    """
)


def test_tensor_parallel_entry_points_on_four_cpu_devices(tmp_path) -> None:
    pytest.importorskip("torch")
    directory = build_checkpoint(
        tiny_config(
            REAL_MODELS["llama"],
            num_attention_heads=8,
            num_key_value_heads=4,
            attention_bias=True,
        ),
        tmp_path / "llama",
        tie_word_embeddings=False,
    )
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        "PYTHONPATH": os.pathsep.join(
            [str(REPO_ROOT / "src"), str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]
        ),
    }
    result = subprocess.run(
        [sys.executable, "-c", _TENSOR_PARALLEL_SCRIPT, str(directory)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "tokens/True" in result.stdout and "score/last_lp" in result.stdout
