# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
Parity of the engine's generation with transformers' `generate` (float32 on CPU, eager
attention) on tiny random-weight checkpoints built from the configurations of real
models: greedy generation with a repetition penalty, the padding token in the EOS list
and mixed prompt lengths whose longest prompt is not a bucket size (identical tokens
and response lengths per row); adapters against PEFT; Phi-3 generation across the
long-factor switch against transformers without a cache; and dynamic RoPE, which
raises at the step that crosses `max_position_embeddings` and only then.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
from transformers import GenerationConfig

from heretic_tpu.backend import engine, weights
from heretic_tpu.backend.arch import ArchConfig
from heretic_tpu.backend.sharding import choose_plan
from tests.backend.tiny import (
    REAL_MODELS,
    build_checkpoint,
    load_reference,
    tiny_config,
)

torch = pytest.importorskip("torch")

# A sliding window shorter than prompt plus generation.
WINDOW = 8

# The Phi-3 case switches to the long RoPE factors above this sequence length.
PHI3_SWITCH = 20

# The dynamic RoPE case recomputes its frequencies above this sequence length.
DYNAMIC_LIMIT = 64

PENALTY = 1.3

# Mixed prompt lengths; the longest is not a bucket size (the bucket is 32).
LENGTHS = [21, 13, 5]


def _case(name: str) -> tuple[Any, dict[str, Any]]:
    """The tiny configuration of a test case, and the build_checkpoint arguments."""

    if name == "llama":
        # llama3 RoPE; tied, without lm_head.
        return tiny_config(REAL_MODELS["llama"]), {}
    if name in ("mistral", "phi3_4k", "gemma3", "gemma3_text"):
        return tiny_config(REAL_MODELS[name], sliding_window=WINDOW), {}
    if name in ("qwen2", "qwen3", "qwen3_moe", "mistral3"):
        return tiny_config(REAL_MODELS[name]), {}
    if name == "phi3":
        # Shaped like Phi-4-mini (partial rotary, longrope, tied), with a short
        # original context so that generation crosses the long-factor switch.
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
    if name == "llama-dynamic":
        rope = {"rope_type": "dynamic", "factor": 2.0, "rope_theta": 500000.0}
        config = tiny_config(
            REAL_MODELS["llama"],
            rope_parameters=rope,
            max_position_embeddings=DYNAMIC_LIMIT,
        )
        return config, {}
    raise ValueError(name)


class Model(NamedTuple):
    directory: Path
    arch: ArchConfig
    params: weights.Params
    engine: engine.Engine
    reference: Any


def _load(directory: Path) -> Model:
    ckpt = weights.resolve_checkpoint(str(directory), None)
    tensors = weights.build_tensor_index(ckpt)
    arch = ArchConfig.from_hf(ckpt.config, ckpt.raw_config, tensors)
    plan = choose_plan(arch, np.float32, "single")
    params = weights.load_params(ckpt, tensors, arch, np.float32, plan)

    reference = load_reference(directory)
    if arch.is_moe:
        # The numerics the engine follows (transformers defaults to grouped matmuls).
        reference.set_experts_implementation("eager")

    return Model(
        directory, arch, params, engine.Engine(arch, plan, np.float32), reference
    )


@pytest.fixture(scope="module")
def models(tmp_path_factory) -> Callable[[str], Model]:
    """Builds and loads each case on first use."""

    root = tmp_path_factory.mktemp("generation")
    cache: dict[str, Model] = {}

    def get(name: str) -> Model:
        if name not in cache:
            config, kwargs = _case(name)
            cache[name] = _load(build_checkpoint(config, root / name, **kwargs))
        return cache[name]

    return get


def _rows(arch: ArchConfig, lengths: Sequence[int], seed: int = 0) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [rng.integers(3, arch.vocab_size, length).tolist() for length in lengths]


def _spec(
    *,
    pad_id: int,
    eos: list[int],
    max_new_tokens: int,
    chunk: int = 8,
    penalty: float = PENALTY,
) -> engine.DecodeSpec:
    config = GenerationConfig(eos_token_id=eos, repetition_penalty=penalty)
    return engine.decode_spec(
        config,
        {"do_sample": False},
        max_new_tokens=max_new_tokens,
        chunk=chunk,
        pad_id=pad_id,
        key=None,
    )


def _generate(
    model: Model,
    rows: list[list[int]],
    spec: engine.DecodeSpec,
    *,
    ref_len: Sequence[int] | None = None,
    batch_size: int | None = None,
    lora: Any = None,
    stop_check: engine.StopCheck | None = None,
) -> engine.Generated:
    """Generates for rows that form one upstream batch (unless `ref_len` says else)."""

    if ref_len is None:
        ref_len = [max(len(row) for row in rows)] * len(rows)
    batch = engine.token_batch(rows, ref_len, pad_id=spec.pad_id, batch_size=batch_size)
    return model.engine.generate(model.params, lora, batch, spec, stop_check)


def _reference_generate(
    model: Any,
    rows: list[list[int]],
    *,
    pad_id: int,
    eos: list[int],
    max_new_tokens: int,
    penalty: float = PENALTY,
    use_cache: bool = True,
) -> engine.Generated:
    """transformers' greedy generate on rows left-padded as upstream pads them."""

    longest = max(len(row) for row in rows)
    tokens = np.full((len(rows), longest), pad_id)
    mask = np.zeros((len(rows), longest), dtype=np.int64)
    for index, row in enumerate(rows):
        tokens[index, longest - len(row) :] = row
        mask[index, longest - len(row) :] = 1

    with torch.no_grad():
        output = model.generate(
            input_ids=torch.from_numpy(tokens),
            attention_mask=torch.from_numpy(mask),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=penalty,
            eos_token_id=eos,
            pad_token_id=pad_id,
            use_cache=use_cache,
        )

    # Generation stops when every row is finished; the rest is padding.
    generated = np.full((len(rows), max_new_tokens), pad_id, dtype=np.int32)
    new_tokens = output[:, longest:].numpy()
    generated[:, : new_tokens.shape[1]] = new_tokens

    finish = np.full(len(rows), max_new_tokens, dtype=np.int32)
    for index, row in enumerate(generated):
        hits = np.flatnonzero(np.isin(row, eos))
        if hits.size:
            finish[index] = hits[0] + 1
    return engine.Generated(tokens=generated, finish=finish)


def _assert_generated(actual: engine.Generated, expected: engine.Generated) -> None:
    np.testing.assert_array_equal(actual.tokens, expected.tokens)
    np.testing.assert_array_equal(actual.finish, expected.finish)


GREEDY_CASES = [
    "llama",
    "mistral",
    "qwen2",
    "qwen3",
    "gemma3_text",
    "gemma3",
    "mistral3",
    "phi3_4k",
    "qwen3_moe",
]


@pytest.mark.parametrize("name", GREEDY_CASES)
def test_greedy_matches_transformers(models, name: str) -> None:
    """
    Greedy generation with a repetition penalty, the padding token in the EOS list
    and mixed prompt lengths, in a batch with a filler row.

    The padding token is chosen so that the pad rule (upstream penalises the padding
    token in rows it left-padded) changes the result, and an extra EOS token so that
    a row finishes early.
    """

    model = models(name)
    rows = _rows(model.arch, LENGTHS)
    max_new_tokens = 24

    # The tokens the rows generate show which ids make the test discriminate.
    first = _generate(model, rows, _spec(pad_id=0, eos=[], max_new_tokens=24))
    eos_token = int(first.tokens[0, 6])

    candidates = dict.fromkeys(first.tokens[1:, :12].ravel().tolist())
    for pad_id in candidates:
        if pad_id == eos_token:
            continue
        spec = _spec(pad_id=pad_id, eos=[pad_id, eos_token], max_new_tokens=24)
        actual = _generate(model, rows, spec, batch_size=4)
        without_pad_rule = _generate(model, rows, spec, ref_len=LENGTHS)
        if not np.array_equal(actual.tokens, without_pad_rule.tokens):
            break
    else:
        pytest.fail("No padding token id changes the result through the pad rule.")

    expected = _reference_generate(
        model.reference,
        rows,
        pad_id=pad_id,
        eos=[pad_id, eos_token],
        max_new_tokens=max_new_tokens,
    )
    _assert_generated(actual, expected)
    assert (expected.finish < max_new_tokens).any()


def test_filler_rows_and_chunk_size_do_not_change_results(models) -> None:
    model = models("llama")
    rows = _rows(model.arch, LENGTHS, seed=1)
    spec = _spec(pad_id=0, eos=[0, 5], max_new_tokens=20)

    results = [
        _generate(model, rows, spec._replace(C_chunk=chunk), batch_size=batch_size)
        for chunk, batch_size in ((8, None), (8, 4), (32, 8), (1, 3))
    ]
    for result in results[1:]:
        _assert_generated(result, results[0])


def test_adapters_match_peft(models) -> None:
    peft = pytest.importorskip("peft")

    model = models("llama")
    arch = model.arch
    rank = 2
    rng = np.random.default_rng(1)
    lora = {}
    for component in weights.abliterable_components(arch):
        d_out, d_in = weights.component_shape(arch, component)
        a = rng.standard_normal((arch.num_hidden_layers, 1, rank, d_in)) / d_in**0.5
        b = rng.standard_normal((arch.num_hidden_layers, 1, d_out, rank)) / rank**0.5
        lora[component] = (a.astype(np.float32), b.astype(np.float32))

    peft_model = peft.get_peft_model(
        load_reference(model.directory),
        peft.LoraConfig(
            r=rank,
            lora_alpha=rank,
            lora_dropout=0.0,
            bias="none",
            target_modules=[
                weights.module_path(arch, component, layer_index, 0)
                for component in lora
                for layer_index in range(arch.num_hidden_layers)
            ],
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

    rows = _rows(arch, LENGTHS, seed=2)
    spec = _spec(pad_id=0, eos=[0], max_new_tokens=16)
    actual = _generate(model, rows, spec, lora=lora)
    expected = _reference_generate(
        peft_model, rows, pad_id=0, eos=[0], max_new_tokens=16
    )
    _assert_generated(actual, expected)

    # The adapters change the generation.
    plain = _generate(model, rows, spec)
    assert not np.array_equal(plain.tokens, actual.tokens)


def test_phi3_generation_across_the_switch_matches_transformers_without_cache(
    models,
) -> None:
    """
    Two upstream batches share an engine batch: one switches to the long factors at
    decode step 6, the other at step 3 (each step re-prefills), with a third
    starting above the switch length. Each matches transformers run on its own
    upstream batch without a cache (transformers 5.17's cached Phi-3 generation
    drops the context at the switch).
    """

    model = models("phi3")
    arch = model.arch
    assert arch.rope_switch == "long_factor"

    batches = [
        _rows(arch, [15, 9], seed=3),  # switch step 20 - 15 + 1 = 6
        _rows(arch, [18, 4], seed=4),  # switch step 3
        _rows(arch, [22, 11], seed=5),  # long factors from the prompt on
    ]
    max_new_tokens = 12
    spec = _spec(pad_id=0, eos=[0], max_new_tokens=max_new_tokens, chunk=4)

    rows = [row for batch in batches for row in batch]
    ref_len = [max(map(len, batch)) for batch in batches for _ in batch]
    actual = _generate(model, rows, spec, ref_len=ref_len, batch_size=8)

    for index, batch in enumerate(batches):
        expected = _reference_generate(
            model.reference,
            batch,
            pad_id=0,
            eos=[0],
            max_new_tokens=max_new_tokens,
            use_cache=False,
        )
        rows_slice = slice(2 * index, 2 * index + 2)
        _assert_generated(
            engine.Generated(actual.tokens[rows_slice], actual.finish[rows_slice]),
            expected,
        )

    # Without the re-prefill (every row on short factors), the switching rows differ.
    short = _generate(model, rows, spec, ref_len=[1] * len(rows), batch_size=8)
    assert not np.array_equal(short.tokens[:4], actual.tokens[:4])


def test_dynamic_rope_raises_at_the_crossing_step(models) -> None:
    model = models("llama-dynamic")
    arch = model.arch
    assert arch.rope_switch == "raise" and arch.rope_switch_len == DYNAMIC_LIMIT

    # A 40-token prompt crosses the limit at decode step 64 - 40 + 1 = 25, the step
    # that consumes the 25th generated token.
    rows = _rows(arch, [40, 23], seed=6)

    # 25 tokens need decode steps up to 24 only, and match transformers (whose
    # frequencies are unchanged up to the limit).
    spec = _spec(pad_id=0, eos=[], max_new_tokens=25, penalty=1.0)
    _assert_generated(
        _generate(model, rows, spec),
        _reference_generate(
            model.reference,
            rows,
            pad_id=0,
            eos=[],
            max_new_tokens=25,
            penalty=1.0,
        ),
    )

    # One more token raises, after the tokens up to the crossing step exist.
    seen = []

    def stop_check(tokens: np.ndarray, done: np.ndarray) -> np.ndarray:
        seen.append(tokens.shape[1])
        return np.zeros_like(done)

    with pytest.raises(NotImplementedError):
        _generate(model, rows, spec._replace(max_new_tokens=26), stop_check=stop_check)
    assert seen[-1] == 25

    # A single forward call raises when its reference length exceeds the limit.
    batch = engine.token_batch(rows, [DYNAMIC_LIMIT + 1] * 2, pad_id=0)
    with pytest.raises(NotImplementedError):
        model.engine.capture(model.params, None, batch, frozenset({"logits"}))
    batch = engine.token_batch(rows, [DYNAMIC_LIMIT] * 2, pad_id=0)
    model.engine.capture(model.params, None, batch, frozenset({"logits"}))


def test_dynamic_rope_does_not_raise_when_rows_finish_first(models) -> None:
    model = models("llama-dynamic")
    rows = _rows(model.arch, [40, 23], seed=6)

    # Every row reaches an EOS token early.
    first = _generate(model, rows, _spec(pad_id=0, eos=[], max_new_tokens=8))
    eos = [int(first.tokens[0, 2]), int(first.tokens[1, 3])]

    spec = _spec(pad_id=0, eos=eos, max_new_tokens=40)
    actual = _generate(model, rows, spec)
    assert (actual.finish <= 4).all()
    _assert_generated(
        actual,
        _reference_generate(
            model.reference, rows, pad_id=0, eos=eos, max_new_tokens=40
        ),
    )
