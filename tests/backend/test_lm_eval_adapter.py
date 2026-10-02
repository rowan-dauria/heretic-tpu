# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
JaxLM against lm-eval's HFLM (float32 on CPU, eager attention) on tiny random-weight
checkpoints of Gemma 3, Llama 3 and Qwen2.5 with their real tokenisers and generation
configs: identical encodings and model inputs (empty contexts, contexts ending in
spaces, BOS-prefixed and over-length inputs), loglikelihoods and greedy flags,
rolling loglikelihoods and generate_until strings with stop sequences and per-task
max_gen_toks. State resolution with hand-built states: adapters and model switches
take effect without rebuilding JaxLM, and max_length follows the state. Batching:
the shape policy's batch sizes and filler rows, the re-run after an out-of-memory
error, and one sampling key per batch. A slow test compares simple_evaluate on a
small real model with HFLM, sample by sample.
"""

import itertools
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from huggingface_hub import hf_hub_download
from lm_eval.api.instance import Instance
from transformers import AutoTokenizer, GenerationConfig

from heretic_tpu.backend import engine, weights
from heretic_tpu.backend.arch import ArchConfig
from heretic_tpu.backend.errors import DeviceMemoryError
from heretic_tpu.backend.lm_eval_adapter import JaxLM
from heretic_tpu.backend.sharding import choose_plan
from tests.backend.tiny import (
    REAL_MODELS,
    build_checkpoint,
    load_reference,
    tiny_config,
)

torch = pytest.importorskip("torch")
HFLM = pytest.importorskip("lm_eval.models.huggingface").HFLM

FAMILIES = ["gemma3_text", "llama", "qwen2"]

# Small enough for over-length inputs, and large enough for a few generated tokens
# after a truncated context.
MAX_POSITION_EMBEDDINGS = 64

# Loglikelihood sums over up to MAX_POSITION_EMBEDDINGS tokens, in float32.
ATOL = 2e-4
RTOL = 1e-5


class State(NamedTuple):
    """A hand-built `LMState`."""

    engine: engine.Engine
    params: weights.Params
    lora: Any
    hf_config: Any
    generation_config: GenerationConfig
    pad_id: int
    next_key: Callable[[], jax.Array]


class Case(NamedTuple):
    tokenizer: Any
    state: State
    reference: Any


def _key_factory(seed: int = 0) -> Callable[[], jax.Array]:
    """A sampling-key factory like the facade's: one new key per call."""

    root = jax.random.fold_in(jax.random.key(seed), 3)
    counter = itertools.count()
    return lambda: jax.random.fold_in(root, next(counter))


def _tokenizer(repo_id: str) -> Any:
    # As the facade sets it up.
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _build(name: str, directory: Path, **text_overrides: Any) -> Path:
    """A tiny checkpoint with the real vocabulary and generation config."""

    repo_id = REAL_MODELS[name]
    overrides = {"max_position_embeddings": MAX_POSITION_EMBEDDINGS, **text_overrides}
    if name == "gemma3_text":
        # A window shorter than the longest inputs.
        overrides.setdefault("sliding_window", 16)
    build_checkpoint(tiny_config(repo_id, vocab_size=None, **overrides), directory)
    shutil.copyfile(
        hf_hub_download(repo_id, "generation_config.json"),
        directory / "generation_config.json",
    )
    return directory


def _state(directory: Path, tokenizer: Any) -> State:
    ckpt = weights.resolve_checkpoint(str(directory), None)
    tensors = weights.build_tensor_index(ckpt)
    arch = ArchConfig.from_hf(ckpt.config, ckpt.raw_config, tensors)
    plan = choose_plan(arch, np.float32, "single")
    return State(
        engine=engine.Engine(arch, plan, np.float32),
        params=weights.load_params(ckpt, tensors, arch, np.float32, plan),
        lora=None,
        hf_config=ckpt.config,
        generation_config=ckpt.generation_config,
        pad_id=tokenizer.pad_token_id,
        next_key=_key_factory(),
    )


@pytest.fixture(scope="module")
def cases(tmp_path_factory) -> Callable[[str], Case]:
    """Builds and loads each family on first use."""

    root = tmp_path_factory.mktemp("lm_eval")
    cache: dict[str, Case] = {}

    def get(name: str) -> Case:
        if name not in cache:
            directory = _build(name, root / name)
            tokenizer = _tokenizer(REAL_MODELS[name])
            cache[name] = Case(
                tokenizer, _state(directory, tokenizer), load_reference(directory)
            )
        return cache[name]

    return get


def _hflm(case: Case, batch_size: int = 1) -> Any:
    return HFLM(
        pretrained=case.reference, tokenizer=case.tokenizer, batch_size=batch_size
    )


def _instances(request_type: str, arguments: list[tuple]) -> list[Instance]:
    return [
        Instance(request_type=request_type, doc={}, arguments=args, idx=index)
        for index, args in enumerate(arguments)
    ]


def _record_token_requests(lm: Any, monkeypatch: pytest.MonkeyPatch) -> list:
    """Records the token-level requests that loglikelihood passes on."""

    recorded = []
    original = lm._loglikelihood_tokens

    def record(requests, *args, **kwargs):
        recorded.extend(
            (context, continuation) for _, context, continuation in requests
        )
        return original(requests, *args, **kwargs)

    monkeypatch.setattr(lm, "_loglikelihood_tokens", record)
    return recorded


def _record_hflm_inputs(hflm: Any, monkeypatch: pytest.MonkeyPatch) -> list:
    """The input rows of HFLM's forward passes (unpadded at batch size 1)."""

    recorded = []
    original = hflm._model_call

    def record(inps, *args, **kwargs):
        recorded.extend(row.tolist() for row in inps)
        return original(inps, *args, **kwargs)

    monkeypatch.setattr(hflm, "_model_call", record)
    return recorded


def _record_engine_calls(eng: engine.Engine, monkeypatch: pytest.MonkeyPatch) -> list:
    """The batches of every score and generate call of an engine."""

    recorded = []
    for name in ("score", "generate"):
        original = getattr(eng, name)

        def record(params, lora, batch, *args, original=original, **kwargs):
            recorded.append(batch)
            return original(params, lora, batch, *args, **kwargs)

        monkeypatch.setattr(eng, name, record)
    return recorded


def _real_rows(batch: Any) -> list[list[int]]:
    return [batch.tokens[row][batch.mask[row]].tolist() for row in range(batch.n_real)]


def _assert_loglikelihoods_match(actual: list, expected: list) -> None:
    assert len(actual) == len(expected)
    np.testing.assert_allclose(
        [logprob for logprob, _ in actual],
        [logprob for logprob, _ in expected],
        rtol=RTOL,
        atol=ATOL,
    )
    assert [greedy for _, greedy in actual] == [greedy for _, greedy in expected]


def _loglikelihood_arguments(tokenizer: Any) -> list[tuple[str, str]]:
    long_text = " ".join(f"word{i}" for i in range(80))
    bos = tokenizer.bos_token or tokenizer.eos_token
    return [
        ("The capital of France is", " Paris"),
        ("The capital of France is", " Lyon"),
        # A trailing space moves to the continuation.
        ("The capital of France is ", "Paris"),
        ("Question: 2 + 2 =  ", "4 and more"),
        # Empty contexts are conditioned on the prefix token.
        ("", "Hello world"),
        ("", "x"),
        # A context that already starts with the BOS (or prefix) token.
        (f"{bos}Once upon a time", " there was"),
        # Inputs longer than max_length are truncated from the left.
        (long_text, " and the end"),
        (long_text, " " + " ".join(f"tail{i}" for i in range(10))),
        # Single-token answers to one question share a forward pass.
        ("Answer:", " A"),
        ("Answer:", " B"),
        ("Answer:", " C"),
        ("Answer:", " A"),
    ]


@pytest.mark.parametrize("name", FAMILIES)
def test_loglikelihood_matches_hflm(cases, name: str, monkeypatch) -> None:
    case = cases(name)
    arguments = _loglikelihood_arguments(case.tokenizer)
    jax_lm = JaxLM(case.tokenizer, lambda: case.state)
    hflm = _hflm(case)

    assert jax_lm.max_length == hflm.max_length == MAX_POSITION_EMBEDDINGS
    assert jax_lm.prefix_token_id == hflm.prefix_token_id
    assert jax_lm.eot_token_id == hflm.eot_token_id

    jax_requests = _record_token_requests(jax_lm, monkeypatch)
    hflm_requests = _record_token_requests(hflm, monkeypatch)
    jax_batches = _record_engine_calls(case.state.engine, monkeypatch)
    hflm_inputs = _record_hflm_inputs(hflm, monkeypatch)

    actual = jax_lm.loglikelihood(_instances("loglikelihood", arguments))
    expected = hflm.loglikelihood(_instances("loglikelihood", arguments))

    assert jax_requests == hflm_requests
    jax_inputs = [row for batch in jax_batches for row in _real_rows(batch)]
    assert sorted(jax_inputs) == sorted(hflm_inputs)
    assert max(map(len, jax_inputs)) == MAX_POSITION_EMBEDDINGS
    _assert_loglikelihoods_match(actual, expected)


@pytest.mark.parametrize("name", FAMILIES)
def test_greedy_flags_match_hflm(cases, name: str) -> None:
    """
    Continuations that are (in part) the model's own greedy tokens give true greedy
    flags; requests sharing a context are scored as candidates of one forward pass.
    """

    case = cases(name)
    rng = np.random.default_rng(0)
    vocab_size = case.state.engine.arch.vocab_size

    requests = []
    for length in (5, 17, 40):
        context = rng.integers(1000, vocab_size, length).tolist()
        greedy = case.reference.generate(
            torch.tensor([context]),
            attention_mask=torch.ones(1, length, dtype=torch.long),
            max_new_tokens=4,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=None,
            pad_token_id=case.tokenizer.pad_token_id,
        )[0, length:].tolist()
        other = int(rng.integers(1000, vocab_size))
        requests += [
            (None, context, greedy),
            (None, context, greedy[:1]),
            (None, context, [other]),
            (None, context, greedy[:3] + [other]),
        ]

    actual = JaxLM(case.tokenizer, lambda: case.state)._loglikelihood_tokens(requests)
    expected = _hflm(case)._loglikelihood_tokens(requests)

    _assert_loglikelihoods_match(actual, expected)
    assert sum(greedy for _, greedy in actual) >= 3


def test_members_with_shorter_continuations(cases) -> None:
    """
    A request whose continuation is shorter than that of another request with the
    same context plus continuation without its last token is scored over its own
    continuation (HFLM gathers the first positions of the longer one instead).
    """

    case = cases("llama")
    lm = JaxLM(case.tokenizer, lambda: case.state)
    context = [128000, 791, 6864, 315, 9822]
    continuation = [374, 12366, 13]
    requests = [
        (None, context, continuation),
        (None, context + continuation[:1], continuation[1:]),
        (None, context + continuation[:2], continuation[2:]),
    ]

    together = lm._loglikelihood_tokens(requests)
    alone = [lm._loglikelihood_tokens([request])[0] for request in requests]
    _assert_loglikelihoods_match(together, alone)


@pytest.mark.parametrize("name", FAMILIES)
def test_loglikelihood_rolling_matches_hflm(cases, name: str) -> None:
    case = cases(name)
    strings = [
        "Hello world.",
        " ".join(f"token{i}" for i in range(60)),
        "",
        "A short line\nand another one.",
    ]
    arguments = [(string,) for string in strings]

    actual = JaxLM(case.tokenizer, lambda: case.state).loglikelihood_rolling(
        _instances("loglikelihood_rolling", arguments)
    )
    expected = _hflm(case).loglikelihood_rolling(
        _instances("loglikelihood_rolling", arguments)
    )

    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=2 * ATOL)


def _generation_arguments(stop: str) -> list[tuple[str, dict[str, Any]]]:
    short = ["Once upon a time", "The weather today", "List three colours:", "Hi"]
    long = [" ".join(f"item{i}" for i in range(n)) for n in (40, 55)]
    greedy = {"until": [stop, "\n\n"], "max_gen_toks": 12, "do_sample": False}
    return [
        *[(context, greedy) for context in short],
        # Another group with another max_gen_toks.
        *[(context, {"until": [stop], "max_gen_toks": 5}) for context in short[:2]],
        # Left-truncated to max_length - max_gen_toks tokens.
        *[(context, {"until": ["."], "max_gen_toks": 8}) for context in long],
    ]


@pytest.mark.parametrize("name", FAMILIES)
def test_generate_until_matches_hflm(cases, name: str, monkeypatch) -> None:
    case = cases(name)
    jax_lm = JaxLM(case.tokenizer, lambda: case.state)
    # Each group of generation kwargs in one batch, as in JaxLM, whose rows of each
    # group share a prompt bucket here; so the padding token is penalised alike.
    hflm = _hflm(case, batch_size=64)

    # A stop sequence the model produces: part of an unstopped generation.
    (unstopped,) = jax_lm.generate_until(
        _instances("generate_until", [("Once upon a time", {"max_gen_toks": 12})])
    )
    stop = unstopped[len(unstopped) // 2 :][:3]
    assert stop

    arguments = _generation_arguments(stop)
    batches = _record_engine_calls(case.state.engine, monkeypatch)
    actual = jax_lm.generate_until(_instances("generate_until", arguments))
    expected = hflm.generate_until(_instances("generate_until", arguments))

    assert actual == expected
    assert actual[0] != unstopped and unstopped.startswith(actual[0])
    # One batch per group of generation kwargs (and prompt bucket).
    assert [batch.n_real for batch in batches] == [4, 2, 2]
    assert batches[-1].tokens.shape[1] == MAX_POSITION_EMBEDDINGS


def _lora(arch: ArchConfig, rank: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    lora = {}
    for component in weights.abliterable_components(arch):
        d_out, d_in = weights.component_shape(arch, component)
        shape = (arch.num_hidden_layers, 1)
        lora[component] = (
            jnp.asarray(rng.normal(size=(*shape, rank, d_in)) / d_in**0.5, jnp.float32),
            jnp.asarray(rng.normal(size=(*shape, d_out, rank)), jnp.float32),
        )
    return lora


@pytest.fixture(scope="module")
def switched(tmp_path_factory, cases) -> Case:
    """Another Llama checkpoint, with a different depth and max_length."""

    case = cases("llama")
    directory = _build(
        "llama",
        tmp_path_factory.mktemp("lm_eval_switched") / "llama",
        num_hidden_layers=2,
        max_position_embeddings=48,
    )
    return Case(case.tokenizer, _state(directory, case.tokenizer), None)


def test_state_is_resolved_per_call(cases, switched) -> None:
    """
    Every call evaluates the state state_fn returns at that time: adapters, their
    removal and a switch to another model take effect without rebuilding JaxLM,
    whose results equal those of a JaxLM built for that state.
    """

    case = cases("llama")
    arch = case.state.engine.arch
    states = {
        "base": case.state,
        "adapted": case.state._replace(lora=_lora(arch, rank=2, seed=1)),
        "readapted": case.state._replace(lora=_lora(arch, rank=2, seed=2)),
        "switched": switched.state,
    }
    current = {"state": "base"}
    calls = []

    def state_fn() -> State:
        calls.append(current["state"])
        return states[current["state"]]

    lm = JaxLM(case.tokenizer, state_fn)
    long_text = " ".join(f"word{i}" for i in range(80))
    loglikelihood = _instances(
        "loglikelihood", [("The capital of France is", " Paris"), (long_text, " end")]
    )
    generation = _instances(
        "generate_until", [("Once upon a time", {"until": ["\n"], "max_gen_toks": 6})]
    )

    def evaluate(lm: JaxLM) -> tuple:
        return (
            tuple(lm.loglikelihood(loglikelihood)),
            tuple(lm.generate_until(generation)),
        )

    results = {}
    for name in ["base", "adapted", "readapted", "base", "switched"]:
        current["state"] = name
        calls.clear()
        result = evaluate(lm)
        assert calls == [name, name]
        assert lm.max_length == (48 if name == "switched" else MAX_POSITION_EMBEDDINGS)

        fresh = evaluate(JaxLM(case.tokenizer, lambda name=name: states[name]))
        assert result[1] == fresh[1]
        _assert_loglikelihoods_match(list(result[0]), list(fresh[0]))
        results.setdefault(name, result)

    for name in ["adapted", "readapted", "switched"]:
        assert not np.allclose(
            [logprob for logprob, _ in results[name][0]],
            [logprob for logprob, _ in results["base"][0]],
        )

    # Nothing of the model is kept between calls.
    for value in vars(lm).values():
        assert not isinstance(value, (jax.Array, engine.Engine, dict, tuple))


def test_batches_follow_the_shape_policy(cases, monkeypatch) -> None:
    """
    Rows of one shape key run in batches of up to 64 rows, the last one padded with
    filler rows to a power of two; a batch that runs out of memory is run again at
    half the size, with identical results.
    """

    case = cases("qwen2")
    rng = np.random.default_rng(0)
    requests = [
        (
            None,
            rng.integers(1000, 100000, 20).tolist(),
            rng.integers(1000, 100000, 3).tolist(),
        )
        for _ in range(70)
    ]
    lm = JaxLM(case.tokenizer, lambda: case.state)

    batches = _record_engine_calls(case.state.engine, monkeypatch)
    expected = lm._loglikelihood_tokens(requests)
    assert [(len(batch.tokens), batch.n_real) for batch in batches] == [
        (64, 64),
        (8, 6),
    ]
    # Filler rows repeat the last real row.
    assert (batches[1].tokens[5:] == batches[1].tokens[5]).all()

    eng = case.state.engine
    original_score = eng.score
    lowered = []

    def score(params, lora, batch):
        if len(batch.tokens) == 64:
            raise DeviceMemoryError("simulated")
        return original_score(params, lora, batch)

    def lower_limit(key, failed_b):
        lowered.append(failed_b)
        return engine.Engine.lower_limit(eng, key, failed_b)

    monkeypatch.setattr(eng, "score", score)
    monkeypatch.setattr(eng, "lower_limit", lower_limit)
    batches.clear()
    actual = lm._loglikelihood_tokens(requests)

    assert lowered == [64]
    _assert_loglikelihoods_match(actual, expected)
    # Later tests would otherwise inherit the recorded failure at 64 rows.
    eng.release()


def test_sampling_takes_one_key_per_batch(cases) -> None:
    """
    Sampling requests take one key per engine batch from the state, so equal keys
    give equal samples; greedy requests take none.
    """

    case = cases("qwen2")
    keys = []

    def state_fn() -> State:
        def next_key() -> jax.Array:
            keys.append(len(keys))
            return jax.random.key(len(keys))

        return case.state._replace(next_key=next_key)

    lm = JaxLM(case.tokenizer, state_fn)
    sampling = {"do_sample": True, "temperature": 1.0, "max_gen_toks": 6, "until": []}
    long = " ".join(f"item{i}" for i in range(40))
    arguments = [("Once upon a time", sampling), ("Hi", sampling), (long, sampling)]

    first = lm.generate_until(_instances("generate_until", arguments))
    # Two prompt buckets, so two batches.
    assert keys == [0, 1]

    keys.clear()
    assert lm.generate_until(_instances("generate_until", arguments)) == first

    greedy = lm.generate_until(
        _instances(
            "generate_until",
            [(context, {"max_gen_toks": 6}) for context, _ in arguments],
        )
    )
    assert keys == [0, 1] and len(greedy) == 3


# A small real model for the end-to-end comparison.
SMALL_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.mark.slow
def test_simple_evaluate_matches_hflm() -> None:
    """
    lm_eval.simple_evaluate on a small real model gives the per-sample responses of
    HFLM: loglikelihoods for piqa and generated answers for gsm8k.
    """

    import lm_eval
    from transformers import AutoModelForCausalLM

    tokenizer = _tokenizer(SMALL_MODEL)
    ckpt = weights.resolve_checkpoint(SMALL_MODEL, None)
    weights.fetch_shards(ckpt)
    state = _state(Path(ckpt.snapshot_dir), tokenizer)
    reference = AutoModelForCausalLM.from_pretrained(
        ckpt.snapshot_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    models = {
        "jax": JaxLM(tokenizer, lambda: state),
        "hf": HFLM(pretrained=reference, tokenizer=tokenizer, batch_size=64),
    }
    for task in ["piqa", "gsm8k"]:
        samples = {}
        for name, lm in models.items():
            results = lm_eval.simple_evaluate(
                model=lm, tasks=[task], limit=2, log_samples=True, bootstrap_iters=0
            )
            samples[name] = sorted(
                results["samples"][task], key=lambda sample: sample["doc_id"]
            )

        for actual, expected in zip(samples["jax"], samples["hf"], strict=True):
            if task == "piqa":
                for (logprob, greedy), (expected_logprob, expected_greedy) in zip(
                    *(
                        [response for (response,) in sample["resps"]]
                        for sample in (actual, expected)
                    ),
                    strict=True,
                ):
                    assert logprob == pytest.approx(
                        expected_logprob, rel=2e-5, abs=1e-3
                    )
                    assert greedy == expected_greedy
                assert actual["acc"] == expected["acc"]
                assert actual["acc_norm"] == expected["acc_norm"]
            else:
                assert actual["resps"] == expected["resps"]
                assert actual["filtered_resps"] == expected["filtered_resps"]
