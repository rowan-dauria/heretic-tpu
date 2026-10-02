# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
The engine: ahead-of-time compiled entry points for capturing internals, generating
and scoring, the KV cache of generation, and the batch-size side of the engine shape
policy (see "Engine" and "Engine shape policy" in docs/DESIGN.md).

Executables are compiled with `jit(...).lower(...).compile()` from abstract inputs and
cached by `(ShapeKey, B)`. Parameters, adapters and every per-call value are explicit
arguments, so a jitted function never closes over a device array, and the engine
itself holds only Python state.

Generation runs as a prefill, which builds the KV cache and selects the first token,
followed by chunks of up to `C_chunk` decode steps in a `lax.while_loop` that carry the
donated cache and the decoding state. The host checks between chunks whether to stop
(all rows finished, `stop_check`) and runs the long-RoPE re-prefill when a row reaches
its switch step.

The module also provides the helpers callers use to build batches: prompt buckets,
left padding, filler rows and the split of a group of rows into engine batches.
"""

from __future__ import annotations

import copy
import functools
import math
import warnings
from collections.abc import Callable, Generator, Iterator, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from transformers import GenerationConfig

from . import transformer
from .arch import ArchConfig
from .errors import DeviceMemoryError
from .sharding import DEFAULT_MAX_LORA_RANK, MEMORY_RESERVE, ShardingPlan
from .transformer import CAPTURES, Captures, KVCache, Lora
from .weights import Params, abliterable_components, component_shape, param_skeleton

# StopCheck(tokens [n_real, s], done [n_real]) -> finish_now [n_real]
# (see `Engine.generate`).
StopCheck = Callable[[np.ndarray, np.ndarray], np.ndarray]

ENTRIES = ("capture", "generate", "stream", "score")

# Prompts are left-padded to a power of two of at least this length.
MIN_BUCKET = 32

# Decode steps per chunk of batch generation and of streaming.
GENERATE_CHUNK = 32
STREAM_CHUNK = 1

# Size of the traced EOS id list (padded with -1, which matches no token).
MAX_EOS_IDS = 8

# Fraction of device memory kept free for outputs that stay on the device
# (`offload_outputs_to_cpu = false`).
OUTPUT_RESERVE = 0.10

# The switch step of rows that never switch RoPE parameters.
_NEVER = 2**30


class ShapeKey(NamedTuple):
    """Everything static about an engine call except the batch size."""

    # One of `ENTRIES`.
    entry: str

    # Prompt bucket.
    T: int

    # LoRA rank; None is the program without adapters.
    rank: int | None

    # Capture only.
    want: frozenset[str] = frozenset()

    # Generate and stream.
    max_new_tokens: int = 0
    C_chunk: int = 0
    sampling: bool = False

    # Score only: continuation bucket and candidate bucket.
    C_score: int = 0
    G: int = 0


class TokenBatch(NamedTuple):
    """A left-padded batch of prompts (see `token_batch`)."""

    # int32 [B, T], left-padded to the bucket.
    tokens: np.ndarray

    # bool [B, T], true for real tokens.
    mask: np.ndarray

    # Rows from n_real on are filler rows, whose results are discarded.
    n_real: int

    # int32 [B]: the real token count of the longest real row of the row's reference
    # batch (upstream batch for the facade, JaxLM batch for JaxLM). It drives the
    # repetition-penalty pad rule and RoPE switching.
    ref_len: np.ndarray


class ScoreBatch(NamedTuple):
    """A left-padded batch of scoring inputs (see `score_batch`)."""

    tokens: np.ndarray
    mask: np.ndarray
    n_real: int
    ref_len: np.ndarray

    # int32 [B, G]: candidate tokens scored at the last column, and their mask.
    cand: np.ndarray
    cand_mask: np.ndarray

    # Number of last columns scored against the next input token
    # (power-of-two bucket of the longest continuation).
    C_score: int

    @property
    def G(self) -> int:
        return self.cand.shape[1]


class DecodeSpec(NamedTuple):
    """Generation parameters, as built by `decode_spec`."""

    # Static.
    max_new_tokens: int
    C_chunk: int
    sampling: bool

    # Traced.
    eos_ids: np.ndarray  # int32 [MAX_EOS_IDS], padded with -1
    pad_id: int
    repetition_penalty: float
    temperature: float
    top_k: int
    top_p: float
    min_p: float

    # Sampling key (None when not sampling).
    key: jax.Array | None


class Generated(NamedTuple):
    # int32 [n_real, max_new_tokens], pad_id after a row finished.
    tokens: np.ndarray

    # int32 [n_real]: the number of tokens up to and including the row's first EOS;
    # for a row marked by stop_check, the number of tokens generated when it was
    # marked; max_new_tokens otherwise.
    finish: np.ndarray


class ScoreOutputs(NamedTuple):
    """
    Log-probabilities (float32) and greedy flags, real rows only. Column j of
    `next_*` is column T - C_score + j of the input, scored against the input token
    at the following column; the last column gives 0 and true. `last_*` score the
    candidate tokens at the last column (0 and false for masked candidates).
    """

    next_lp: np.ndarray  # [n_real, C_score]
    next_greedy: np.ndarray  # [n_real, C_score]
    last_lp: np.ndarray  # [n_real, G]
    last_greedy: np.ndarray  # [n_real, G]


class SamplingParams(NamedTuple):
    """The traced generation parameters of a `DecodeSpec`, as arrays."""

    eos_ids: Any  # int32 [MAX_EOS_IDS]
    pad_id: Any  # int32 []
    repetition_penalty: Any  # float32 []
    temperature: Any  # float32 []
    top_k: Any  # int32 []
    top_p: Any  # float32 []

    # float32(1 - top_p), with the subtraction in double precision, as transformers'
    # top-p warper computes its threshold.
    top_p_complement: Any

    min_p: Any  # float32 []

    # None for greedy programs.
    key: Any


class _DecodeState(NamedTuple):
    """The decoding state carried (and donated) from one generation call to the next."""

    # None when only one token is generated.
    cache: KVCache | None

    # bool [B, V]: the tokens the repetition penalty applies to.
    penalised: jax.Array

    # int32 [B, max_new_tokens]: the tokens generated so far, pad_id after a row
    # finished and in the columns not generated yet.
    tokens: jax.Array

    done: jax.Array  # bool [B]
    finish: jax.Array  # int32 [B]

    # Number of tokens generated so far; the next decode step consumes the newest.
    step: jax.Array  # int32 []


def is_out_of_memory(e: BaseException) -> bool:
    """Whether an error means that a batch does not fit into device memory."""

    if isinstance(e, DeviceMemoryError):
        return True
    return isinstance(e, jax.errors.JaxRuntimeError) and "RESOURCE_EXHAUSTED" in str(e)


def next_power_of_two(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()


def bucket_length(length: int) -> int:
    """The prompt bucket T of a batch whose longest real row has `length` tokens."""

    return max(MIN_BUCKET, next_power_of_two(length))


def split_rows(n_rows: int, batch_limit: int) -> list[tuple[int, int, int]]:
    """
    Splits a group of rows into engine batches of at most `batch_limit` (B_eff) rows,
    as `(start, stop, B)`. A batch with fewer rows is padded with filler rows to
    `min(batch_limit, next_power_of_two(rows))`, so that the number of compiled batch
    sizes stays logarithmic.
    """

    batches = []
    for start in range(0, n_rows, batch_limit):
        stop = min(start + batch_limit, n_rows)
        size = min(batch_limit, next_power_of_two(stop - start))
        batches.append((start, stop, size))
    return batches


def _left_pad(
    rows: Sequence[Sequence[int]],
    ref_len: Sequence[int],
    pad_id: int,
    batch_size: int | None,
    length: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_real = len(rows)
    if n_real == 0:
        raise ValueError("A batch needs at least one row.")
    if len(ref_len) != n_real:
        raise ValueError("ref_len needs one entry per row.")

    longest = max(len(row) for row in rows)
    if length is None:
        length = bucket_length(longest)
    if batch_size is None:
        batch_size = n_real
    if longest > length or min(len(row) for row in rows) == 0:
        raise ValueError(f"Rows must have between 1 and {length} tokens.")
    if batch_size < n_real:
        raise ValueError(f"{n_real} rows do not fit into a batch of {batch_size}.")

    # Filler rows repeat the last real row.
    rows = [*rows, *[rows[-1]] * (batch_size - n_real)]
    ref_len = [*ref_len, *[ref_len[-1]] * (batch_size - n_real)]

    tokens = np.full((batch_size, length), pad_id, dtype=np.int32)
    mask = np.zeros((batch_size, length), dtype=bool)
    for index, row in enumerate(rows):
        tokens[index, length - len(row) :] = row
        mask[index, length - len(row) :] = True

    return tokens, mask, np.asarray(ref_len, dtype=np.int32)


def token_batch(
    rows: Sequence[Sequence[int]],
    ref_len: Sequence[int],
    *,
    pad_id: int,
    batch_size: int | None = None,
    length: int | None = None,
) -> TokenBatch:
    """
    Builds a `TokenBatch` from token-id rows: left-padded with `pad_id` to `length`
    (by default the bucket of the longest row), and padded to `batch_size` rows (by
    default the number of rows) with filler rows that repeat the last row.
    `ref_len[i]` is the real token count of the longest row of row i's reference
    batch.
    """

    tokens, mask, ref_len = _left_pad(rows, ref_len, pad_id, batch_size, length)
    return TokenBatch(tokens=tokens, mask=mask, n_real=len(rows), ref_len=ref_len)


def score_batch(
    rows: Sequence[Sequence[int]],
    ref_len: Sequence[int],
    candidates: Sequence[Sequence[int]],
    *,
    pad_id: int,
    C_score: int,
    G: int | None = None,
    batch_size: int | None = None,
    length: int | None = None,
) -> ScoreBatch:
    """
    Builds a `ScoreBatch` like `token_batch`, with the candidate tokens of each row
    (padded to `G` entries, by default the power-of-two bucket of the largest
    candidate count) scored at the last column.
    """

    tokens, mask, ref_len = _left_pad(rows, ref_len, pad_id, batch_size, length)
    if len(candidates) != len(rows):
        raise ValueError("candidates needs one entry per row.")

    largest = max(len(row_candidates) for row_candidates in candidates)
    if G is None:
        G = next_power_of_two(largest)
    if largest > G:
        raise ValueError(f"A row has more than {G} candidates.")

    candidates = [*candidates, *[candidates[-1]] * (len(tokens) - len(candidates))]
    cand = np.zeros((len(tokens), G), dtype=np.int32)
    cand_mask = np.zeros((len(tokens), G), dtype=bool)
    for index, row_candidates in enumerate(candidates):
        cand[index, : len(row_candidates)] = row_candidates
        cand_mask[index, : len(row_candidates)] = True

    return ScoreBatch(
        tokens=tokens,
        mask=mask,
        n_real=len(rows),
        ref_len=ref_len,
        cand=cand,
        cand_mask=cand_mask,
        C_score=C_score,
    )


# Each warning is given once per process.
_warned: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        warnings.warn(message, stacklevel=3)


# Generation config fields whose logits processors are not implemented, with the
# predicate under which transformers would apply them.
_UNSUPPORTED_PROCESSORS: dict[str, Callable[[Any], bool]] = {
    "no_repeat_ngram_size": lambda value: value is not None and value > 0,
    "bad_words_ids": lambda value: value is not None,
    "min_length": lambda value: value is not None and value > 0,
    "min_new_tokens": lambda value: value is not None and value > 0,
    "forced_bos_token_id": lambda value: value is not None,
    "forced_eos_token_id": lambda value: value is not None,
    "suppress_tokens": lambda value: value is not None,
    "begin_suppress_tokens": lambda value: value is not None,
    "sequence_bias": lambda value: value is not None,
    "exponential_decay_length_penalty": lambda value: value is not None,
    "renormalize_logits": lambda value: value is True,
    "remove_invalid_values": lambda value: value is True,
    # Beam search would change the decoding strategy altogether.
    "num_beams": lambda value: value is not None and value > 1,
}

# The same for warpers, which transformers applies only when sampling.
_UNSUPPORTED_WARPERS: dict[str, Callable[[Any], bool]] = {
    "top_h": lambda value: value is not None,
    "typical_p": lambda value: value is not None and value < 1.0,
    "epsilon_cutoff": lambda value: value is not None and 0.0 < value < 1.0,
    "eta_cutoff": lambda value: value is not None and 0.0 < value < 1.0,
}


def decode_spec(
    generation_config: GenerationConfig,
    overrides: dict[str, Any],
    *,
    max_new_tokens: int,
    chunk: int,
    pad_id: int,
    key: jax.Array | None,
) -> DecodeSpec:
    """
    Turns a generation config into a `DecodeSpec`. Fields the config leaves unset take
    transformers' global defaults, and `overrides` (for example `{"do_sample":
    False}`, or an lm-eval request's normalised kwargs) are applied over it, as
    `generate()` resolves its configuration. Processors and warpers that are not
    implemented produce one warning each.
    """

    config = copy.deepcopy(generation_config)
    for name, value in GenerationConfig._get_default_generation_params().items():
        if getattr(config, name, None) is None:
            setattr(config, name, value)
    # As GenerationConfig.update, without its validation warnings.
    for name, value in overrides.items():
        if hasattr(config, name):
            setattr(config, name, value)
        else:
            _warn_once(f"Ignoring the unknown generation parameter '{name}'.")

    sampling = bool(config.do_sample)

    eos_token_id = config.eos_token_id
    if eos_token_id is None:
        eos_token_id = []
    elif isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if len(eos_token_id) > MAX_EOS_IDS:
        raise ValueError(f"At most {MAX_EOS_IDS} EOS token ids are supported.")
    eos_ids = np.full(MAX_EOS_IDS, -1, dtype=np.int32)
    eos_ids[: len(eos_token_id)] = eos_token_id

    unsupported = dict(_UNSUPPORTED_PROCESSORS)
    if sampling:
        unsupported.update(_UNSUPPORTED_WARPERS)
    for name, applies in unsupported.items():
        if applies(getattr(config, name, None)):
            _warn_once(
                f"The generation parameter '{name}' is not supported and is ignored."
            )

    repetition_penalty = config.repetition_penalty
    if repetition_penalty != 1.0 and not repetition_penalty > 0:
        raise ValueError(
            f"repetition_penalty has to be a strictly positive float, "
            f"but is {repetition_penalty}."
        )

    # Neutral values for the warpers, which only sampling applies.
    temperature, top_k, top_p, min_p = 1.0, 0, 1.0, 0.0
    if sampling:
        temperature = config.temperature
        top_k = config.top_k or 0
        top_p = config.top_p
        min_p = config.min_p or 0.0
        if temperature != 1.0 and not temperature > 0:
            raise ValueError(
                f"temperature has to be a strictly positive float, but is {temperature}."
            )
        if top_k < 0:
            raise ValueError(f"top_k has to be a non-negative integer, but is {top_k}.")
        if not 0 <= top_p <= 1:
            raise ValueError(f"top_p has to be in [0, 1], but is {top_p}.")
        if not 0 <= min_p <= 1:
            raise ValueError(f"min_p has to be in [0, 1], but is {min_p}.")
        if key is None:
            raise ValueError("Sampling needs a PRNG key.")

    return DecodeSpec(
        max_new_tokens=max_new_tokens,
        C_chunk=chunk,
        sampling=sampling,
        eos_ids=eos_ids,
        pad_id=pad_id,
        repetition_penalty=float(repetition_penalty),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        min_p=float(min_p),
        key=key if sampling else None,
    )


def sampling_params(spec: DecodeSpec) -> SamplingParams:
    """The traced parameters of a spec, as host arrays (and the device key)."""

    key = spec.key
    if key is not None and not jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        # A raw uint32 key (jax.random.PRNGKey).
        key = jax.random.wrap_key_data(key)

    return SamplingParams(
        eos_ids=np.asarray(spec.eos_ids, dtype=np.int32),
        pad_id=np.int32(spec.pad_id),
        repetition_penalty=np.float32(spec.repetition_penalty),
        temperature=np.float32(spec.temperature),
        top_k=np.int32(spec.top_k),
        top_p=np.float32(spec.top_p),
        top_p_complement=np.float32(1.0 - spec.top_p),
        min_p=np.float32(spec.min_p),
        key=key if spec.sampling else None,
    )


def apply_repetition_penalty(
    scores: jax.Array,
    penalised: jax.Array,
    penalty: jax.Array,
) -> jax.Array:
    """
    The repetition penalty on float32 scores [B, V], once per penalised token:
    negative scores are multiplied by the penalty, others divided by it.
    """

    penalised_scores = jnp.where(scores < 0, scores * penalty, scores / penalty)
    return jnp.where(penalised, penalised_scores, scores)


# Bisection steps that narrow any range of int32 keys to a single key.
_BISECTION_STEPS = 32


def _order_keys(scores: jax.Array) -> jax.Array:
    """
    int32 keys [B, V] that order like the float32 scores (NaN-free), so that
    thresholds can be searched by bisection over integers. -0.0 counts as 0.0.
    """

    bits = lax.bitcast_convert_type(jnp.where(scores == 0, 0.0, scores), jnp.int32)
    # Negative floats order in reverse of their bit patterns.
    return bits ^ ((bits >> 31) & 0x7FFFFFFF)


def _kth_largest_key(keys: jax.Array, k: jax.Array) -> jax.Array:
    """
    The key [B] of each row's k-th largest score (k >= 1): the largest t with at least
    k keys >= t.
    """

    def narrow(
        _: jax.Array,
        bounds: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        low, high = bounds
        # ceil((low + high) / 2), without overflowing int32.
        middle = (low | high) - ((low ^ high) >> 1)
        enough = jnp.sum(keys >= middle[:, None], axis=-1) >= k
        return jnp.where(enough, middle, low), jnp.where(enough, high, middle - 1)

    bounds = (jnp.min(keys, axis=-1), jnp.max(keys, axis=-1))
    return lax.fori_loop(0, _BISECTION_STEPS, narrow, bounds)[0]


def _top_p_threshold(
    keys: jax.Array,
    probs: jax.Array,
    complement: jax.Array,
) -> jax.Array:
    """
    The smallest key t [B] whose cumulative probability (of all tokens with keys up to
    t) exceeds `complement`, or the largest key if none does.
    """

    def narrow(
        _: jax.Array,
        bounds: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        low, high = bounds
        # floor((low + high) / 2), without overflowing int32.
        middle = (low & high) + ((low ^ high) >> 1)
        cumulative = jnp.sum(jnp.where(keys <= middle[:, None], probs, 0.0), axis=-1)
        exceeds = cumulative > complement
        return jnp.where(exceeds, low, middle + 1), jnp.where(exceeds, middle, high)

    # The upper bound is always a key that exceeds or the largest key (the lower
    # bound passes it once no key exceeds).
    bounds = (jnp.min(keys, axis=-1), jnp.max(keys, axis=-1))
    return lax.fori_loop(0, _BISECTION_STEPS, narrow, bounds)[1]


def apply_warpers(scores: jax.Array, params: SamplingParams) -> jax.Array:
    """
    The sampling warpers of transformers on float32 scores [B, V], in its order:
    temperature, top-k, top-p and min-p, each keeping at least one token. The
    parameters are traced; a warper whose value is neutral leaves the scores as they
    are.

    Top-k and top-p need only a threshold in the sorted order of the scores, which is
    found by bisection over keys that order like the scores: 32 masked reductions
    instead of sorting the vocabulary (which on TPU costs far more than the decode
    step itself).
    """

    vocab_size = scores.shape[-1]
    removed = -jnp.inf

    temperature = params.temperature
    scores = jnp.where(temperature != 1.0, scores / temperature, scores)

    # Top-k keeps every score at least as large as the k-th largest.
    def top_k(scores: jax.Array) -> jax.Array:
        keys = _order_keys(scores)
        kth = _kth_largest_key(keys, jnp.clip(params.top_k, 1, vocab_size))
        return jnp.where(keys < kth[:, None], removed, scores)

    scores = lax.cond(params.top_k != 0, top_k, lambda scores: scores, scores)

    # Top-p removes the tokens (in ascending order) whose cumulative probability is
    # at most 1 - top_p, except for the most likely one. Tokens with equal scores are
    # removed or kept together.
    def top_p(scores: jax.Array) -> jax.Array:
        keys = _order_keys(scores)
        probs = jax.nn.softmax(scores, axis=-1)
        threshold = _top_p_threshold(keys, probs, params.top_p_complement)
        return jnp.where(keys < threshold[:, None], removed, scores)

    scores = lax.cond(params.top_p < 1.0, top_p, lambda scores: scores, scores)

    # Min-p removes the tokens less likely than min_p times the most likely one
    # (which itself is always kept, because min_p <= 1).
    probs = jax.nn.softmax(scores, axis=-1)
    threshold = params.min_p * jnp.max(probs, axis=-1, keepdims=True)
    return jnp.where((params.min_p > 0.0) & (probs < threshold), removed, scores)


def _select(
    logits: jax.Array,
    penalised: jax.Array,
    params: SamplingParams,
    step: jax.Array,
    *,
    sampling: bool,
) -> jax.Array:
    """The next token [B] from float32 logits [B, V] (step: tokens generated so far)."""

    scores = apply_repetition_penalty(logits, penalised, params.repetition_penalty)
    if not sampling:
        # The first maximal index, as torch.argmax.
        return jnp.argmax(scores, axis=-1).astype(jnp.int32)

    scores = apply_warpers(scores, params)
    key = jax.random.fold_in(params.key, step)
    return jax.random.categorical(key, scores, axis=-1).astype(jnp.int32)


def _advance(
    state: _DecodeState,
    logits: jax.Array,
    params: SamplingParams,
    *,
    sampling: bool,
) -> _DecodeState:
    """Selects and records the next token of every row from its logits."""

    token = _select(logits, state.penalised, params, state.step, sampling=sampling)

    # Finished rows continue with padding, as in transformers' generate.
    token = jnp.where(state.done, params.pad_id, token)
    step = state.step + 1
    finished = ~state.done & jnp.any(token[:, None] == params.eos_ids[None, :], axis=-1)

    rows = jnp.arange(token.shape[0])
    return _DecodeState(
        cache=state.cache,
        penalised=state.penalised.at[rows, token].set(True),
        tokens=lax.dynamic_update_index_in_dim(state.tokens, token, state.step, axis=1),
        done=state.done | finished,
        finish=jnp.where(finished, step, state.finish),
        step=step,
    )


def _mark(state: _DecodeState, marks: jax.Array) -> _DecodeState:
    """Marks rows as finished now (for stop_check), recording their token count."""

    marked = marks & ~state.done
    return state._replace(
        done=state.done | marks,
        finish=jnp.where(marked, state.step, state.finish),
    )


def _generation_prefill(
    params: Params,
    lora: Lora | None,
    tokens: jax.Array,
    mask: jax.Array,
    ref_len: jax.Array,
    real: jax.Array,
    long_factor: jax.Array | None,
    sampling_params: SamplingParams,
    *,
    arch: ArchConfig,
    max_new_tokens: int,
    sampling: bool,
) -> _DecodeState:
    """
    Runs the prompt, builds the KV cache (only needed for a second token) and selects
    the first token. Filler rows (`real` false) start as finished.
    """

    batch, length = tokens.shape
    cache_len = length + max_new_tokens if max_new_tokens > 1 else None
    captures, cache = transformer.prefill(
        params,
        lora,
        tokens,
        mask,
        long_factor,
        arch=arch,
        cache_len=cache_len,
    )

    # Every real prompt token is penalised (padding slots are scattered out of
    # bounds and dropped), and the padding token if upstream would have left-padded
    # the row, i.e. if it is shorter than the longest row of its reference batch.
    # Bucket padding and filler rows never add it.
    rows = jnp.arange(batch)[:, None]
    penalised = jnp.zeros((batch, arch.vocab_size), dtype=bool)
    penalised = penalised.at[rows, jnp.where(mask, tokens, arch.vocab_size)].set(
        True, mode="drop"
    )
    pad_id = sampling_params.pad_id
    pad_penalised = jnp.sum(mask, axis=1) < ref_len
    penalised = penalised.at[:, pad_id].set(penalised[:, pad_id] | pad_penalised)

    state = _DecodeState(
        cache=cache,
        penalised=penalised,
        tokens=jnp.full((batch, max_new_tokens), pad_id, dtype=jnp.int32),
        done=~real,
        finish=jnp.full((batch,), max_new_tokens, dtype=jnp.int32),
        step=jnp.int32(0),
    )
    return _advance(state, captures.logits, sampling_params, sampling=sampling)


def _generation_chunk(
    params: Params,
    lora: Lora | None,
    state: _DecodeState,
    mask: jax.Array,
    switch_steps: jax.Array | None,
    stop_at: jax.Array,
    marks: jax.Array,
    sampling_params: SamplingParams,
    *,
    arch: ArchConfig,
    chunk: int,
    sampling: bool,
) -> _DecodeState:
    """
    Up to `chunk` decode steps. The loop stops early when every row is finished,
    when all tokens exist or before the step `stop_at` (a RoPE switch step).
    """

    batch, length = mask.shape
    max_new_tokens = state.tokens.shape[1]
    state = _mark(state, marks)

    # Slots from the current one on are treated as unwritten by the decode step,
    # so the generated slots can be marked as real throughout.
    kv_mask = jnp.concatenate(
        [mask, jnp.ones((batch, max_new_tokens), dtype=bool)], axis=1
    )
    prompt_len = jnp.sum(mask, axis=1, dtype=jnp.int32)

    def cond(carry: tuple[jax.Array, _DecodeState]) -> jax.Array:
        count, state = carry
        return (
            (count < chunk)
            & (state.step < max_new_tokens)
            & (state.step < stop_at)
            & ~jnp.all(state.done)
        )

    def body(
        carry: tuple[jax.Array, _DecodeState],
    ) -> tuple[jax.Array, _DecodeState]:
        count, state = carry

        # Step s consumes generated token s, at cache slot T + s - 1.
        step = state.step
        long_factor = None if switch_steps is None else step >= switch_steps
        logits, cache = transformer.decode_step(
            params,
            lora,
            state.cache,
            lax.dynamic_index_in_dim(state.tokens, step - 1, axis=1, keepdims=False),
            prompt_len + step - 1,
            length + step - 1,
            kv_mask,
            long_factor,
            arch=arch,
        )

        state = _advance(
            state._replace(cache=cache), logits, sampling_params, sampling=sampling
        )
        return count + 1, state

    _, state = lax.while_loop(cond, body, (jnp.int32(0), state))
    return state


def _generation_reprefill(
    params: Params,
    lora: Lora | None,
    state: _DecodeState,
    tokens: jax.Array,
    mask: jax.Array,
    switch_steps: jax.Array,
    marks: jax.Array,
    sampling_params: SamplingParams,
    *,
    arch: ArchConfig,
    sampling: bool,
) -> _DecodeState:
    """
    Runs the decode step at which some rows switch to the long RoPE factors as a
    prefill over every cache slot (the prompt followed by the tokens generated so
    far, unwritten slots masked), with each row's factors for this step. This
    rebuilds the whole KV cache, which equals transformers' generation without a
    cache, and then selects the next token exactly as a decode step does.
    """

    batch, length = tokens.shape
    max_new_tokens = state.tokens.shape[1]
    state = _mark(state, marks)
    step = state.step

    generated = jnp.broadcast_to(
        jnp.arange(max_new_tokens) < step, (batch, max_new_tokens)
    )
    written = jnp.concatenate([mask, generated], axis=1)
    all_tokens = jnp.where(
        written,
        jnp.concatenate([tokens, state.tokens], axis=1),
        sampling_params.pad_id,
    )

    captures, cache = transformer.prefill(
        params,
        lora,
        all_tokens,
        written,
        step >= switch_steps,
        arch=arch,
        cache_len=length + max_new_tokens,
        column=length + step - 1,
    )
    return _advance(
        state._replace(cache=cache), captures.logits, sampling_params, sampling=sampling
    )


def _capture(
    params: Params,
    lora: Lora | None,
    tokens: jax.Array,
    mask: jax.Array,
    long_factor: jax.Array | None,
    *,
    arch: ArchConfig,
    want: frozenset[str],
) -> Captures:
    captures, _ = transformer.prefill(
        params, lora, tokens, mask, long_factor, arch=arch, want=want
    )
    return captures


def _score(
    params: Params,
    lora: Lora | None,
    tokens: jax.Array,
    mask: jax.Array,
    long_factor: jax.Array | None,
    cand: jax.Array,
    cand_mask: jax.Array,
    *,
    arch: ArchConfig,
    C_score: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Scores the last `C_score` columns against the next input token and the
    candidates at the last column. The head is applied column by column, so
    [B, C_score, V] logits never exist at once.
    """

    batch, length = tokens.shape
    x = transformer.decoder(params, lora, tokens, mask, long_factor, arch=arch).x

    # The last column's target is a placeholder; its outputs are replaced below.
    targets = jnp.concatenate(
        [tokens[:, length - C_score + 1 :], jnp.zeros((batch, 1), dtype=jnp.int32)],
        axis=1,
    )

    def column(
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        x_column, target = inputs
        logits = transformer.unembed(params, x_column, arch=arch)
        normaliser = jax.nn.logsumexp(logits, axis=-1)
        greedy = jnp.argmax(logits, axis=-1)
        target_lp = jnp.take_along_axis(logits, target[:, None], axis=-1)[:, 0]
        cand_lp = jnp.take_along_axis(logits, cand, axis=-1)
        return (
            target_lp - normaliser,
            greedy == target,
            cand_lp - normaliser[:, None],
            greedy[:, None] == cand,
        )

    next_lp, next_greedy, cand_lp, cand_greedy = lax.map(
        column,
        (jnp.swapaxes(x[:, length - C_score :], 0, 1), targets.T),
    )

    last = C_score - 1
    next_lp = next_lp.T.at[:, last].set(0.0)
    next_greedy = next_greedy.T.at[:, last].set(True)
    last_lp = jnp.where(cand_mask, cand_lp[last], 0.0)
    last_greedy = cand_greedy[last] & cand_mask
    return next_lp, next_greedy, last_lp, last_greedy


class _Program(NamedTuple):
    compiled: Any  # jax.stages.Compiled

    # Bytes of the per-call inputs (tokens, masks, ...), excluding parameters,
    # adapters and donated state.
    transient_bytes: int


class _Executables:
    """The executables of one (ShapeKey, B)."""

    def __init__(self, programs: dict[str, _Program]):
        self.programs = programs

        # Whether `fits` has been checked before the first run.
        self.checked = False


class Engine:
    """
    Compiled entry points over explicit parameters and adapters.

    Every entry point runs exactly the batch it is given. Before the first run of an
    executable at (key, B) it checks that the run fits into device memory (`fits`)
    and raises DeviceMemoryError otherwise; out-of-memory errors propagate. Batching
    is the caller's job (see `batch_limit`, `split_rows` and `token_batch`).

    Shape keys are formed as follows (T is the batch's bucket, `rank` the LoRA rank or
    None): `ShapeKey("capture", T, rank, want=want)`, `ShapeKey("generate", T, rank,
    max_new_tokens=..., C_chunk=..., sampling=...)`, `ShapeKey("stream", T, rank,
    max_new_tokens=..., C_chunk=1, sampling=...)` and `ShapeKey("score", T, rank,
    C_score=..., G=...)`.
    """

    def __init__(
        self,
        arch: ArchConfig,
        plan: ShardingPlan,
        dtype: Any,
        *,
        offload_outputs_to_cpu: bool = True,
    ):
        self.arch = arch
        self.plan = plan
        self.dtype = np.dtype(dtype)

        # Outputs left on the device need room (see "Engine shape policy").
        self.offload_outputs_to_cpu = offload_outputs_to_cpu

        self._executables: dict[tuple[ShapeKey, int], _Executables] = {}

        # Per key: the largest batch size recorded as fitting,
        # and the smallest recorded as failing.
        self._fitting: dict[ShapeKey, int] = {}
        self._failing: dict[ShapeKey, int] = {}

    # Entry points.

    def capture(
        self,
        params: Params,
        lora: Lora | None,
        batch: TokenBatch,
        want: frozenset[str],
    ) -> Captures:
        """
        Captures `want` (a subset of {"logits", "hidden", "module_io"}) at the last
        prompt column, for every row of the batch (filler rows included), as device
        arrays: logits [B, V] float32, hidden [B, L + 1, D] and module I/O
        {component: (inputs [B, L, M, d_in], outputs [B, L, M, d_out])}.
        """

        want = frozenset(want)
        if not want or want - CAPTURES:
            raise ValueError(f"Invalid captures: {sorted(want)}")

        self._check_batch(batch)
        key = ShapeKey("capture", batch.tokens.shape[1], self._rank(lora), want=want)
        long_factor = self._forward_long_factor(batch)
        program = self._ready(key, batch.tokens.shape[0])["capture"]
        return program.compiled(params, lora, batch.tokens, batch.mask, long_factor)

    def generate(
        self,
        params: Params,
        lora: Lora | None,
        batch: TokenBatch,
        spec: DecodeSpec,
        stop_check: StopCheck | None = None,
    ) -> Generated:
        """
        Generates up to `spec.max_new_tokens` tokens per row, greedily or by sampling.

        `stop_check(tokens, done)`, if given, is called between chunks with the
        tokens the real rows generated so far (int32 [n_real, s], pad_id after a row
        finished) and their finished mask; it returns bool [n_real], true for rows to
        mark as finished now (entries for finished rows are ignored).
        """

        generation = self._generation(
            params, lora, batch, spec, spec.C_chunk, "generate", stop_check
        )
        while True:
            try:
                next(generation)
            except StopIteration as result:
                return result.value

    def stream(
        self,
        params: Params,
        lora: Lora | None,
        batch: TokenBatch,
        spec: DecodeSpec,
    ) -> Iterator[np.ndarray]:
        """
        Generates for a single row one decode step at a time and yields each new
        token (int32 [1]) as soon as it exists, up to and including the EOS token.
        """

        if batch.tokens.shape[0] != 1:
            raise ValueError("Streaming needs a batch of one row.")

        emitted = 0
        for state in self._generation(
            params, lora, batch, spec, STREAM_CHUNK, "stream", None
        ):
            tokens = np.asarray(state.tokens)[0]
            step = int(state.step)
            for index in range(emitted, step):
                yield tokens[index : index + 1]
            emitted = step

    def score(
        self,
        params: Params,
        lora: Lora | None,
        batch: ScoreBatch,
    ) -> ScoreOutputs:
        """Log-probabilities and greedy flags (see `ScoreOutputs`), real rows only."""

        self._check_batch(batch)
        length = batch.tokens.shape[1]
        if not 1 <= batch.C_score <= length:
            raise ValueError(f"C_score must be between 1 and {length}.")
        if batch.cand.shape != batch.cand_mask.shape or batch.cand.shape[0] != len(
            batch.tokens
        ):
            raise ValueError("cand and cand_mask must be [B, G].")

        key = ShapeKey(
            "score", length, self._rank(lora), C_score=batch.C_score, G=batch.G
        )
        long_factor = self._forward_long_factor(batch)
        program = self._ready(key, batch.tokens.shape[0])["score"]
        outputs = program.compiled(
            params,
            lora,
            batch.tokens,
            batch.mask,
            long_factor,
            batch.cand,
            batch.cand_mask,
        )
        return ScoreOutputs(
            *(np.asarray(output)[: batch.n_real] for output in jax.device_get(outputs))
        )

    # Shape policy.

    def batch_limit(self, key: ShapeKey, cap: int) -> int:
        """
        The largest batch size up to `cap` that fits for `key` (B_eff): `cap` itself
        if it is at most the largest size recorded as fitting; otherwise the first of
        cap, cap // 2, ..., 1 (skipping sizes at or above the smallest recorded as
        failing) whose executables compile without running out of memory and that
        `fits`. The result is recorded; if no size fits, DeviceMemoryError is raised.
        """

        if cap < 1:
            raise ValueError(f"Invalid batch size cap: {cap}")

        size = cap
        while size >= 1:
            if size <= self._fitting.get(key, 0):
                return size
            if size < self._failing.get(key, _NEVER):
                if self._try(key, size):
                    self._fitting[key] = max(self._fitting.get(key, 0), size)
                    return size
                self._failing[key] = size
            size //= 2

        raise DeviceMemoryError(
            f"Not even a single row fits into device memory ({key})."
        )

    def lower_limit(self, key: ShapeKey, failed_b: int) -> int:
        """
        Records that batch size `failed_b` ran out of memory at run time and returns
        the halved B_eff. Raises DeviceMemoryError if a single row failed.
        """

        if failed_b <= 1:
            raise DeviceMemoryError(f"A single row ran out of device memory ({key}).")

        self._failing[key] = min(self._failing.get(key, failed_b), failed_b)
        if self._fitting.get(key, 0) >= failed_b:
            del self._fitting[key]
        return self.batch_limit(key, failed_b // 2)

    def prepare(self, key: ShapeKey, batch_size: int) -> None:
        """
        Compiles the executables of `key` at `batch_size` (a warm-up) and checks that
        they fit, raising DeviceMemoryError otherwise.
        """

        self._ready(key, batch_size)

    def fits(self, key: ShapeKey, batch_size: int) -> bool:
        """
        Whether running `key` at `batch_size` leaves the memory reserve free on every
        device. Always true for devices without memory statistics (CPU).
        """

        stats = _memory_stats(self.plan.devices)
        if stats is None:
            return True

        need = self.memory_need(key, batch_size)
        adapter_bytes = self._default_adapter_bytes() if key.rank is None else 0
        for device_stats in stats:
            bytes_limit = device_stats["bytes_limit"]
            free = bytes_limit - device_stats["bytes_in_use"]
            reserve = MEMORY_RESERVE * bytes_limit + adapter_bytes
            if not self.offload_outputs_to_cpu:
                reserve += OUTPUT_RESERVE * bytes_limit
            if need > free - reserve:
                return False
        return True

    def memory_need(self, key: ShapeKey, batch_size: int) -> int:
        """
        Device bytes a call at (key, B) needs beyond the resident parameters and
        adapters: temporaries, outputs that are not aliased to donated inputs, and the
        per-call inputs. For generation, the prefill outputs (the cache and the
        decoding state) plus the largest temporaries of the prefill, a chunk and a
        re-prefill.
        """

        programs = self._compile(key, batch_size).programs
        analyses = self.memory_analysis(key, batch_size)

        def cost(name: str) -> tuple[int, int]:
            analysis = analyses[name]
            if analysis is None:
                return 0, 0
            return (
                analysis.temp_size_in_bytes,
                analysis.output_size_in_bytes - analysis.alias_size_in_bytes,
            )

        if key.entry in ("generate", "stream"):
            temp, output = cost("prefill")
            for name in ("chunk", "reprefill"):
                if name in programs:
                    temp = max(temp, sum(cost(name)))
            return output + temp + programs["prefill"].transient_bytes

        (program,) = programs.values()
        return sum(cost(key.entry)) + program.transient_bytes

    def memory_analysis(self, key: ShapeKey, batch_size: int) -> dict[str, Any]:
        """
        `memory_analysis()` of each executable of (key, B), by name ("capture",
        "score", or "prefill", "chunk" and, for long RoPE, "reprefill"), compiling
        them if needed. Sizes are per device; None where the backend has none.
        """

        programs = self._compile(key, batch_size).programs
        return {
            name: program.compiled.memory_analysis()
            for name, program in programs.items()
        }

    def release(self) -> None:
        """Empties the executable and batch-limit caches."""

        self._executables.clear()
        self._fitting.clear()
        self._failing.clear()

    # Generation.

    def _generation(
        self,
        params: Params,
        lora: Lora | None,
        batch: TokenBatch,
        spec: DecodeSpec,
        chunk: int,
        entry: str,
        stop_check: StopCheck | None,
    ) -> Generator[_DecodeState, None, Generated]:
        """
        Runs a generation and yields the decoding state after every device call (to
        be read before resuming, because the next call donates it).
        """

        self._check_batch(batch)
        max_new_tokens = spec.max_new_tokens
        if max_new_tokens < 1 or chunk < 1:
            raise ValueError("max_new_tokens and C_chunk must be positive.")

        batch_size, length = batch.tokens.shape
        n_real = batch.n_real
        key = ShapeKey(
            entry,
            length,
            self._rank(lora),
            max_new_tokens=max_new_tokens,
            C_chunk=chunk,
            sampling=spec.sampling,
        )

        switch_steps = self._switch_steps(batch.ref_len)
        if self.arch.rope_switch == "raise" and np.any(switch_steps[:n_real] == 0):
            raise self._dynamic_rope_error()
        long_factor = None
        if self.arch.rope_switch == "long_factor":
            long_factor = switch_steps == 0

        programs = self._ready(key, batch_size)

        # Inputs reused by every call are placed once.
        place = functools.partial(jax.device_put, device=self.plan.replicated)
        tokens, mask = place(batch.tokens), place(batch.mask)
        sampling_arrays = jax.tree.map(place, sampling_params(spec))
        device_switch_steps = None
        if self.arch.rope_switch == "long_factor":
            device_switch_steps = place(switch_steps.astype(np.int32))
        real = np.arange(batch_size) < n_real
        marks = np.zeros(batch_size, dtype=bool)

        state = programs["prefill"].compiled(
            params,
            lora,
            tokens,
            mask,
            np.asarray(batch.ref_len, dtype=np.int32),
            real,
            long_factor,
            sampling_arrays,
        )

        while True:
            step, done = jax.device_get((state.step, state.done))
            step = int(step)
            done = np.array(done[:n_real])
            yield state

            if step >= max_new_tokens or done.all():
                break

            if stop_check is not None:
                generated = np.asarray(state.tokens)[:n_real, :step]
                finish_now = np.asarray(stop_check(generated, done.copy()), dtype=bool)
                marks[:n_real] = finish_now & ~done
                done |= finish_now
                if done.all():
                    break

            # The earliest switch step among unfinished rows that is still ahead.
            pending = switch_steps[:n_real][~done]
            pending = pending[pending >= step]
            if pending.size and pending.min() == step:
                if self.arch.rope_switch == "raise":
                    raise self._dynamic_rope_error()
                state = programs["reprefill"].compiled(
                    params,
                    lora,
                    state,
                    tokens,
                    mask,
                    device_switch_steps,
                    marks.copy(),
                    sampling_arrays,
                )
            else:
                stop_at = pending.min() if pending.size else max_new_tokens
                state = programs["chunk"].compiled(
                    params,
                    lora,
                    state,
                    mask,
                    device_switch_steps,
                    np.int32(stop_at),
                    marks.copy(),
                    sampling_arrays,
                )
            marks[:] = False

        generated_tokens, finish = (
            np.array(array[:n_real])
            for array in jax.device_get((state.tokens, state.finish))
        )
        # Rows marked by the last stop_check, which no device call has seen.
        marked = marks[:n_real]
        finish[marked] = step

        # The cache is no longer needed.
        for array in jax.tree.leaves(state):
            array.delete()

        return Generated(tokens=generated_tokens, finish=finish)

    def _switch_steps(self, ref_len: np.ndarray) -> np.ndarray:
        """
        Each row's RoPE switch step (see "RoPE switching" in docs/DESIGN.md): the first
        decode step whose sequence-length measure `ref_len + s` exceeds the switch
        length, 0 when the prompt already does.
        """

        ref_len = np.asarray(ref_len, dtype=np.int64)
        if self.arch.rope_switch is None:
            return np.full(ref_len.shape, _NEVER, dtype=np.int64)
        return np.maximum(0, self.arch.rope_switch_len - ref_len + 1)

    def _forward_long_factor(self, batch: TokenBatch | ScoreBatch) -> np.ndarray | None:
        """The per-row factor choice of a single forward call."""

        if self.arch.rope_switch is None:
            return None

        switched = self._switch_steps(batch.ref_len) == 0
        if self.arch.rope_switch == "raise":
            if switched[: batch.n_real].any():
                raise self._dynamic_rope_error()
            return None
        return switched

    def _dynamic_rope_error(self) -> NotImplementedError:
        return NotImplementedError(
            "Dynamic RoPE beyond max_position_embeddings "
            f"({self.arch.rope_switch_len} tokens) is not supported."
        )

    # Executables.

    def _ready(self, key: ShapeKey, batch_size: int) -> dict[str, _Program]:
        """The executables of (key, B), checked to fit before their first run."""

        executables = self._compile(key, batch_size)
        if not executables.checked:
            if not self.fits(key, batch_size):
                raise DeviceMemoryError(
                    f"A batch of {batch_size} ({key}) does not fit into device memory."
                )
            executables.checked = True
        return executables.programs

    def _try(self, key: ShapeKey, batch_size: int) -> bool:
        try:
            self._compile(key, batch_size)
        except Exception as error:
            if is_out_of_memory(error):
                return False
            raise
        return self.fits(key, batch_size)

    def _compile(self, key: ShapeKey, batch_size: int) -> _Executables:
        executables = self._executables.get((key, batch_size))
        if executables is None:
            executables = _Executables(self._lower_all(key, batch_size))
            self._executables[(key, batch_size)] = executables
        return executables

    def _lower_all(self, key: ShapeKey, batch_size: int) -> dict[str, _Program]:
        if key.entry not in ENTRIES:
            raise ValueError(f"Unknown entry point: {key.entry}")

        arch = self.arch
        replicated = self.plan.replicated
        params = param_skeleton(arch, self.dtype, self.plan)
        lora = self._lora_skeleton(key.rank)
        tokens = self._abstract((batch_size, key.T), np.int32)
        mask = self._abstract((batch_size, key.T), np.bool_)
        row_flags = self._abstract((batch_size,), np.bool_)
        long_factor = row_flags if arch.rope_switch == "long_factor" else None

        if key.entry == "capture":
            capture = self._lower(
                functools.partial(_capture, arch=arch, want=key.want),
                (params, lora, tokens, mask, long_factor),
                out_shardings=replicated,
            )
            return {"capture": capture}

        if key.entry == "score":
            cand = self._abstract((batch_size, key.G), np.int32)
            cand_mask = self._abstract((batch_size, key.G), np.bool_)
            score = self._lower(
                functools.partial(_score, arch=arch, C_score=key.C_score),
                (params, lora, tokens, mask, long_factor, cand, cand_mask),
                out_shardings=replicated,
            )
            return {"score": score}

        statics = {"arch": arch, "sampling": key.sampling}
        sampling_arrays = self._sampling_skeleton(key.sampling)
        with_cache = key.max_new_tokens > 1
        state = self._state_skeleton(key, batch_size, with_cache)
        state_shardings = jax.tree.map(lambda leaf: leaf.sharding, state)
        ref_len = self._abstract((batch_size,), np.int32)

        programs = {
            "prefill": self._lower(
                functools.partial(
                    _generation_prefill, max_new_tokens=key.max_new_tokens, **statics
                ),
                (params, lora, tokens, mask, ref_len, row_flags, long_factor)
                + (sampling_arrays,),
                out_shardings=state_shardings,
            )
        }
        if not with_cache:
            # A single token needs neither a cache nor a decode step.
            return programs

        switch_steps = None
        if arch.rope_switch == "long_factor":
            switch_steps = self._abstract((batch_size,), np.int32)
        stop_at = self._abstract((), np.int32)

        programs["chunk"] = self._lower(
            functools.partial(_generation_chunk, chunk=key.C_chunk, **statics),
            (params, lora, state, mask, switch_steps, stop_at, row_flags)
            + (sampling_arrays,),
            out_shardings=state_shardings,
            donate_argnums=(2,),
        )
        if arch.rope_switch == "long_factor":
            programs["reprefill"] = self._lower(
                functools.partial(_generation_reprefill, **statics),
                (params, lora, state, tokens, mask, switch_steps, row_flags)
                + (sampling_arrays,),
                out_shardings=state_shardings,
                donate_argnums=(2,),
            )
        return programs

    def _lower(
        self,
        function: Callable,
        args: tuple[Any, ...],
        *,
        out_shardings: Any,
        donate_argnums: tuple[int, ...] = (),
    ) -> _Program:
        compiled = (
            jax.jit(
                function,
                out_shardings=out_shardings,
                donate_argnums=donate_argnums,
                # The re-prefill never reads the donated cache, and JAX would prune it
                # instead of letting XLA write the rebuilt cache into its buffer.
                keep_unused=bool(donate_argnums),
            )
            .lower(*args)
            .compile()
        )

        # Parameters and adapters (the first two arguments) are resident,
        # and donated state was counted where it was created.
        transient = [
            arg
            for index, arg in enumerate(args)
            if index >= 2 and index not in donate_argnums
        ]
        transient_bytes = sum(
            math.prod(leaf.shape) * np.dtype(leaf.dtype).itemsize
            for leaf in jax.tree.leaves(transient)
            if not jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key)
        )
        return _Program(compiled=compiled, transient_bytes=transient_bytes)

    def _abstract(self, shape: tuple[int, ...], dtype: Any) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(shape, dtype, sharding=self.plan.replicated)

    def _lora_skeleton(self, rank: int | None) -> Lora | None:
        if rank is None:
            return None

        lora = {}
        layers = self.arch.num_hidden_layers
        for component in abliterable_components(self.arch):
            d_out, d_in = component_shape(self.arch, component)
            a_sharding, b_sharding = self.plan.lora_shardings(component)
            lora[component] = (
                jax.ShapeDtypeStruct(
                    (layers, 1, rank, d_in), np.float32, sharding=a_sharding
                ),
                jax.ShapeDtypeStruct(
                    (layers, 1, d_out, rank), np.float32, sharding=b_sharding
                ),
            )
        return lora

    def _sampling_skeleton(self, sampling: bool) -> SamplingParams:
        def scalar(dtype: Any) -> jax.ShapeDtypeStruct:
            return self._abstract((), dtype)

        key = None
        if sampling:
            key = self._abstract((), jax.eval_shape(lambda: jax.random.key(0)).dtype)

        return SamplingParams(
            eos_ids=self._abstract((MAX_EOS_IDS,), np.int32),
            pad_id=scalar(np.int32),
            repetition_penalty=scalar(np.float32),
            temperature=scalar(np.float32),
            top_k=scalar(np.int32),
            top_p=scalar(np.float32),
            top_p_complement=scalar(np.float32),
            min_p=scalar(np.float32),
            key=key,
        )

    def _state_skeleton(
        self,
        key: ShapeKey,
        batch_size: int,
        with_cache: bool,
    ) -> _DecodeState:
        arch = self.arch
        cache = None
        if with_cache:
            shape = (
                arch.num_hidden_layers,
                batch_size,
                key.T + key.max_new_tokens,
                arch.num_key_value_heads,
                arch.head_dim,
            )
            part = jax.ShapeDtypeStruct(
                shape, self.dtype, sharding=self.plan.kv_cache_sharding
            )
            cache = (part, part)

        return _DecodeState(
            cache=cache,
            penalised=self._abstract((batch_size, arch.vocab_size), np.bool_),
            tokens=self._abstract((batch_size, key.max_new_tokens), np.int32),
            done=self._abstract((batch_size,), np.bool_),
            finish=self._abstract((batch_size,), np.int32),
            step=self._abstract((), np.int32),
        )

    def _default_adapter_bytes(self) -> int:
        """Per-device bytes of float32 adapters for every component at rank 50."""

        total = 0
        for adapter in self._lora_skeleton(DEFAULT_MAX_LORA_RANK).values():
            for leaf in adapter:
                total += math.prod(leaf.sharding.shard_shape(leaf.shape)) * 4
        return total

    # Validation.

    def _rank(self, lora: Lora | None) -> int | None:
        if lora is None:
            return None

        expected = set(abliterable_components(self.arch))
        if set(lora) != expected:
            raise ValueError(
                f"Adapters must cover exactly the components {sorted(expected)}."
            )
        ranks = {a.shape[2] for a, _ in lora.values()}
        if len(ranks) != 1:
            raise ValueError("All adapters must have the same rank.")
        return ranks.pop()

    @staticmethod
    def _check_batch(batch: TokenBatch | ScoreBatch) -> None:
        batch_size = batch.tokens.shape[0]
        if batch.tokens.ndim != 2 or batch.mask.shape != batch.tokens.shape:
            raise ValueError("tokens and mask must be [B, T].")
        if batch.tokens.dtype != np.int32 or batch.mask.dtype != np.bool_:
            raise ValueError("tokens must be int32 and mask bool.")
        if np.shape(batch.ref_len) != (batch_size,):
            raise ValueError("ref_len must be [B].")
        if not 1 <= batch.n_real <= batch_size:
            raise ValueError(f"n_real must be between 1 and {batch_size}.")


def _memory_stats(devices: Sequence[jax.Device]) -> list[dict[str, int]] | None:
    """The memory statistics of every device, or None if a device reports none."""

    stats = [device.memory_stats() for device in devices]
    if any(device_stats is None for device_stats in stats):
        return None
    return stats
