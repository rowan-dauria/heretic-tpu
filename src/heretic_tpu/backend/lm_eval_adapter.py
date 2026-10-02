# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
JaxLM, an lm-evaluation-harness model backed by the engine (see "lm-eval adapter
(JaxLM)" in docs/DESIGN.md).

JaxLM reproduces HFLM as upstream heretic constructs it (`HFLM(pretrained, tokenizer,
batch_size="auto")`, so no BOS, length or truncation overrides and no chat template):
the same tokenisation, request grouping, left truncation, stop sequences and
post-processing, with forward passes and generation running on the engine. Batches
follow the engine shape policy (grouped by bucket, at most 64 rows, filler rows)
instead of HFLM's out-of-memory probing.

The model is resolved through `state_fn` at the start of every request call and on
every read of `max_length`, and no reference to the engine, parameters or adapters is
kept between calls. Adapters, `lora_disabled()` and model reloads therefore take
effect without rebuilding the object.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypeVar

import numpy as np
from lm_eval.api.model import TemplateLM
from lm_eval.models.utils import (
    DEFAULT_MAX_LENGTH,
    Collator,
    _add_special_kwargs,
    handle_stop_sequences,
    has_bos_prefix,
    normalize_gen_kwargs,
    postprocess_generated_text,
    resolve_max_length,
)
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window
from tqdm import tqdm

from .engine import (
    GENERATE_CHUNK,
    Engine,
    ShapeKey,
    StopCheck,
    bucket_length,
    decode_spec,
    is_out_of_memory,
    next_power_of_two,
    score_batch,
    token_batch,
)

if TYPE_CHECKING:
    import jax
    from lm_eval.api.instance import Instance
    from transformers import (
        GenerationConfig,
        PretrainedConfig,
        PreTrainedTokenizerBase,
    )

    from .transformer import Lora
    from .weights import Params

# A child of lm-eval's logger, so that the log level heretic-tpu sets for lm-eval
# applies to these messages as it does to HFLM's.
eval_logger = logging.getLogger("lm_eval.models.heretic_tpu")

# The batch-size cap of the engine shape policy for JaxLM.
MAX_BATCH_SIZE = 64

# HFLM's default number of generated tokens, used when a task sets none.
MAX_GEN_TOKS = 256

T = TypeVar("T")


class LMState(Protocol):
    """
    The model state JaxLM evaluates: the interface of `heretic_tpu.model.LMState`,
    which the backend cannot import.
    """

    @property
    def engine(self) -> Engine: ...

    @property
    def params(self) -> Params: ...

    # None runs the model without adapters.
    @property
    def lora(self) -> Lora | None: ...

    # The top-level resolved configuration of the current model.
    @property
    def hf_config(self) -> PretrainedConfig: ...

    # Resolved from the current model's checkpoint.
    @property
    def generation_config(self) -> GenerationConfig: ...

    # tokenizer.pad_token_id
    @property
    def pad_id(self) -> int: ...

    # Called once per sampling batch for its PRNG key.
    @property
    def next_key(self) -> Callable[[], jax.Array]: ...


# A loglikelihood request: the original strings (None for rolling windows), and the
# context and continuation tokens.
TokenRequest = tuple[tuple[str, str] | None, list[int], list[int]]


class _ScoreRow(NamedTuple):
    """
    One forward pass, shared by the requests whose context plus continuation without
    its last token are equal.
    """

    # Request indices.
    members: list[int]

    # The input tokens: context and continuation without the last token, cut to
    # max_length from the left.
    inp: list[int]

    # The distinct last continuation tokens of the members, scored at the last column.
    candidates: list[int]

    # Length of the longest continuation among the members.
    continuation_length: int


class JaxLM(TemplateLM):
    """
    An lm-eval model that evaluates the model `state_fn()` returns at request time.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        state_fn: Callable[[], LMState],
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self._state_fn = state_fn

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self) -> int:
        # Used as the context of loglikelihood requests with an empty context.
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        # Evaluated on every access, so that it follows model reloads.
        return self._max_length(self._state_fn())

    def _max_length(self, state: LMState) -> int:
        return resolve_max_length(
            state.hf_config, self.tokenizer, default=DEFAULT_MAX_LENGTH
        )

    @property
    def max_gen_toks(self) -> int:
        return MAX_GEN_TOKS

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        left_truncate_len: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        special_tokens_kwargs = _add_special_kwargs(add_special_tokens, None)
        # A string that already starts with the BOS token gets no second one.
        if add_special_tokens is None and has_bos_prefix(
            string, self.tokenizer.decode(self.prefix_token_id)
        ):
            special_tokens_kwargs["add_special_tokens"] = False
        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_decode(
        self,
        tokens: int | Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    # Loglikelihoods.

    def _loglikelihood_tokens(
        self,
        requests: list[TokenRequest],
        disable_tqdm: bool = False,
        **kwargs: Any,
    ) -> list[tuple[float, bool]]:
        return self._score(self._state_fn(), requests, disable_tqdm)

    def loglikelihood_rolling(
        self,
        requests: list[Instance],
        disable_tqdm: bool = False,
    ) -> list[float]:
        state = self._state_fn()
        max_length = self._max_length(state)

        windows: list[TokenRequest] = []
        owners = []
        for index, (string,) in enumerate(
            tqdm([request.args for request in requests], disable=disable_tqdm)
        ):
            for context_enc, continuation_enc in map(
                make_disjoint_window,
                get_rolling_token_windows(
                    token_list=self.tok_encode(string),
                    prefix_token=self.prefix_token_id,
                    max_seq_len=max_length,
                    context_len=1,
                ),
            ):
                # Windows have no strings, so they are not cached individually.
                windows.append((None, context_enc, continuation_enc))
                owners.append(index)

        totals = [0.0] * len(requests)
        for owner, (logprob, _) in zip(
            owners, self._score(state, windows, disable_tqdm), strict=True
        ):
            totals[owner] += logprob

        for request, total in zip(requests, totals, strict=True):
            self.cache_hook.add_partial("loglikelihood_rolling", request.args, total)

        return totals

    def _score(
        self,
        state: LMState,
        requests: list[TokenRequest],
        disable_tqdm: bool,
    ) -> list[tuple[float, bool]]:
        max_length = self._max_length(state)

        # Requests whose context plus continuation without its last token are equal
        # (typically single-token answers to one question) share a forward pass, as
        # with HFLM's "contexts" grouping: their last tokens are scored as candidates
        # at the last column.
        groups: dict[tuple[int, ...], list[int]] = {}
        for index, (_, context_enc, continuation_enc) in enumerate(requests):
            if not context_enc or not continuation_enc:
                raise ValueError("Contexts and continuations must not be empty.")
            if len(continuation_enc) > max_length:
                raise ValueError(
                    f"A continuation of {len(continuation_enc)} tokens is longer than "
                    f"the model's maximum length ({max_length})."
                )
            key = tuple(context_enc + continuation_enc[:-1])
            groups.setdefault(key, []).append(index)

        ordered_rows = []
        for members in groups.values():
            # The (first) member with the longest continuation provides the input,
            # which depends only on the group's key.
            representative = max(members, key=lambda index: len(requests[index][2]))
            _, context_enc, continuation_enc = requests[representative]
            tokens = context_enc + continuation_enc
            if len(tokens) > max_length + 1:
                eval_logger.warning(
                    f"Combined length of context ({len(context_enc)}) and "
                    f"continuation ({len(continuation_enc)}) exceeds model's maximum "
                    f"length ({max_length}). Truncating "
                    f"{len(tokens) - max_length + 1} tokens from the left."
                )
            row = _ScoreRow(
                members=members,
                inp=tokens[-(max_length + 1) :][:-1],
                candidates=list(
                    dict.fromkeys(requests[index][2][-1] for index in members)
                ),
                continuation_length=len(continuation_enc),
            )
            # HFLM's order: longest first, so that batches hold rows of similar
            # lengths.
            ordered_rows.append(((-len(tokens), tuple(tokens)), row))

        ordered_rows.sort(key=lambda item: item[0])

        # Rows sharing a prompt bucket and a continuation bucket share executables.
        buckets: dict[tuple[int, int], list[_ScoreRow]] = {}
        for _, row in ordered_rows:
            shape = (
                bucket_length(len(row.inp)),
                next_power_of_two(row.continuation_length),
            )
            buckets.setdefault(shape, []).append(row)

        results: list[tuple[float, bool] | None] = [None] * len(requests)
        progress = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )

        def run(
            key: ShapeKey, batch_rows: Sequence[_ScoreRow], batch_size: int
        ) -> None:
            # HFLM pads its batch to the longest row and counts positions from the
            # start of each row, so the reference length is the batch's.
            longest = max(len(row.inp) for row in batch_rows)
            batch = score_batch(
                [row.inp for row in batch_rows],
                [longest] * len(batch_rows),
                [row.candidates for row in batch_rows],
                pad_id=state.pad_id,
                C_score=key.C_score,
                G=key.G,
                batch_size=batch_size,
                length=key.T,
            )
            outputs = state.engine.score(state.params, state.lora, batch)

            for index, row in enumerate(batch_rows):
                for member in row.members:
                    continuation_enc = requests[member][2]
                    # The continuation's tokens before the last are the inputs of
                    # the last columns, each scored at the column before it.
                    span = slice(key.C_score - len(continuation_enc), key.C_score - 1)
                    candidate = row.candidates.index(continuation_enc[-1])
                    logprob = np.sum(
                        outputs.next_lp[index, span], dtype=np.float64
                    ) + float(outputs.last_lp[index, candidate])
                    is_greedy = bool(
                        np.all(outputs.next_greedy[index, span])
                        and outputs.last_greedy[index, candidate]
                    )
                    results[member] = (float(logprob), is_greedy)

            progress.update(sum(len(row.members) for row in batch_rows))

        for (length, C_score), bucket in buckets.items():
            G = next_power_of_two(max(len(row.candidates) for row in bucket))
            key = ShapeKey(
                "score", length, state.engine.rank(state.lora), C_score=C_score, G=G
            )
            _run_batched(state.engine, key, bucket, run)

        progress.close()

        for (request_str, _, _), answer in zip(requests, results, strict=True):
            # Rolling windows are cached per string by loglikelihood_rolling.
            if request_str is not None:
                self.cache_hook.add_partial("loglikelihood", request_str, answer)

        return results

    # Generation.

    def generate_until(
        self,
        requests: list[Instance],
        disable_tqdm: bool = False,
    ) -> list[str]:
        state = self._state_fn()
        max_length = self._max_length(state)
        eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)

        # Requests are grouped by their generation kwargs, each group taken whole,
        # longest context first.
        collator = Collator(
            [request.args for request in requests],
            sort_fn=lambda args: (-len(self.tok_encode(args[0])), args[0]),
            group_by="gen_kwargs",
            group_fn=lambda args: args[1],
        )
        progress = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running generate_until requests",
        )

        results = []
        for chunk in collator.get_batched(n=0):
            contexts, all_gen_kwargs = zip(*chunk, strict=True)
            gen_kwargs = all_gen_kwargs[0]
            if not isinstance(gen_kwargs, dict):
                raise TypeError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )

            kwargs = normalize_gen_kwargs(gen_kwargs, self.max_gen_toks)
            until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            max_gen_toks = kwargs.pop("max_gen_toks")

            # Room for max_gen_toks tokens after the context.
            max_ctx_len = max_length - max_gen_toks
            if max_ctx_len <= 0:
                raise ValueError(
                    f"Invalid configuration: requested max tokens to generate "
                    f"({max_gen_toks}) must be less than model's maximum sequence "
                    f"length ({max_length})."
                )

            if "max_length" in kwargs:
                # HFLM would take it as the length of the padded batch plus the
                # generated tokens, which depends on its batches.
                eval_logger.warning(
                    f"Ignoring `max_length` in generation kwargs; generating up to "
                    f"max_gen_toks ({max_gen_toks}) tokens."
                )
                kwargs.pop("max_length")

            texts = self._generate(
                state,
                [self._encode_context(context, max_ctx_len) for context in contexts],
                kwargs,
                max_gen_toks,
                until,
                progress,
            )
            for context, text in zip(contexts, texts, strict=True):
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), text
                )
            results.extend(texts)

        progress.close()

        return collator.get_original(results)

    def _encode_context(self, context: str, max_ctx_len: int) -> list[int]:
        # As HFLM's batch encoding, which looks for the BOS token itself (tok_encode
        # looks for the decoded prefix token), but per context instead of for the
        # first context of a batch.
        kwargs = {}
        if has_bos_prefix(context, getattr(self.tokenizer, "bos_token", None)):
            kwargs["add_special_tokens"] = False
        tokens = self.tokenizer.encode(context, **kwargs)

        if len(tokens) > max_ctx_len:
            eval_logger.warning(
                f"Left truncation applied. Original sequence length was {len(tokens)}, "
                f"truncating to last {max_ctx_len} tokens. Some content will be lost."
            )
        return tokens[-max_ctx_len:]

    def _generate(
        self,
        state: LMState,
        encoded: list[list[int]],
        kwargs: dict[str, Any],
        max_gen_toks: int,
        until: list[str],
        progress: tqdm,
    ) -> list[str]:
        """Generates for the encoded contexts of one kwargs group, in their order."""

        stops = [stop for stop in until if stop]
        stop_check = self._stop_check(stops) if stops else None
        texts: list[str | None] = [None] * len(encoded)

        # Rows sharing a prompt bucket share executables.
        buckets: dict[int, list[int]] = {}
        for index, tokens in enumerate(encoded):
            buckets.setdefault(bucket_length(len(tokens)), []).append(index)

        def run(key: ShapeKey, batch_indices: Sequence[int], batch_size: int) -> None:
            rows = [encoded[index] for index in batch_indices]
            # HFLM left-pads its batch to the longest row, which decides whether the
            # padding token is penalised and the Phi-3 RoPE factors.
            longest = max(len(row) for row in rows)
            batch = token_batch(
                rows,
                [longest] * len(rows),
                pad_id=state.pad_id,
                batch_size=batch_size,
                length=key.T,
            )
            # One sampling key per batch.
            spec = decode_spec(
                state.generation_config,
                kwargs,
                max_new_tokens=key.max_new_tokens,
                chunk=key.C_chunk,
                pad_id=state.pad_id,
                key=state.next_key() if key.sampling else None,
            )
            generated = state.engine.generate(
                state.params, state.lora, batch, spec, stop_check
            )

            for index, tokens, finish in zip(
                batch_indices, generated.tokens, generated.finish, strict=True
            ):
                text = self.tok_decode(tokens[:finish].tolist())
                texts[index] = postprocess_generated_text(text, until, None)

            progress.update(len(batch_indices))

        for length, indices in buckets.items():
            key = ShapeKey(
                "generate",
                length,
                state.engine.rank(state.lora),
                max_new_tokens=max_gen_toks,
                C_chunk=GENERATE_CHUNK,
                sampling=bool(kwargs["do_sample"]),
            )
            _run_batched(state.engine, key, indices, run)

        return texts

    def _stop_check(self, stops: list[str]) -> StopCheck:
        """
        Marks rows whose text contains a stop sequence as finished. This only ends
        batches earlier: rows are independent, and post-processing discards
        everything from the first stop sequence on.
        """

        def stop_check(tokens: np.ndarray, done: np.ndarray) -> np.ndarray:
            finish_now = np.zeros(len(tokens), dtype=bool)
            for row in np.flatnonzero(~done):
                text = self.tok_decode(tokens[row].tolist())
                finish_now[row] = any(stop in text for stop in stops)
            return finish_now

        return stop_check


def _run_batched(
    engine: Engine,
    key: ShapeKey,
    items: Sequence[T],
    run: Callable[[ShapeKey, Sequence[T], int], None],
) -> None:
    """
    Runs `items` (rows of one shape key, in order) through `run(key, rows,
    batch_size)` in engine batches as the shape policy says: up to B_eff rows per
    batch, a smaller last batch padded to a power of two, and halving B_eff when a
    batch runs out of memory, after which the batch is run again in smaller batches.
    """

    limit = engine.batch_limit(key, min(MAX_BATCH_SIZE, next_power_of_two(len(items))))
    start = 0
    while start < len(items):
        stop = min(start + limit, len(items))
        batch_size = min(limit, next_power_of_two(stop - start))
        try:
            run(key, items[start:stop], batch_size)
        except Exception as error:
            if not is_out_of_memory(error):
                raise
            limit = engine.lower_limit(key, batch_size)
            continue
        start = stop
