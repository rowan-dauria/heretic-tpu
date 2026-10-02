# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
Device memory of the engine's executables, from `memory_analysis()` of programs
compiled for abstract inputs (nothing is allocated): the KV cache costs one copy in the
generation prefill and is updated in place by the decode chunk and the long-RoPE
re-prefill, and neither scoring nor capturing materialises logits for every column.
"""

import dataclasses
from typing import Any

import numpy as np
import pytest

from heretic_tpu.backend.arch import RopeSpec
from heretic_tpu.backend.engine import Engine, ShapeKey
from heretic_tpu.backend.sharding import choose_plan
from tests.backend.test_engine import synthetic_arch

BATCH = 8
PROMPT = 32

# Deep enough that one layer's cache slice (which a decode step may materialise) is
# a small part of the cache.
LAYERS = 16

# Bound on the bytes of an output tuple's index table, which is never aliased.
TUPLE_BYTES = 128

# Attention shapes, with the cache lengths to compile at: the narrow heads of the
# synthetic architecture, and heads as wide as those of real models (TPU layouts pad
# narrow heads, and XLA:TPU chooses the cache layout by head shape).
HEADS = {
    "narrow": ({}, (4096, 8192)),
    "wide": (
        {
            "hidden_size": 512,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rope": (
                RopeSpec(
                    rope_type="default",
                    rot=128,
                    params=(("rope_theta", 10000.0), ("rope_type", "default")),
                ),
            )
            * LAYERS,
        },
        # Cache lengths that are not multiples of 8, as with real prompts.
        (1028, 2052),
    ),
}


def _engine(**changes: Any) -> Engine:
    arch = synthetic_arch(**{"num_hidden_layers": LAYERS, **changes})
    return Engine(arch, choose_plan(arch, np.float32, "single"), np.float32)


def _analyses(eng: Engine, key: ShapeKey) -> dict[str, Any]:
    analyses = eng.memory_analysis(key, BATCH)
    if any(analysis is None for analysis in analyses.values()):
        pytest.skip("no memory analysis on this backend")
    return analyses


def _cache_bytes(eng: Engine, max_new_tokens: int) -> int:
    arch = eng.arch
    return (
        2
        * arch.num_hidden_layers
        * BATCH
        * (PROMPT + max_new_tokens)
        * arch.num_key_value_heads
        * arch.head_dim
        * 4
    )


def _assert_in_place(analysis: Any, state_bytes: int) -> None:
    """
    The donated decoding state is the output, except for the output tuple. (Device
    layouts may pad buffers, so the unpadded size is a lower bound.)
    """

    assert analysis.alias_size_in_bytes >= state_bytes
    alias = analysis.alias_size_in_bytes
    assert 0 <= analysis.output_size_in_bytes - alias <= TUPLE_BYTES


def _state_bytes(eng: Engine, max_new_tokens: int) -> int:
    """The decoding state: the cache, the penalised set and the token buffer, etc."""

    return (
        _cache_bytes(eng, max_new_tokens)
        + BATCH * eng.arch.vocab_size
        + BATCH * max_new_tokens * 4
        + BATCH * (1 + 4)
        + 4
    )


@pytest.mark.parametrize("heads", sorted(HEADS))
@pytest.mark.parametrize("sampling", [False, True])
def test_kv_cache_is_built_once_and_updated_in_place(
    sampling: bool, heads: str
) -> None:
    """
    On a configuration whose cache dwarfs its activations, compiled at two cache
    lengths: the prefill (which allocates the cache) costs about one cache, and the
    chunk, whose decoding state is donated, updates it in place (in particular, it
    never copies the cache into another layout).
    """

    changes, (short, long) = HEADS[heads]
    eng = _engine(**changes)
    prefill_cost = {}
    chunk_temp = {}
    for max_new_tokens in (short, long):
        key = ShapeKey(
            "generate",
            PROMPT,
            None,
            max_new_tokens=max_new_tokens,
            C_chunk=32,
            sampling=sampling,
        )
        analyses = _analyses(eng, key)

        prefill = analyses["prefill"]
        prefill_cost[max_new_tokens] = (
            prefill.temp_size_in_bytes
            + prefill.output_size_in_bytes
            - prefill.alias_size_in_bytes
        )

        chunk = analyses["chunk"]
        _assert_in_place(chunk, _state_bytes(eng, max_new_tokens))
        chunk_temp[max_new_tokens] = chunk.temp_size_in_bytes

    cache_delta = _cache_bytes(eng, long) - _cache_bytes(eng, short)
    assert prefill_cost[long] - prefill_cost[short] <= 1.25 * cache_delta
    assert chunk_temp[long] - chunk_temp[short] <= 0.25 * cache_delta

    # The engine's estimate counts the cache once.
    need = eng.memory_need(key, BATCH)
    assert _cache_bytes(eng, long) <= need <= 1.5 * _cache_bytes(eng, long)


def test_long_rope_reprefill_rebuilds_the_cache_in_place() -> None:
    head_dim = 16
    factors = tuple(1.0 + i / 8 for i in range(head_dim // 2))
    rope = RopeSpec(
        rope_type="longrope",
        rot=head_dim,
        params=(
            ("long_factor", factors),
            ("original_max_position_embeddings", 4096),
            ("rope_theta", 10000.0),
            ("rope_type", "longrope"),
            ("short_factor", factors),
        ),
    )
    eng = _engine(
        rope=(rope,) * LAYERS, rope_switch="long_factor", rope_switch_len=4096
    )

    key = ShapeKey("generate", PROMPT, 2, max_new_tokens=1024, C_chunk=32)
    analyses = _analyses(eng, key)
    assert set(analyses) == {"prefill", "chunk", "reprefill"}
    _assert_in_place(analyses["reprefill"], _state_bytes(eng, 1024))


def test_single_token_generation_needs_no_cache() -> None:
    eng = _engine()
    key = ShapeKey("generate", PROMPT, None, max_new_tokens=1, C_chunk=32)
    analyses = _analyses(eng, key)

    # Only the prefill, whose output is the decoding state without a cache.
    assert set(analyses) == {"prefill"}
    output = analyses["prefill"].output_size_in_bytes
    assert _state_bytes(eng, 1) - _cache_bytes(eng, 1) <= output
    assert output < _cache_bytes(eng, 1) / 8


def test_scoring_and_capture_never_hold_logits_for_every_column() -> None:
    """
    With a vocabulary that dwarfs everything else: scoring every column needs no more
    temporary memory than scoring one (the head is applied column by column), and
    neither entry point comes near the size of all logits.
    """

    eng = _engine(vocab_size=2**16, num_hidden_layers=2)
    length = 256
    all_logits = BATCH * length * eng.arch.vocab_size * 4
    one_column = BATCH * eng.arch.vocab_size * 4

    score_temp = {}
    for columns in (1, length):
        key = ShapeKey("score", length, None, C_score=columns, G=4)
        score_temp[columns] = _analyses(eng, key)["score"].temp_size_in_bytes
    assert score_temp[length] - score_temp[1] < one_column
    assert score_temp[length] < all_logits / 8

    want = frozenset({"logits", "hidden", "module_io"})
    analyses = _analyses(eng, ShapeKey("capture", length, 4, want=want))
    assert analyses["capture"].temp_size_in_bytes < all_logits / 8


def test_memory_need_matches_the_programs() -> None:
    eng = _engine()
    key = ShapeKey("capture", PROMPT, None, want=frozenset({"logits"}))
    (analysis,) = _analyses(eng, key).values()
    transient = BATCH * PROMPT * (4 + 1)
    assert eng.memory_need(key, BATCH) == (
        analysis.temp_size_in_bytes
        + analysis.output_size_in_bytes
        - analysis.alias_size_in_bytes
        + transient
    )

    # Deeper models need more only through their activations and outputs.
    deeper = dataclasses.replace(
        eng.arch,
        num_hidden_layers=2 * LAYERS,
        layer_types=eng.arch.layer_types * 2,
        windows=eng.arch.windows * 2,
        rope=eng.arch.rope * 2,
    )
    deeper_engine = Engine(
        deeper, choose_plan(deeper, np.float32, "single"), np.float32
    )
    assert deeper_engine.memory_need(key, BATCH) < 2 * eng.memory_need(key, BATCH)
