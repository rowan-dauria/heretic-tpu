# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
Decoder primitives: linear layers and LoRA, RMS norms, activations, RoPE, attention
and the routed mixture-of-experts MLP.

Numerics follow the transformers eager implementations (see "Forward pass semantics"
in docs/DESIGN.md). PyTorch evaluates an operation between a reduced-precision tensor
and a Python number in float32 and rounds the result once, so such operations are
written here as explicit float32 computations followed by a cast. For float32 models
every cast is a no-op.
"""

import math

import jax
import jax.numpy as jnp
from jax import lax

# The float32 LoRA path must not be computed in a single bfloat16 pass on TPU.
HIGHEST = lax.Precision.HIGHEST

# Query block sizes of blocked attention (see `query_block_size`).
QUERY_BLOCK = 512
LONG_QUERY_BLOCK = 256
LONG_SEQUENCE = 8192


def linear(x: jax.Array, weight: jax.Array, bias: jax.Array | None = None) -> jax.Array:
    """
    `x @ weightᵀ (+ bias)` for a weight in Hugging Face orientation [out, in]:
    accumulated in float32 and rounded to the dtype of `x`, then the bias is added in
    that dtype.
    """

    y = jnp.einsum(
        "...i,oi->...o",
        x,
        weight,
        preferred_element_type=jnp.float32,
    ).astype(x.dtype)
    if bias is not None:
        y = y + bias
    return y


def adapted_linear(
    x: jax.Array,
    weight: jax.Array,
    bias: jax.Array | None,
    adapter: tuple[jax.Array, jax.Array] | None,
) -> jax.Array:
    """
    A linear layer with an optional float32 LoRA adapter (A [r, in], B [out, r]),
    computed as PEFT does for an fp32 adapter with `lora_alpha = r`:
    `(base(x).astype(f32) + (x.astype(f32) @ Aᵀ) @ Bᵀ).astype(x.dtype)`.
    """

    y = linear(x, weight, bias)
    if adapter is None:
        return y

    a, b = adapter
    down = jnp.einsum("...i,ri->...r", x.astype(jnp.float32), a, precision=HIGHEST)
    delta = jnp.einsum("...r,or->...o", down, b, precision=HIGHEST)
    return (y.astype(jnp.float32) + delta).astype(x.dtype)


def rms_norm(
    x: jax.Array,
    weight: jax.Array,
    eps: float,
    *,
    gemma: bool = False,
) -> jax.Array:
    """
    RMSNorm over the last axis. Llama style normalises in float32, casts, then
    multiplies by the weight in the model dtype; Gemma style multiplies by
    `(1 + weight)` in float32 and casts last.
    """

    dtype = x.dtype
    hidden = x.astype(jnp.float32)
    variance = jnp.mean(hidden * hidden, axis=-1, keepdims=True)
    hidden = hidden * lax.rsqrt(variance + eps)

    if gemma:
        return (hidden * (1.0 + weight.astype(jnp.float32))).astype(dtype)
    return weight * hidden.astype(dtype)


def activation(name: str, x: jax.Array) -> jax.Array:
    """
    The MLP activation, evaluated in float32 and rounded once, written as PyTorch's
    CPU kernels write it.
    """

    h = x.astype(jnp.float32)

    if name == "silu":
        y = h / (1.0 + jnp.exp(-h))
    elif name == "gelu":
        # Exact (erf) GELU.
        y = h * 0.5 * (1.0 + lax.erf(h * math.sqrt(0.5)))
    elif name == "gelu_pytorch_tanh":
        beta = math.sqrt(2.0) * (2.0 / math.sqrt(math.pi)) * 0.5
        inner = beta * (h + 0.044715 * (h * h * h))
        y = 0.5 * h * (1.0 + jnp.tanh(inner))
    else:
        raise ValueError(f"Unsupported activation: {name}")

    return y.astype(x.dtype)


def softcap(x: jax.Array, cap: float) -> jax.Array:
    """`tanh(x / cap) * cap`, rounding to the dtype of `x` after each operation."""

    dtype = x.dtype
    x = (x.astype(jnp.float32) / cap).astype(dtype)
    x = jnp.tanh(x.astype(jnp.float32)).astype(dtype)
    return (x.astype(jnp.float32) * cap).astype(dtype)


def rope_cos_sin(
    positions: jax.Array,
    inv_freq: jax.Array,
    attention_scaling: jax.Array,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
    """
    cos and sin [B, T, rot] of RoPE for integer positions [B, T] and per-row inverse
    frequencies [B, rot/2], computed in float32 from the elementwise product
    `position * inv_freq`, multiplied by the attention scaling and cast to `dtype`.
    """

    freqs = positions.astype(jnp.float32)[:, :, None] * inv_freq[:, None, :]
    # cos(concat(f, f)) == concat(cos(f), cos(f)) elementwise.
    cos = jnp.cos(freqs) * attention_scaling
    sin = jnp.sin(freqs) * attention_scaling
    cos = jnp.concatenate([cos, cos], axis=-1).astype(dtype)
    sin = jnp.concatenate([sin, sin], axis=-1).astype(dtype)
    return cos, sin


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """
    Rotates heads x [B, T, heads, hd] on their first `rot` dimensions
    (cos and sin are [B, T, rot]); the remaining dimensions pass through.
    """

    rot = cos.shape[-1]
    half = rot // 2
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]

    rotated = x[..., :rot]
    rotated_half = jnp.concatenate([-rotated[..., half:], rotated[..., :half]], axis=-1)
    rotated = rotated * cos + rotated_half * sin

    if rot == x.shape[-1]:
        return rotated
    return jnp.concatenate([rotated, x[..., rot:]], axis=-1)


def query_block_size(length: int) -> int:
    """Number of queries per block of blocked attention over `length` queries."""

    if length > LONG_SEQUENCE:
        return LONG_QUERY_BLOCK
    return min(length, QUERY_BLOCK)


def _attention_probs(
    scores: jax.Array,
    visible: jax.Array,
    scale: float,
    cap: float | None,
) -> jax.Array:
    """
    Attention probabilities in the model dtype from raw scores `q @ kᵀ` (in the model
    dtype): scaled, soft-capped, masked with the dtype's minimum (after the cap, so
    masked entries are never squashed to `-cap`) and normalised in float32.
    A row without visible entries gives a uniform distribution, never NaN.
    """

    dtype = scores.dtype
    scores = (scores.astype(jnp.float32) * scale).astype(dtype)
    if cap is not None:
        scores = softcap(scores, cap)
    scores = jnp.where(visible, scores, jnp.finfo(dtype).min)
    return jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(dtype)


def attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_mask: jax.Array,
    window: jax.Array,
    *,
    scale: float,
    cap: float | None,
) -> jax.Array:
    """
    Causal self-attention of queries q [B, T, H, hd] over keys and values
    k, v [B, T, KV, hd] occupying slots 0 to T - 1, blocked over queries.

    Key slot j is visible from query slot i iff `kv_mask[b, j]`, `j <= i` and
    `i - j < window`. Each block of `query_block_size(T)` queries builds its mask from
    slot indices, so temporaries are [B, H, q_blk, T] and never [B, H, T, T]. If the
    block size does not divide T, the query axis is padded with masked queries whose
    outputs are discarded. Returns [B, T, H·hd] in the model dtype.
    """

    batch, length, heads, head_dim = q.shape
    kv_heads = k.shape[2]
    dtype = q.dtype

    # Grouped-query attention: query head h uses key/value head h // (H / KV),
    # as transformers' repeat_kv arranges them.
    q = q.reshape(batch, length, kv_heads, heads // kv_heads, head_dim)

    block = query_block_size(length)
    blocks = -(-length // block)
    padded = blocks * block
    if padded != length:
        q = jnp.pad(q, ((0, 0), (0, padded - length), (0, 0), (0, 0), (0, 0)))

    key_slots = jnp.arange(length)

    def attend_block(q_block: jax.Array, start: jax.Array) -> jax.Array:
        query_slots = start + jnp.arange(block)
        visible = (
            kv_mask[:, None, :]
            & (key_slots[None, :] <= query_slots[:, None])
            & (query_slots[:, None] - key_slots[None, :] < window)
            & (query_slots < length)[:, None]
        )
        scores = jnp.einsum(
            "bqkgd,btkd->bkgqt",
            q_block,
            k,
            preferred_element_type=jnp.float32,
        ).astype(dtype)
        probs = _attention_probs(scores, visible[:, None, None], scale, cap)
        return jnp.einsum(
            "bkgqt,btkd->bqkgd",
            probs,
            v,
            preferred_element_type=jnp.float32,
        ).astype(dtype)

    if blocks == 1:
        out = attend_block(q, jnp.int32(0))
    else:
        q_blocks = jnp.swapaxes(
            q.reshape(batch, blocks, block, *q.shape[2:]),
            0,
            1,
        )
        starts = jnp.arange(blocks, dtype=jnp.int32) * block
        # Without batch_size, lax.map runs the blocks sequentially.
        out = lax.map(lambda args: attend_block(*args), (q_blocks, starts))
        out = jnp.swapaxes(out, 0, 1).reshape(batch, padded, *q.shape[2:])

    return out[:, :length].reshape(batch, length, heads * head_dim)


def decode_attention(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    k_new: jax.Array,
    v_new: jax.Array,
    kv_mask: jax.Array,
    slot: jax.Array,
    window: jax.Array,
    *,
    scale: float,
    cap: float | None,
) -> jax.Array:
    """
    Attention of one new token per row, q [B, H, hd] at cache slot `slot`, over the
    cached keys and values [B, KV, T_cache, hd] before that slot plus the token's own
    key and value k_new, v_new [B, KV, hd] as an extra column.

    Cached slot j is visible iff `kv_mask[b, j]`, `j < slot` and `slot - j < window`;
    the extra column is always visible and gets the same scaling and soft-capping.
    Returns [B, H·hd] in the model dtype.
    """

    batch, heads, head_dim = q.shape
    kv_heads, cache_len = k_cache.shape[1:3]
    dtype = q.dtype

    q = q.reshape(batch, kv_heads, heads // kv_heads, head_dim)
    k_new = k_new.astype(k_cache.dtype)
    v_new = v_new.astype(v_cache.dtype)

    key_slots = jnp.arange(cache_len)
    visible = kv_mask & (key_slots < slot) & (slot - key_slots < window)
    visible = jnp.concatenate([visible, jnp.ones((batch, 1), dtype=bool)], axis=-1)

    cached_scores = jnp.einsum(
        "bkgd,bktd->bkgt",
        q,
        k_cache,
        preferred_element_type=jnp.float32,
    ).astype(dtype)
    new_scores = jnp.einsum(
        "bkgd,bkd->bkg",
        q,
        k_new,
        preferred_element_type=jnp.float32,
    ).astype(dtype)
    scores = jnp.concatenate([cached_scores, new_scores[..., None]], axis=-1)
    probs = _attention_probs(scores, visible[:, None, None], scale, cap)

    # Both parts are accumulated in float32 and rounded once, like one matmul.
    out = jnp.einsum(
        "bkgt,bktd->bkgd",
        probs[..., :cache_len],
        v_cache,
        preferred_element_type=jnp.float32,
    )
    new_values = v_new[:, :, None, :].astype(jnp.float32)
    out = out + probs[..., cache_len:].astype(jnp.float32) * new_values
    return out.astype(dtype).reshape(batch, heads * head_dim)


def gated_mlp(
    x: jax.Array,
    gate: jax.Array,
    up: jax.Array,
    gate_bias: jax.Array | None,
    up_bias: jax.Array | None,
    activation_name: str,
) -> jax.Array:
    """The input of the down projection, `act(gate(x)) * up(x)`, in the model dtype."""

    return activation(activation_name, linear(x, gate, gate_bias)) * linear(
        x, up, up_bias
    )


def _grouped_linear(
    x: jax.Array,
    weights: jax.Array,
    group_sizes: jax.Array,
) -> jax.Array:
    """
    `linear` for rows x [M, in] grouped by expert (`group_sizes [E]` consecutive rows
    per expert), each group with its expert's weight from weights [E, out, in].
    """

    numbers = lax.RaggedDotDimensionNumbers(
        dot_dimension_numbers=(([1], [2]), ([], [])),
        lhs_ragged_dimensions=[0],
        rhs_group_dimensions=[0],
    )
    return lax.ragged_dot_general(
        x,
        weights,
        group_sizes,
        numbers,
        preferred_element_type=jnp.float32,
    ).astype(x.dtype)


def routed_moe(
    x: jax.Array,
    router: jax.Array,
    gate: jax.Array,
    up: jax.Array,
    down: jax.Array,
    *,
    top_k: int,
    normalise: bool,
    float32_weights: bool,
    activation_name: str,
) -> jax.Array:
    """
    A routed mixture-of-experts MLP over tokens x [N, D], with router [E, D], expert
    projections gate, up [E, I_e, D] and down [E, D, I_e], following the eager
    per-expert loop of transformers:

    * router logits in the model dtype, softmax in float32, then the top `top_k`
      experts (ties towards the lower index), whose weights are divided by their sum
      if `normalise`, and kept in float32 (Mixtral) or cast to the model dtype
      (Qwen3-MoE) according to `float32_weights`;
    * each contribution `expert_e(x) * w_e` is computed in float32 and cast (float32
      weights) or in the model dtype;
    * per token, the contributions are summed in ascending expert index (not in rank
      order), rounding to the model dtype after every addition.

    Only the selected experts are computed: the (token, expert) pairs are sorted by
    expert and each expert's rows are multiplied with grouped matmuls.
    """

    dtype = x.dtype
    tokens = x.shape[0]
    experts = router.shape[0]

    probs = jax.nn.softmax(linear(x, router).astype(jnp.float32), axis=-1)
    weights, selected = lax.top_k(probs, top_k)
    if normalise:
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    if not float32_weights:
        weights = weights.astype(dtype)

    # Each token's experts in ascending index, the order of the combination.
    order = jnp.argsort(selected, axis=-1)
    selected = jnp.take_along_axis(selected, order, axis=-1)
    weights = jnp.take_along_axis(weights, order, axis=-1)

    # The (token, expert) pairs grouped by expert.
    pairs = selected.reshape(-1)
    by_expert = jnp.argsort(pairs, stable=True)
    group_sizes = jnp.bincount(pairs, length=experts).astype(jnp.int32)
    rows = x[by_expert // top_k]

    inner = activation(
        activation_name, _grouped_linear(rows, gate, group_sizes)
    ) * _grouped_linear(rows, up, group_sizes)
    outputs = _grouped_linear(inner, down, group_sizes)
    outputs = jnp.zeros_like(outputs).at[by_expert].set(outputs)
    outputs = outputs.reshape(tokens, top_k, -1)

    if float32_weights:
        contributions = (outputs.astype(jnp.float32) * weights[..., None]).astype(dtype)
    else:
        contributions = outputs * weights[..., None]

    # The first addition, to zero, is exact.
    combined = contributions[:, 0]
    for rank in range(1, top_k):
        combined = combined + contributions[:, rank]
    return combined
