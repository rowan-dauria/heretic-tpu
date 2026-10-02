# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
The decoder forward pass: one `lax.scan` over the stacked layers of the parameter
pytree (see weights.py), with optional float32 LoRA adapters, captures at one column,
and the KV-cache primitives of generation.

Every function here is a pure traceable function of explicit array arguments. The
architecture and every shape-determining option are keyword arguments that the
caller binds statically (for example with `functools.partial` before `jax.jit`).
Passing `lora=None` traces a program without adapters, which is a separate
executable from the adapted one.

Inputs are left-padded batches: `tokens int32 [B, T]` and `mask bool [B, T]`
(true for real tokens). Position ids are `cumsum(mask) - 1`, with padding positions
set to 0, as in transformers' generate.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental.layout import Layout, with_layout_constraint

from . import layers
from .arch import ArchConfig
from .weights import Params, abliterable_components

# Component -> (A [L, M, r, d_in], B [L, M, d_out, r]), both float32.
Lora = dict[str, tuple[jax.Array, jax.Array]]

# (keys, values), each [L, B, KV, T_cache, hd] in the model dtype. The slot axis is
# next to the head dimension, as decode attention reads the cache on TPUs (with the
# slots outside the key/value heads, XLA would copy the whole cache into this layout
# and back on every call that carries it).
KVCache = tuple[jax.Array, jax.Array]

# Component -> (inputs [B, L, M, d_in], outputs [B, L, M, d_out]), model dtype.
ModuleIO = dict[str, tuple[jax.Array, jax.Array]]

CAPTURES = frozenset({"logits", "hidden", "module_io"})


class Captures(NamedTuple):
    """Quantities captured at one column (by default the last prompt position)."""

    # [B, V] float32, after final soft-capping.
    logits: jax.Array | None

    # [B, L + 1, D] in the model dtype, following the transformers hidden_states
    # convention: index 0 is the (scaled) embedding output, index i for 1 <= i < L
    # the output of layer i - 1, and index L the final-norm-applied output of the
    # last layer.
    hidden: jax.Array | None

    module_io: ModuleIO | None


class DecoderOutputs(NamedTuple):
    # [B, T, D]: the output of the last layer (before the final norm) at every column.
    x: jax.Array

    # The KV cache, if requested.
    cache: KVCache | None

    # Captured at one column, if requested (see `Captures`).
    hidden: jax.Array | None
    module_io: ModuleIO | None


class _LayerOutputs(NamedTuple):
    # [B, T, D]
    x: jax.Array

    # [B, T, KV, hd] after RoPE (keys) and as projected (values), as transformers
    # caches them.
    k: jax.Array
    v: jax.Array

    # Component -> (input [B, T, d_in], output [B, T, d_out]) of the abliterable
    # modules, the output including bias and LoRA delta.
    module_io: dict[str, tuple[jax.Array, jax.Array]]


# Computes attention from q [B, T, H, hd], k, v [B, T, KV, hd] and the layer's window,
# returning [B, T, H·hd].
_Attend = Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]


def positions_from_mask(mask: jax.Array) -> jax.Array:
    """Position ids [B, T] of a left-padded batch: `cumsum(mask) - 1`, padding 0."""

    positions = jnp.cumsum(mask.astype(jnp.int32), axis=-1) - 1
    return jnp.where(mask, positions, 0)


def embed(params: Params, tokens: jax.Array, *, arch: ArchConfig) -> jax.Array:
    """The embedding output [..., D] in the model dtype."""

    x = params["embed"][tokens]
    if arch.has_sandwich_norms:
        # Gemma 3 scales by sqrt(hidden_size), rounded to float32 and then to the
        # model dtype, as transformers' embed_scale buffer is.
        x = x * jnp.asarray(np.float32(arch.hidden_size**0.5), dtype=x.dtype)
    return x


def final_norm(params: Params, x: jax.Array, *, arch: ArchConfig) -> jax.Array:
    """The final norm applied to hidden states [..., D]."""

    return layers.rms_norm(
        x,
        params["final_norm"],
        arch.rms_norm_eps,
        gemma=arch.has_sandwich_norms,
    )


def unembed(params: Params, x: jax.Array, *, arch: ArchConfig) -> jax.Array:
    """
    Logits [..., V] float32 of last-layer outputs x [..., D]: final norm, then the head
    (or the tied embedding matrix) in the model dtype, then final soft-capping.
    """

    head = params["embed"] if arch.tied_head else params["head"]
    logits = layers.linear(final_norm(params, x, arch=arch), head)
    if arch.final_logit_softcapping is not None:
        logits = layers.softcap(logits, arch.final_logit_softcapping)
    return logits.astype(jnp.float32)


def _rope(
    arch: ArchConfig,
    buffers: dict[str, jax.Array],
    positions: jax.Array,
    long_factor: jax.Array | None,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
    """cos and sin [B, T, rot] of one layer."""

    batch = positions.shape[0]
    inv_freq = jnp.broadcast_to(buffers["inv_freq"], (batch, arch.rot // 2))
    if arch.rope_switch == "long_factor" and long_factor is not None:
        # Each row uses the long factors from its switch step on (traced per row).
        inv_freq = jnp.where(
            long_factor[:, None],
            buffers["long_inv_freq"][None, :],
            inv_freq,
        )
    return layers.rope_cos_sin(
        positions,
        inv_freq,
        buffers["attention_scaling"],
        dtype,
    )


def _llama_4_query_scale(
    arch: ArchConfig,
    positions: jax.Array,
    dtype: jnp.dtype,
) -> jax.Array:
    """
    The factor [B, T] Ministral 3 multiplies queries by after RoPE,
    `1 + beta * log(1 + floor(position / original_max_position_embeddings))`,
    computed in float32 from the position ids and cast to the model dtype.
    """

    spec = arch.rope[0]
    beta = spec.get("llama_4_scaling_beta")
    original = spec.get("original_max_position_embeddings")
    scale = 1 + beta * jnp.log(1 + jnp.floor(positions.astype(jnp.float32) / original))
    return scale.astype(dtype)


def _layer(
    arch: ArchConfig,
    x: jax.Array,
    layer: dict[str, jax.Array],
    buffers: dict[str, jax.Array],
    adapters: Lora,
    positions: jax.Array,
    long_factor: jax.Array | None,
    attend: _Attend,
) -> _LayerOutputs:
    """One decoder layer applied to x [B, T, D]."""

    batch, length, _ = x.shape
    eps = arch.rms_norm_eps
    gemma = arch.has_sandwich_norms
    head_dim = arch.head_dim

    def bias(name: str) -> jax.Array | None:
        return layer.get(f"{name}_bias")

    # Self-attention.
    residual = x
    h = layers.rms_norm(x, layer["attn_norm"], eps, gemma=gemma)

    q = layers.linear(h, layer["q"], bias("q"))
    k = layers.linear(h, layer["k"], bias("k"))
    v = layers.linear(h, layer["v"], bias("v"))
    q = q.reshape(batch, length, arch.num_attention_heads, head_dim)
    k = k.reshape(batch, length, arch.num_key_value_heads, head_dim)
    v = v.reshape(batch, length, arch.num_key_value_heads, head_dim)

    if arch.has_qk_norm:
        # The projections are materialised in the model dtype first. Otherwise
        # XLA:TPU keeps their float32 accumulators for the norms and moves them into
        # the layout of attention in float32 (a tenth of the prefill time of
        # Qwen3-4B).
        q, k = lax.optimization_barrier((q, k))
        q = layers.rms_norm(q, layer["q_norm"], eps, gemma=gemma)
        k = layers.rms_norm(k, layer["k_norm"], eps, gemma=gemma)

    cos, sin = _rope(arch, buffers, positions, long_factor, x.dtype)
    q = layers.apply_rope(q, cos, sin)
    k = layers.apply_rope(k, cos, sin)
    if arch.model_type == "ministral3":
        q = q * _llama_4_query_scale(arch, positions, x.dtype)[:, :, None, None]

    attended = attend(q, k, v, buffers["window"])

    # The abliterable modules are stored with a module axis (M = 1).
    o_proj_bias = bias("o_proj")
    o_out = layers.adapted_linear(
        attended,
        layer["o_proj"][0],
        None if o_proj_bias is None else o_proj_bias[0],
        _module_adapter(adapters, "attn.o_proj"),
    )
    module_io = {"attn.o_proj": (attended, o_out)}

    h = o_out
    if gemma:
        h = layers.rms_norm(h, layer["attn_out_norm"], eps, gemma=True)
    x = residual + h

    # MLP.
    residual = x
    h = layers.rms_norm(x, layer["mlp_norm"], eps, gemma=gemma)

    if arch.is_moe:
        # Mixtral always normalises the routing weights and keeps them in float32.
        mixtral = arch.model_type == "mixtral"
        h = layers.routed_moe(
            h.reshape(batch * length, -1),
            layer["router"],
            layer["expert_gate"],
            layer["expert_up"],
            layer["expert_down"],
            top_k=arch.num_experts_per_tok,
            normalise=mixtral or bool(arch.norm_topk_prob),
            float32_weights=mixtral,
            activation_name=arch.activation,
        )
        x = residual + h.reshape(batch, length, -1)
        return _LayerOutputs(x=x, k=k, v=v, module_io=module_io)

    inner = layers.gated_mlp(
        h,
        layer["gate"],
        layer["up"],
        bias("gate"),
        bias("up"),
        arch.activation,
    )
    down_proj_bias = bias("down_proj")
    down_out = layers.adapted_linear(
        inner,
        layer["down_proj"][0],
        None if down_proj_bias is None else down_proj_bias[0],
        _module_adapter(adapters, "mlp.down_proj"),
    )
    module_io["mlp.down_proj"] = (inner, down_out)

    h = down_out
    if gemma:
        h = layers.rms_norm(h, layer["mlp_out_norm"], eps, gemma=True)
    x = residual + h

    return _LayerOutputs(x=x, k=k, v=v, module_io=module_io)


def _module_adapter(
    adapters: Lora,
    component: str,
) -> tuple[jax.Array, jax.Array] | None:
    """One layer's adapter (A [r, d_in], B [d_out, r]) of a component, if any."""

    if component not in adapters:
        return None
    a, b = adapters[component]
    return a[0], b[0]


def _layer_inputs(
    arch: ArchConfig,
    params: Params,
    lora: Lora | None,
) -> tuple[Any, ...]:
    """The per-layer inputs the scan slices along the leading layer axis."""

    adapters = {} if lora is None else {c: tuple(lora[c]) for c in sorted(lora)}
    unknown = set(adapters) - set(abliterable_components(arch))
    if unknown:
        raise ValueError(f"Unknown components: {sorted(unknown)}")
    return params["layers"], params["buffers"], adapters


def _column(x: jax.Array, column: jax.Array | int) -> jax.Array:
    """Selects column `column` of x [B, T, ...] (a traced or static index)."""

    return lax.dynamic_index_in_dim(x, column, axis=1, keepdims=False)


def decoder(
    params: Params,
    lora: Lora | None,
    tokens: jax.Array,
    mask: jax.Array,
    long_factor: jax.Array | None = None,
    *,
    arch: ArchConfig,
    cache_len: int | None = None,
    cache_layout: tuple[int, ...] | None = None,
    capture: frozenset[str] = frozenset(),
    column: jax.Array | int | None = None,
) -> DecoderOutputs:
    """
    Runs the decoder over a left-padded batch, tokens and mask [B, T].

    `long_factor` (bool [B]) selects the long RoPE factors per row (`longrope` only;
    None means short factors for every row). With `cache_len` (at least T), the
    output includes the KV cache [L, B, KV, cache_len, hd], built in its final shape:
    each layer returns its keys and values padded to `cache_len` as scan outputs, so
    the stacked outputs are the cache. `cache_layout` (major to minor), if given, is
    the layout of each layer's slice [B, KV, cache_len, hd] of the cache, which XLA
    then stacks in the layout the cache is needed in instead of copying it at the
    end. `capture` (a subset of {"hidden", "module_io"}) selects quantities captured
    at column `column` (default T - 1).
    """

    unknown = capture - {"hidden", "module_io"}
    if unknown:
        raise ValueError(f"Unknown captures: {sorted(unknown)}")

    length = tokens.shape[1]
    if cache_len is not None and cache_len < length:
        raise ValueError(f"cache_len {cache_len} is shorter than the input ({length}).")
    if column is None:
        column = length - 1

    embedded = embed(params, tokens, arch=arch)
    positions = positions_from_mask(mask)

    def attend(q, k, v, window):
        return layers.attention(
            q,
            k,
            v,
            mask,
            window,
            scale=arch.attention_scale,
            cap=arch.attn_logit_softcapping,
        )

    def body(x, inputs):
        layer, buffers, adapters = inputs
        out = _layer(arch, x, layer, buffers, adapters, positions, long_factor, attend)

        ys = {}
        if cache_len is not None:
            # [B, T, KV, hd] -> [B, KV, cache_len, hd].
            padding = ((0, 0), (0, 0), (0, cache_len - length), (0, 0))
            for name, value in (("k", out.k), ("v", out.v)):
                value = jnp.pad(jnp.swapaxes(value, 1, 2), padding)
                if cache_layout is not None:
                    value = with_layout_constraint(
                        value, Layout(major_to_minor=cache_layout)
                    )
                ys[name] = value
        if "hidden" in capture:
            ys["hidden"] = _column(out.x, column)
        if "module_io" in capture:
            ys["module_io"] = {
                component: (_column(module_in, column), _column(module_out, column))
                for component, (module_in, module_out) in out.module_io.items()
            }
        return out.x, ys

    x, ys = lax.scan(body, embedded, _layer_inputs(arch, params, lora))

    cache = None
    if cache_len is not None:
        cache = (ys["k"], ys["v"])

    hidden = None
    if "hidden" in capture:
        # ys["hidden"] is [L, B, D]: the output of every layer.
        layer_outputs = ys["hidden"]
        hidden = jnp.concatenate(
            [
                _column(embedded, column)[None],
                layer_outputs[:-1],
                final_norm(params, layer_outputs[-1], arch=arch)[None],
            ]
        )
        hidden = jnp.swapaxes(hidden, 0, 1)

    module_io = None
    if "module_io" in capture:
        # [L, B, d] -> [B, L, M, d] with M = 1.
        module_io = {
            component: tuple(
                jnp.swapaxes(array, 0, 1)[:, :, None, :] for array in arrays
            )
            for component, arrays in ys["module_io"].items()
        }

    return DecoderOutputs(x=x, cache=cache, hidden=hidden, module_io=module_io)


def prefill(
    params: Params,
    lora: Lora | None,
    tokens: jax.Array,
    mask: jax.Array,
    long_factor: jax.Array | None = None,
    *,
    arch: ArchConfig,
    want: frozenset[str] = frozenset({"logits"}),
    cache_len: int | None = None,
    cache_layout: tuple[int, ...] | None = None,
    column: jax.Array | int | None = None,
) -> tuple[Captures, KVCache | None]:
    """
    Runs the decoder over a left-padded batch and returns the quantities in `want`
    (a subset of `CAPTURES`) at column `column` (default T - 1, the last prompt
    position) and, with `cache_len`, the KV cache (see `decoder`, also for
    `cache_layout`).

    The re-prefill of long-RoPE generation is a prefill over all `cache_len` slots
    (unwritten slots masked) with `column` the newest slot.
    """

    unknown = want - CAPTURES
    if unknown:
        raise ValueError(f"Unknown captures: {sorted(unknown)}")
    if column is None:
        column = tokens.shape[1] - 1

    out = decoder(
        params,
        lora,
        tokens,
        mask,
        long_factor,
        arch=arch,
        cache_len=cache_len,
        cache_layout=cache_layout,
        capture=want - {"logits"},
        column=column,
    )

    logits = None
    if "logits" in want:
        logits = unembed(params, _column(out.x, column), arch=arch)

    captures = Captures(logits=logits, hidden=out.hidden, module_io=out.module_io)
    return captures, out.cache


def decode_step(
    params: Params,
    lora: Lora | None,
    cache: KVCache,
    tokens: jax.Array,
    positions: jax.Array,
    slot: jax.Array,
    kv_mask: jax.Array,
    long_factor: jax.Array | None = None,
    *,
    arch: ArchConfig,
) -> tuple[jax.Array, KVCache]:
    """
    Consumes one token per row, tokens and positions int32 [B], at cache slot `slot`
    (an int32 scalar), and returns its logits [B, V] float32 and the updated cache.

    `kv_mask` (bool [B, T_cache]) is true for the slots that hold real tokens; slots
    at or after `slot` are treated as unwritten whatever it says. `long_factor` is as
    for `decoder`, evaluated for this step. Each layer reads its cache slice as a
    scan input and returns only the new keys and values [B, KV, hd]; one
    `dynamic_update_slice` then writes [L, B, KV, 1, hd] at `slot`, which updates a
    donated cache in place.
    """

    k_cache, v_cache = cache
    x = embed(params, tokens[:, None], arch=arch)

    def body(x, inputs):
        layer, buffers, adapters, layer_k_cache, layer_v_cache = inputs

        def attend(q, k, v, window):
            out = layers.decode_attention(
                q[:, 0],
                layer_k_cache,
                layer_v_cache,
                k[:, 0],
                v[:, 0],
                kv_mask,
                slot,
                window,
                scale=arch.attention_scale,
                cap=arch.attn_logit_softcapping,
            )
            return out[:, None]

        out = _layer(
            arch,
            x,
            layer,
            buffers,
            adapters,
            positions[:, None],
            long_factor,
            attend,
        )
        return out.x, (out.k[:, 0], out.v[:, 0])

    x, (k_new, v_new) = lax.scan(
        body,
        x,
        (*_layer_inputs(arch, params, lora), k_cache, v_cache),
    )

    start = (0, 0, 0, slot, 0)
    k_cache = lax.dynamic_update_slice(
        k_cache, k_new[:, :, :, None].astype(k_cache.dtype), start
    )
    v_cache = lax.dynamic_update_slice(
        v_cache, v_new[:, :, :, None].astype(v_cache.dtype), start
    )

    return unembed(params, x[:, 0], arch=arch), (k_cache, v_cache)


def logits_all(
    params: Params,
    lora: Lora | None,
    tokens: jax.Array,
    mask: jax.Array,
    long_factor: jax.Array | None = None,
    *,
    arch: ArchConfig,
) -> jax.Array:
    """Logits [B, T, V] float32 at every position. Only for tests."""

    out = decoder(params, lora, tokens, mask, long_factor, arch=arch)
    return unembed(params, out.x, arch=arch)
