# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
NumPy port of the transformers RoPE initialisation functions
(`ROPE_INIT_FUNCTIONS` and the models' `compute_default_rope_parameters`).

Every operation that transformers performs on float32 tensors is performed on float32
arrays here, in the same order, and every operation on Python numbers stays on Python
numbers. The one exception is `base ** exponent`: PyTorch evaluates it with a
vectorised approximation that is accurate to about one ulp and differs between
platforms, so it is evaluated in float64 and rounded, which gives the correctly rounded
float32 value on every host.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from .arch import ArchConfig, RopeSpec


class RopeTables(NamedTuple):
    # [L, rot/2] float32: the inverse frequencies of each layer.
    inv_freq: np.ndarray

    # [L] float32: the factor cos and sin are multiplied by.
    attention_scaling: np.ndarray

    # [L, rot/2] float32: the inverse frequencies beyond the original context length
    # (longrope only, None otherwise).
    long_inv_freq: np.ndarray | None


def build_tables(arch: ArchConfig) -> RopeTables:
    """Builds the per-layer RoPE tables of an architecture."""

    # Layers share few distinct specs (at most one per layer type).
    cache: dict[tuple[RopeSpec, bool], tuple[np.ndarray, float]] = {}

    def compute(spec: RopeSpec, long: bool) -> tuple[np.ndarray, float]:
        if (spec, long) not in cache:
            cache[(spec, long)] = inv_freq_and_scaling(arch, spec, long=long)
        return cache[(spec, long)]

    inv_freq = []
    attention_scaling = []
    long_inv_freq = []

    for spec in arch.rope:
        layer_inv_freq, layer_attention_scaling = compute(spec, long=False)
        inv_freq.append(layer_inv_freq)
        attention_scaling.append(layer_attention_scaling)

        if spec.rope_type == "longrope":
            # The attention factor does not depend on the factor list,
            # so one scaling table serves both.
            long_inv_freq.append(compute(spec, long=True)[0])

    return RopeTables(
        inv_freq=np.stack(inv_freq),
        attention_scaling=np.array(attention_scaling, dtype=np.float32),
        long_inv_freq=np.stack(long_inv_freq) if long_inv_freq else None,
    )


def inv_freq_and_scaling(
    arch: ArchConfig,
    spec: RopeSpec,
    *,
    long: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Returns the inverse frequencies ([rot/2] float32) and the attention scaling of one
    layer, as transformers computes them at initialisation. With `long`, longrope
    returns the frequencies for sequences longer than the original context length.
    """

    parameters = dict(spec.params)
    base = parameters["rope_theta"]
    dim = spec.rot

    if spec.rope_type == "default":
        return 1.0 / _pow(base, _exponents(dim)), 1.0

    if spec.rope_type == "linear":
        inv_freq = 1.0 / _pow(base, _exponents(dim))
        return inv_freq / np.float32(parameters["factor"]), 1.0

    if spec.rope_type == "dynamic":
        # At initialisation, the sequence length is max_position_embeddings,
        # which leaves the base unchanged up to rounding.
        factor = parameters["factor"]
        seq_len = arch.max_position_embeddings
        base = base * (
            (factor * seq_len / arch.max_position_embeddings) - (factor - 1)
        ) ** (dim / (dim - 2))
        return 1.0 / _pow(base, _exponents(dim)), 1.0

    if spec.rope_type == "llama3":
        return _llama3(base, dim, parameters), 1.0

    if spec.rope_type == "yarn":
        return _yarn(arch, base, dim, parameters)

    if spec.rope_type == "longrope":
        return _longrope(arch, base, dim, parameters, long=long)

    raise ValueError(f"Unsupported RoPE type: {spec.rope_type}")


def _exponents(dim: int) -> np.ndarray:
    # torch.arange(0, dim, 2, dtype=torch.int64).float() / dim
    return np.arange(0, dim, 2, dtype=np.int64).astype(np.float32) / np.float32(dim)


def _pow(base: float, exponents: np.ndarray) -> np.ndarray:
    # base ** exponents with a float32 base and float32 exponents (see module docstring).
    return (np.float64(np.float32(base)) ** exponents.astype(np.float64)).astype(
        np.float32
    )


def _rdiv(numerator: float, denominator: np.ndarray) -> np.ndarray:
    # PyTorch evaluates `number / tensor` as `tensor.reciprocal() * number`,
    # which rounds twice. For a numerator of 1 this equals a plain division.
    return (np.float32(1) / denominator) * np.float32(numerator)


def _llama3(base: float, dim: int, parameters: dict) -> np.ndarray:
    inv_freq = 1.0 / _pow(base, _exponents(dim))

    factor = parameters["factor"]
    low_freq_factor = parameters["low_freq_factor"]
    high_freq_factor = parameters["high_freq_factor"]
    old_context_len = parameters["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = _rdiv(2 * math.pi, inv_freq)
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = np.where(
        wavelen > np.float32(low_freq_wavelen),
        inv_freq / np.float32(factor),
        inv_freq,
    )
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (
        _rdiv(old_context_len, wavelen) - np.float32(low_freq_factor)
    ) / np.float32(high_freq_factor - low_freq_factor)
    scaled = (np.float32(1) - smooth_factor) * inv_freq_llama / np.float32(factor)
    smoothed_inv_freq = scaled + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < np.float32(high_freq_wavelen)) & ~(
        wavelen > np.float32(low_freq_wavelen)
    )
    return np.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)


def _yarn(
    arch: ArchConfig,
    base: float,
    dim: int,
    parameters: dict,
) -> tuple[np.ndarray, float]:
    factor = parameters.get("factor")
    attention_factor = parameters.get("attention_factor")
    mscale = parameters.get("mscale")
    mscale_all_dim = parameters.get("mscale_all_dim")
    original_max_position_embeddings = parameters["original_max_position_embeddings"]

    if factor is None:
        factor = arch.max_position_embeddings / original_max_position_embeddings

    def get_mscale(scale: float, mscale: float = 1) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(
                get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
            )
        else:
            attention_factor = get_mscale(factor)

    beta_fast = parameters.get("beta_fast") or 32
    beta_slow = parameters.get("beta_slow") or 1

    def find_correction_dim(num_rotations: float) -> float:
        return (
            dim
            * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(base))

    # Transformers reads `truncate` from the top-level RoPE parameters, which for
    # per-layer-type parameters are keyed by layer type, so it is always true there.
    truncate = True if arch.rope_per_layer_type else parameters.get("truncate", True)

    low = find_correction_dim(beta_fast)
    high = find_correction_dim(beta_slow)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)

    if low == high:
        high += 0.001  # Prevent singularity

    ramp = np.clip(
        (np.arange(dim // 2, dtype=np.float32) - np.float32(low))
        / np.float32(high - low),
        0,
        1,
    )

    pos_freqs = _pow(base, _exponents(dim))
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (np.float32(factor) * pos_freqs)

    inv_freq_extrapolation_factor = np.float32(1) - ramp
    inv_freq = (
        inv_freq_interpolation * (np.float32(1) - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return inv_freq, attention_factor


def _longrope(
    arch: ArchConfig,
    base: float,
    dim: int,
    parameters: dict,
    *,
    long: bool,
) -> tuple[np.ndarray, float]:
    factor = parameters.get("factor")
    attention_factor = parameters.get("attention_factor")
    original_max_position_embeddings = parameters["original_max_position_embeddings"]

    # Phi-3 derives the factor from the extended and the original context lengths.
    if factor is None:
        factor = arch.max_position_embeddings / original_max_position_embeddings

    if attention_factor is None:
        if factor <= 1.0:
            attention_factor = 1.0
        else:
            attention_factor = math.sqrt(
                1 + math.log(factor) / math.log(original_max_position_embeddings)
            )

    ext_factors = np.array(
        parameters["long_factor" if long else "short_factor"],
        dtype=np.float32,
    )
    return 1.0 / (ext_factors * _pow(base, _exponents(dim))), attention_factor
