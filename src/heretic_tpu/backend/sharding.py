# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
Parallelism: the choice between a single device and tensor parallelism over all
local devices, and the partition specs of the parameters, adapters and KV cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec, SingleDeviceSharding

from .arch import ArchConfig
from .weights import (
    abliterable_components,
    component_shape,
    device_footprint,
    param_layout,
)

# The mesh axis tensor parallelism shards over.
MODEL_AXIS = "model"

# Largest default LoRA rank (ARA's), whose adapters the "auto" choice leaves room for.
DEFAULT_MAX_LORA_RANK = 50

# Fraction of device memory kept free by the "auto" choice.
MEMORY_RESERVE = 0.05

PARALLELISMS = ("auto", "single", "tensor")


@dataclass(frozen=True)
class ShardingPlan:
    """
    How parameters, adapters and the KV cache are placed. Plans compare by value,
    so an engine (whose executables are compiled for given shardings) can be reused
    exactly when the plan is equal.
    """

    # "single" or "tensor".
    kind: str

    device_ids: tuple[int, ...]

    # () for "single", (number of devices,) for "tensor".
    mesh_shape: tuple[int, ...]

    # Partition spec of every parameter path (see weights.param_layout), sorted.
    specs: tuple[tuple[str, PartitionSpec], ...]

    # Whether the attention heads (q/k/v rows, the attn.o_proj input axis and the
    # KV cache head axis) are sharded.
    attention_sharded: bool

    # Whether the MLP intermediate axis (or the expert intermediate axis) is sharded.
    mlp_sharded: bool

    @cached_property
    def devices(self) -> list[jax.Device]:
        devices = {device.id: device for device in jax.local_devices()}
        return [devices[device_id] for device_id in self.device_ids]

    @cached_property
    def mesh(self) -> Mesh | None:
        if self.kind == "single":
            return None
        return Mesh(np.array(self.devices), (MODEL_AXIS,))

    @property
    def model_axis_size(self) -> int:
        return math.prod(self.mesh_shape)

    def sharding(self, spec: PartitionSpec | None = None) -> jax.sharding.Sharding:
        """The sharding of an array with the given spec (replicated by default)."""

        if self.mesh is None:
            return SingleDeviceSharding(self.devices[0])
        return NamedSharding(self.mesh, spec if spec is not None else PartitionSpec())

    @property
    def replicated(self) -> jax.sharding.Sharding:
        return self.sharding()

    @cached_property
    def _specs_by_path(self) -> dict[str, PartitionSpec]:
        return dict(self.specs)

    def param_spec(self, path: str) -> PartitionSpec:
        return self._specs_by_path[path]

    def param_sharding(self, path: str) -> jax.sharding.Sharding:
        return self.sharding(self.param_spec(path))

    def is_sharded(self, path: str) -> bool:
        return self.kind == "tensor" and MODEL_AXIS in self.param_spec(path)

    def lora_shardings(
        self,
        component: str,
    ) -> tuple[jax.sharding.Sharding, jax.sharding.Sharding]:
        """
        Shardings of a component's adapters A [L, M, r, d_in] and B [L, M, d_out, r]:
        A is sharded like the module's input axis, B is replicated.
        """

        sharded = (
            self.attention_sharded if component == "attn.o_proj" else self.mlp_sharded
        )
        a_spec = PartitionSpec(None, None, None, MODEL_AXIS) if sharded else None
        return self.sharding(a_spec), self.replicated

    @property
    def kv_cache_sharding(self) -> jax.sharding.Sharding:
        """Sharding of a KV cache array [L, B, T_cache, KV, hd]."""

        if self.attention_sharded:
            return self.sharding(PartitionSpec(None, None, None, MODEL_AXIS, None))
        return self.replicated


def choose_plan(arch: ArchConfig, dtype: Any, parallelism: str) -> ShardingPlan:
    """
    Chooses the plan for the `parallelism` setting:

    * "single": everything on the first local device, without a mesh.
    * "tensor": a one-dimensional mesh over all local devices.
    * "auto": "single" if there is one local device, if devices report no memory
      statistics (CPU), or if the parameters plus a reserve fit into one device;
      otherwise "tensor".
    """

    if parallelism not in PARALLELISMS:
        raise ValueError(f"Unknown parallelism: {parallelism}")

    devices = jax.local_devices()

    single = ShardingPlan(
        kind="single",
        device_ids=(devices[0].id,),
        mesh_shape=(),
        specs=tuple((path, PartitionSpec()) for path in sorted(param_layout(arch))),
        attention_sharded=False,
        mlp_sharded=False,
    )

    if parallelism == "auto":
        parallelism = "single" if _fits_one_device(arch, dtype, single) else "tensor"

    if parallelism == "single":
        return single

    size = len(devices)
    attention_sharded = (
        arch.num_attention_heads % size == 0 and arch.num_key_value_heads % size == 0
    )
    mlp_size = arch.moe_intermediate_size if arch.is_moe else arch.intermediate_size
    mlp_sharded = mlp_size % size == 0
    vocab_sharded = arch.vocab_size % size == 0

    return ShardingPlan(
        kind="tensor",
        device_ids=tuple(device.id for device in devices),
        mesh_shape=(size,),
        specs=tuple(
            sorted(
                partition_specs(
                    arch,
                    attention_sharded=attention_sharded,
                    mlp_sharded=mlp_sharded,
                    vocab_sharded=vocab_sharded,
                ).items()
            )
        ),
        attention_sharded=attention_sharded,
        mlp_sharded=mlp_sharded,
    )


def partition_specs(
    arch: ArchConfig,
    *,
    attention_sharded: bool,
    mlp_sharded: bool,
    vocab_sharded: bool,
) -> dict[str, PartitionSpec]:
    """Partition specs of the parameters under tensor parallelism."""

    rows = PartitionSpec(None, MODEL_AXIS, None)
    bias_rows = PartitionSpec(None, MODEL_AXIS)
    expert_rows = PartitionSpec(None, None, MODEL_AXIS, None)
    module_inputs = PartitionSpec(None, None, None, MODEL_AXIS)

    sharded_specs = {}
    if vocab_sharded:
        sharded_specs["embed"] = PartitionSpec(MODEL_AXIS, None)
        sharded_specs["head"] = PartitionSpec(MODEL_AXIS, None)
    if attention_sharded:
        for name in ("q", "k", "v"):
            sharded_specs[f"layers/{name}"] = rows
            sharded_specs[f"layers/{name}_bias"] = bias_rows
        sharded_specs["layers/o_proj"] = module_inputs
    if mlp_sharded:
        for name in ("gate", "up"):
            sharded_specs[f"layers/{name}"] = rows
            sharded_specs[f"layers/{name}_bias"] = bias_rows
        sharded_specs["layers/down_proj"] = module_inputs
        sharded_specs["layers/expert_gate"] = expert_rows
        sharded_specs["layers/expert_up"] = expert_rows
        sharded_specs["layers/expert_down"] = module_inputs

    # Everything else (norms, biases of outputs, the router and the buffers)
    # is replicated.
    return {
        path: sharded_specs.get(path, PartitionSpec()) for path in param_layout(arch)
    }


def _fits_one_device(arch: ArchConfig, dtype: Any, single: ShardingPlan) -> bool:
    if len(jax.local_devices()) == 1:
        return True

    stats = single.devices[0].memory_stats()
    if stats is None:
        return True

    # Float32 adapters of every component at the largest default rank.
    adapter_bytes = 0
    for component in abliterable_components(arch):
        d_out, d_in = component_shape(arch, component)
        adapter_bytes += (
            arch.num_hidden_layers * DEFAULT_MAX_LORA_RANK * (d_in + d_out) * 4
        )

    footprint = device_footprint(arch, dtype, single)
    bytes_limit = stats["bytes_limit"]
    return footprint + adapter_bytes + MEMORY_RESERVE * bytes_limit <= bytes_limit
