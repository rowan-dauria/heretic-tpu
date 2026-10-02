# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Sharding plans, and loading onto a forced 4-device CPU mesh."""

import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P
from jax.sharding import SingleDeviceSharding

from heretic_tpu.backend import arch as arch_module
from heretic_tpu.backend.sharding import MODEL_AXIS, choose_plan, partition_specs
from heretic_tpu.backend.weights import (
    abliterable_components,
    component_shape,
    param_layout,
)
from tests.backend.tiny import REAL_MODELS, real_arch

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_single_plan() -> None:
    arch = real_arch("Qwen/Qwen3-4B-Instruct-2507")
    plan = choose_plan(arch, ml_dtypes.bfloat16, "single")

    assert plan.kind == "single"
    assert plan.device_ids == (jax.local_devices()[0].id,)
    assert plan.mesh_shape == ()
    assert plan.mesh is None
    assert plan.model_axis_size == 1
    assert plan.devices == [jax.local_devices()[0]]
    assert dict(plan.specs).keys() == param_layout(arch).keys()
    assert set(dict(plan.specs).values()) == {P()}
    assert isinstance(plan.param_sharding("layers/q"), SingleDeviceSharding)
    assert isinstance(plan.kv_cache_sharding, SingleDeviceSharding)
    assert not any(plan.is_sharded(path) for path in param_layout(arch))

    # Plans compare and hash by value.
    again = choose_plan(arch, ml_dtypes.bfloat16, "single")
    assert again == plan
    assert hash(again) == hash(plan)
    assert choose_plan(real_arch("Qwen/Qwen3-0.6B"), np.float32, "single") != plan


@pytest.mark.skipif(
    jax.default_backend() != "cpu",
    reason="tests the plan on CPU devices",
)
def test_auto_is_single_on_cpu() -> None:
    arch = real_arch("Qwen/Qwen3-4B-Instruct-2507")
    assert jax.local_devices()[0].memory_stats() is None
    assert choose_plan(arch, np.float32, "auto").kind == "single"

    with pytest.raises(ValueError):
        choose_plan(arch, np.float32, "data")


class _FakeDevice:
    def __init__(self, device_id: int, bytes_limit: int):
        self.id = device_id
        self.bytes_limit = bytes_limit

    def memory_stats(self) -> dict[str, int]:
        return {"bytes_limit": self.bytes_limit, "bytes_in_use": 0}


def test_auto_choice_with_memory_statistics(monkeypatch) -> None:
    arch = real_arch("Qwen/Qwen3-4B-Instruct-2507")
    dtype = np.dtype(ml_dtypes.bfloat16)

    footprint = sum(
        math.prod(leaf.shape) * (leaf.dtype or dtype).itemsize
        for leaf in param_layout(arch).values()
    )
    adapters = sum(
        arch.num_hidden_layers * 50 * sum(component_shape(arch, component)) * 4
        for component in abliterable_components(arch)
    )
    # A limit just above the smallest one for which parameters,
    # rank-50 adapters and a 5% reserve fit.
    limit = math.ceil((footprint + adapters) / 0.95) + 1

    for bytes_limit, kind in [(limit, "single"), (limit - 64, "tensor")]:
        devices = [_FakeDevice(i, bytes_limit) for i in range(2)]
        monkeypatch.setattr(jax, "local_devices", lambda devices=devices: devices)
        plan = choose_plan(arch, dtype, "auto")
        assert plan.kind == kind
        assert plan.device_ids == ((0,) if kind == "single" else (0, 1))

    # A single device is always used alone.
    devices = [_FakeDevice(0, 1)]
    monkeypatch.setattr(jax, "local_devices", lambda: devices)
    assert choose_plan(arch, dtype, "auto").kind == "single"


def test_partition_specs() -> None:
    arch = real_arch("Qwen/Qwen2.5-0.5B-Instruct")

    specs = partition_specs(
        arch, attention_sharded=True, mlp_sharded=True, vocab_sharded=True
    )
    assert specs.keys() == param_layout(arch).keys()
    assert specs["embed"] == P(MODEL_AXIS, None)
    assert "head" not in specs  # Tied.
    for name in ("q", "k", "v", "gate", "up"):
        assert specs[f"layers/{name}"] == P(None, MODEL_AXIS, None)
    for name in ("q_bias", "k_bias", "v_bias"):
        assert specs[f"layers/{name}"] == P(None, MODEL_AXIS)
    assert specs["layers/o_proj"] == P(None, None, None, MODEL_AXIS)
    assert specs["layers/down_proj"] == P(None, None, None, MODEL_AXIS)
    for path in ("final_norm", "layers/attn_norm", "layers/mlp_norm"):
        assert specs[path] == P()
    for path in specs:
        if path.startswith("buffers/"):
            assert specs[path] == P()

    specs = partition_specs(
        arch, attention_sharded=False, mlp_sharded=False, vocab_sharded=False
    )
    assert set(specs.values()) == {P()}


def test_moe_partition_specs(monkeypatch) -> None:
    monkeypatch.setitem(
        arch_module.SUPPORTED_ARCHITECTURES, "qwen3_moe", ("qwen3_moe",)
    )
    arch = real_arch("Qwen/Qwen3-30B-A3B")

    specs = partition_specs(
        arch, attention_sharded=True, mlp_sharded=True, vocab_sharded=True
    )
    assert specs["layers/expert_gate"] == P(None, None, MODEL_AXIS, None)
    assert specs["layers/expert_up"] == P(None, None, MODEL_AXIS, None)
    assert specs["layers/expert_down"] == P(None, None, None, MODEL_AXIS)
    assert specs["layers/router"] == P()
    assert "layers/gate" not in specs and "layers/down_proj" not in specs


# Loads checkpoints onto a forced 4-device CPU mesh and checks every shard against
# a single-device load. Prints a JSON summary per checkpoint.
_MESH_SCRIPT = textwrap.dedent(
    """
    import gc
    import json
    import sys

    import jax
    import numpy as np

    from heretic_tpu.backend import weights
    from heretic_tpu.backend.arch import SUPPORTED_ARCHITECTURES, ArchConfig
    from heretic_tpu.backend.sharding import MODEL_AXIS, choose_plan

    assert len(jax.local_devices()) == 4

    # Accept the planned Qwen3-MoE architecture.
    SUPPORTED_ARCHITECTURES["qwen3_moe"] = ("qwen3_moe",)

    placed_shapes = []
    device_put = jax.device_put

    def recording_device_put(x, *args, **kwargs):
        placed_shapes.append(list(np.shape(x)))
        return device_put(x, *args, **kwargs)

    for directory in sys.argv[1:]:
        ckpt = weights.resolve_checkpoint(directory, None)
        tensors = weights.build_tensor_index(ckpt)
        arch = ArchConfig.from_hf(ckpt.config, ckpt.raw_config, tensors)
        plan = choose_plan(arch, np.float32, "tensor")
        assert plan == choose_plan(arch, np.float32, "tensor")
        assert plan.mesh.shape == {MODEL_AXIS: 4}

        single = choose_plan(arch, np.float32, "single")
        reference = weights.load_params(ckpt, tensors, arch, np.float32, single)

        placed_shapes.clear()
        live_before = len(jax.live_arrays())
        jax.device_put = recording_device_put
        params = weights.load_params(ckpt, tensors, arch, np.float32, plan)
        jax.device_put = device_put
        # Counted before shard.data below creates per-device views.
        new_live_arrays = len(jax.live_arrays()) - live_before
        layout = weights.param_layout(arch)

        sharded = []
        for path in layout:
            *parents, name = path.split("/")
            array, expected = params, reference
            for key in [*parents, name]:
                array, expected = array[key], expected[key]
            expected = np.asarray(expected)

            assert array.sharding == plan.param_sharding(path), path
            assert array.sharding.spec == plan.param_spec(path), path
            assert array.shape == expected.shape, path
            assert len(array.addressable_shards) == 4, path
            for shard in array.addressable_shards:
                np.testing.assert_array_equal(np.asarray(shard.data), expected[shard.index])
            if plan.is_sharded(path):
                axis = list(plan.param_spec(path)).index(MODEL_AXIS)
                assert array.addressable_shards[0].data.shape[axis] * 4 == array.shape[axis]
                sharded.append(path)

        on_first_device = sum(
            shard.data.nbytes
            for array in jax.tree.leaves(params)
            for shard in array.addressable_shards
            if shard.device == plan.devices[0]
        )

        a_sharding, b_sharding = plan.lora_shardings("attn.o_proj")
        print(json.dumps({
            "directory": directory,
            "attention_sharded": plan.attention_sharded,
            "mlp_sharded": plan.mlp_sharded,
            "sharded": sorted(sharded),
            "placed_shapes": sorted(placed_shapes),
            "layout_shapes": sorted(list(leaf.shape) for leaf in layout.values()),
            "new_live_arrays": new_live_arrays,
            "leaves": len(layout),
            "footprint": weights.device_footprint(arch, np.float32, plan),
            "on_first_device": on_first_device,
            "lora_a_spec": list(a_sharding.spec),
            "lora_b_spec": list(b_sharding.spec),
            "kv_cache_spec": list(plan.kv_cache_sharding.spec),
        }))

        # Release this checkpoint's arrays before counting for the next one.
        del params, reference, array, expected, shard
        gc.collect()
    """
)


def test_tensor_parallel_loading_on_four_cpu_devices(tmp_path) -> None:
    pytest.importorskip("torch")
    from tests.backend.tiny import build_checkpoint, tiny_config

    checkpoints = {
        # 8 query and 4 key/value heads: attention is sharded, with biases.
        "attention": build_checkpoint(
            tiny_config(
                REAL_MODELS["llama"],
                num_attention_heads=8,
                num_key_value_heads=4,
                attention_bias=True,
                mlp_bias=True,
            ),
            tmp_path / "attention",
            tie_word_embeddings=False,
        ),
        # 2 key/value heads: attention is replicated, the MLP and vocabulary are not.
        "gemma3": build_checkpoint(
            tiny_config(REAL_MODELS["gemma3"]), tmp_path / "gemma3"
        ),
        # 2 key/value heads, and experts of width 32.
        "moe": build_checkpoint(
            tiny_config(REAL_MODELS["qwen3_moe"]), tmp_path / "moe"
        ),
        # Nothing divides by 4 but the heads.
        "replicated": build_checkpoint(
            tiny_config(
                REAL_MODELS["phi3_4k"],
                intermediate_size=102,
                vocab_size=250,
                num_key_value_heads=4,
            ),
            tmp_path / "replicated",
        ),
    }

    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        "PYTHONPATH": os.pathsep.join(
            [str(REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")]
        ),
    }
    result = subprocess.run(
        [sys.executable, "-c", _MESH_SCRIPT, *map(str, checkpoints.values())],
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    summaries = {
        Path(summary["directory"]).name: summary
        for summary in map(json.loads, result.stdout.splitlines())
    }
    assert summaries.keys() == checkpoints.keys()

    for summary in summaries.values():
        # One device_put of each whole parameter and no other arrays.
        assert summary["placed_shapes"] == summary["layout_shapes"]
        assert summary["new_live_arrays"] == summary["leaves"]
        assert summary["footprint"] == summary["on_first_device"]

    attention = summaries["attention"]
    assert attention["attention_sharded"] and attention["mlp_sharded"]
    assert attention["sharded"] == [
        "embed",
        "head",
        "layers/down_proj",
        "layers/gate",
        "layers/gate_bias",
        "layers/k",
        "layers/k_bias",
        "layers/o_proj",
        "layers/q",
        "layers/q_bias",
        "layers/up",
        "layers/up_bias",
        "layers/v",
        "layers/v_bias",
    ]
    assert attention["lora_a_spec"] == [None, None, None, "model"]
    assert attention["lora_b_spec"] == []
    assert attention["kv_cache_spec"] == [None, None, "model", None, None]

    gemma3 = summaries["gemma3"]
    assert not gemma3["attention_sharded"] and gemma3["mlp_sharded"]
    assert gemma3["sharded"] == [
        "embed",
        "layers/down_proj",
        "layers/gate",
        "layers/up",
    ]
    assert gemma3["lora_a_spec"] == []
    assert gemma3["kv_cache_spec"] == []

    moe = summaries["moe"]
    assert not moe["attention_sharded"] and moe["mlp_sharded"]
    assert moe["sharded"] == [
        "embed",
        "head",
        "layers/expert_down",
        "layers/expert_gate",
        "layers/expert_up",
    ]

    replicated = summaries["replicated"]
    assert replicated["attention_sharded"] and not replicated["mlp_sharded"]
    assert replicated["sharded"] == [
        "layers/k",
        "layers/o_proj",
        "layers/q",
        "layers/v",
    ]
