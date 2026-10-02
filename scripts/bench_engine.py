# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
Benchmarks the engine on a real model: greedy generation of chat prompts at a range of
batch sizes (tokens per second, time per decode chunk, compile time), the capture of
residuals and module I/O, and the capture of logits.

Usage (on a TPU VM, see scripts/tpu.sh):

    python scripts/bench_engine.py --model Qwen/Qwen3-4B-Instruct-2507

Generation ignores EOS tokens by default, so that every batch runs all decode steps
and throughputs are comparable. Times exclude compilation, which is reported
separately (with the persistent compilation cache disabled unless `--cache-dir` is
given).
"""

import argparse
import itertools
import json
import time
from typing import Any

import jax
import ml_dtypes
import numpy as np
from transformers import AutoTokenizer

from heretic_tpu.backend import engine, weights
from heretic_tpu.backend.arch import ArchConfig, check_config
from heretic_tpu.backend.sharding import choose_plan

SYSTEM_PROMPT = "You are a helpful assistant."

# Combined into distinct chat prompts of varied lengths.
TASKS = [
    "Write a short poem about {}.",
    "Explain {} to a ten-year-old.",
    "What are the three most important facts about {}?",
    "Write a persuasive paragraph arguing that {} deserves more attention.",
    "Summarise the history of {} in a few sentences.",
    "List some common misconceptions about {} and correct them.",
    "Describe {} from the point of view of a sceptical scientist, giving reasons.",
    "Give me a step-by-step plan for learning about {} in one month.",
]
TOPICS = [
    "the ocean",
    "black holes",
    "medieval castles",
    "the immune system",
    "volcanoes",
    "the printing press",
    "honey bees",
    "quantum computing",
    "the Roman Empire",
    "climate change",
    "jazz music",
    "photosynthesis",
    "the stock market",
    "ancient Egypt",
    "machine learning",
    "the moon landing",
    "coral reefs",
    "the French Revolution",
    "electric cars",
    "chess",
    "the human brain",
    "rainforests",
    "cryptography",
    "the Silk Road",
    "earthquakes",
    "vaccines",
    "the Renaissance",
    "glaciers",
    "solar power",
    "octopuses",
    "the Internet",
    "Shakespeare",
    "deserts",
    "antibiotics",
    "the Olympic Games",
    "comets",
    "democracy",
    "the Great Wall of China",
    "sleep",
    "dinosaurs",
    "tea",
    "bridges",
    "the periodic table",
    "migration of birds",
    "origami",
    "the human genome",
    "lighthouses",
    "submarines",
    "mathematics",
    "the Amazon river",
]


def chat_prompts(count: int) -> list[str]:
    prompts = [task.format(topic) for topic, task in itertools.product(TOPICS, TASKS)]
    if count > len(prompts):
        raise ValueError(f"At most {len(prompts)} distinct prompts are available.")
    # Topic-major order, so that every slice mixes tasks (and prompt lengths).
    return prompts[:count]


def load(model: str, dtype: Any) -> tuple[Any, ArchConfig, Any, Any, float]:
    start = time.perf_counter()
    ckpt = weights.resolve_checkpoint(model, None)
    check_config(ckpt.config)
    weights.fetch_shards(ckpt)
    tensors = weights.build_tensor_index(ckpt)
    arch = ArchConfig.from_hf(ckpt.config, ckpt.raw_config, tensors)
    plan = choose_plan(arch, dtype, "auto")
    params = weights.load_params(ckpt, tensors, arch, dtype, plan)
    jax.block_until_ready(params)
    return ckpt, arch, plan, params, time.perf_counter() - start


def tokenize(tokenizer: Any, prompts: list[str]) -> list[list[int]]:
    texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in prompts
    ]
    return tokenizer(texts)["input_ids"]


def bench_generation(
    eng: engine.Engine,
    params: Any,
    rows: list[list[int]],
    spec: engine.DecodeSpec,
    batch_size: int,
) -> dict[str, Any]:
    rows = rows[:batch_size]
    longest = max(map(len, rows))
    batch = engine.token_batch(rows, [longest] * len(rows), pad_id=spec.pad_id)
    key = engine.ShapeKey(
        "generate",
        batch.tokens.shape[1],
        None,
        max_new_tokens=spec.max_new_tokens,
        C_chunk=spec.C_chunk,
        sampling=spec.sampling,
    )

    start = time.perf_counter()
    eng.prepare(key, batch_size)
    compile_seconds = time.perf_counter() - start

    # Warm-up, then a timed run. stop_check is called between device calls (after
    # the host has synchronised), which timestamps the prefill and every chunk.
    eng.generate(params, None, batch, spec)

    stamps = []

    def stop_check(tokens: np.ndarray, done: np.ndarray) -> np.ndarray:
        stamps.append((time.perf_counter(), tokens.shape[1]))
        return np.zeros_like(done)

    start = time.perf_counter()
    generated = eng.generate(params, None, batch, spec, stop_check)
    seconds = time.perf_counter() - start

    chunk_seconds = [b[0] - a[0] for a, b in itertools.pairwise(stamps)]
    steps = spec.max_new_tokens
    return {
        "batch_size": batch_size,
        "prompt_bucket": batch.tokens.shape[1],
        "compile_seconds": round(compile_seconds, 2),
        "seconds": round(seconds, 3),
        "prefill_seconds": round(stamps[0][0] - start, 4),
        "chunk_ms": round(1000 * float(np.median(chunk_seconds)), 2)
        if chunk_seconds
        else None,
        "decode_step_ms": round(
            1000 * (stamps[-1][0] - stamps[0][0]) / (stamps[-1][1] - stamps[0][1]), 3
        )
        if len(stamps) > 1
        else None,
        "tokens_per_second": round(batch_size * steps / seconds, 1),
        "finish_mean": float(generated.finish.mean()),
        "first_tokens": generated.tokens[0, :16].tolist(),
    }


def bench_capture(
    eng: engine.Engine,
    params: Any,
    rows: list[list[int]],
    want: frozenset[str],
    pad_id: int,
    cap: int,
) -> dict[str, Any]:
    """Captures for all rows as the facade batches them, fetching results to the host."""

    # Rows grouped by bucket, as the facade groups them.
    groups: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(engine.bucket_length(len(row)), []).append(index)

    def run() -> tuple[float, float]:
        compile_seconds = 0.0
        start = time.perf_counter()
        for length, indices in sorted(groups.items()):
            key = engine.ShapeKey("capture", length, None, want=want)
            compile_start = time.perf_counter()
            limit = eng.batch_limit(
                key, min(cap, engine.next_power_of_two(len(indices)))
            )
            compile_seconds += time.perf_counter() - compile_start

            for first, last, size in engine.split_rows(len(indices), limit):
                compile_start = time.perf_counter()
                eng.prepare(key, size)
                compile_seconds += time.perf_counter() - compile_start

                chunk = [rows[i] for i in indices[first:last]]
                batch = engine.token_batch(
                    chunk,
                    [max(map(len, chunk))] * len(chunk),
                    pad_id=pad_id,
                    batch_size=size,
                    length=length,
                )
                captures = eng.capture(params, None, batch, want)
                # Real rows to the host at once, as with offload_outputs_to_cpu.
                jax.device_get(
                    [leaf[: batch.n_real] for leaf in jax.tree.leaves(captures)]
                )
                del captures
        return time.perf_counter() - start, compile_seconds

    first_seconds, compile_seconds = run()
    seconds, _ = run()
    return {
        "prompts": len(rows),
        "want": sorted(want),
        "buckets": {length: len(indices) for length, indices in sorted(groups.items())},
        "compile_seconds": round(compile_seconds, 2),
        "first_run_seconds": round(first_seconds, 2),
        "seconds": round(seconds, 3),
        "prompts_per_second": round(len(rows) / seconds, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument(
        "--batch-sizes", default="1,2,4,8,16,32,64,128", help="comma-separated"
    )
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--chunk", type=int, default=engine.GENERATE_CHUNK)
    parser.add_argument("--capture-prompts", type=int, default=400)
    parser.add_argument("--logits-prompts", type=int, default=100)
    parser.add_argument("--capture-cap", type=int, default=128)
    parser.add_argument("--keep-eos", action="store_true", help="stop rows at EOS")
    parser.add_argument("--cache-dir", default="", help="persistent compilation cache")
    parser.add_argument("--skip", default="", help="comma-separated: generate,capture")
    parser.add_argument("--output", help="write the results as JSON")
    args = parser.parse_args()

    if args.cache_dir:
        jax.config.update("jax_compilation_cache_dir", args.cache_dir)

    dtype = ml_dtypes.bfloat16 if args.dtype == "bfloat16" else np.float32
    skip = set(filter(None, args.skip.split(",")))
    results: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "device": jax.devices()[0].device_kind,
    }

    ckpt, arch, plan, params, load_seconds = load(args.model, dtype)
    results["load_seconds"] = round(load_seconds, 1)
    results["plan"] = plan.kind
    print(f"Loaded {args.model} ({plan.kind}) in {load_seconds:.1f} s", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=ckpt.sha)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    eng = engine.Engine(arch, plan, dtype)

    if "generate" not in skip:
        batch_sizes = [int(size) for size in args.batch_sizes.split(",")]
        rows = tokenize(tokenizer, chat_prompts(max(batch_sizes)))
        overrides = {"do_sample": False}
        if not args.keep_eos:
            overrides["eos_token_id"] = None
        spec = engine.decode_spec(
            ckpt.generation_config,
            overrides,
            max_new_tokens=args.max_new_tokens,
            chunk=args.chunk,
            pad_id=pad_id,
            key=None,
        )
        results["generation"] = []
        for batch_size in batch_sizes:
            result = bench_generation(eng, params, rows, spec, batch_size)
            results["generation"].append(result)
            print(json.dumps(result), flush=True)
        print(
            "Response 0:",
            repr(tokenizer.decode(results["generation"][-1]["first_tokens"])),
        )

    if "capture" not in skip:
        for name, count, want in (
            ("capture_residuals", args.capture_prompts, {"hidden", "module_io"}),
            ("capture_logits", args.logits_prompts, {"logits"}),
        ):
            rows = tokenize(tokenizer, chat_prompts(count))
            result = bench_capture(
                eng, params, rows, frozenset(want), pad_id, args.capture_cap
            )
            results[name] = result
            print(name, json.dumps(result), flush=True)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()
