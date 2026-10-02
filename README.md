# Heretic TPU

A port of [Heretic](https://github.com/p-e-w/heretic) to Google Cloud TPUs.

Heretic removes censorship ("safety alignment") from transformer language models
without post-training. It combines directional ablation ("abliteration") and
Arbitrary-Rank Ablation (ARA) with an Optuna TPE search that co-minimises refusals
and the KL divergence from the original model. Heretic TPU runs the same workflow
on TPUs. It replaces upstream's PyTorch/Transformers backend with a purpose-built
JAX inference engine and keeps everything else:

* the configuration format (`config.toml`, CLI flags, `HERETIC_*` environment variables);
* the plugin system, modifiers, scorers and search spaces;
* the interactive trial selection, chat and benchmarks;
* export as a merged Hugging Face checkpoint or as a PEFT LoRA adapter;
* reproducibility files.

Exported models load in stock Transformers and PEFT.

## Results

On a single **TPU v6e** chip, the default configuration (ARA, 100 trials,
30 start-up trials) for `Qwen/Qwen3-4B-Instruct-2507` finishes in **19 minutes**.
That is about 11 seconds per trial at an automatically chosen batch size of 128,
with peak HBM of 10.5 of 31 GB. Upstream quotes 20–30 minutes on an RTX 3090 for
the same model.

Selected trials from that run's Pareto front:

| Trial | Refusals (harmful prompts) | KL divergence (harmless prompts) |
| :-- | --: | --: |
| Original model | 100/100 | 0 |
| 62 | 5/100 | 0.076 |
| 63 | 3/100 | 0.240 |
| 64 | 0/100 | 11.8 *(degenerate output)* |

Trial 62, exported and loaded in Transformers, answered 9 of 10 held-out harmful
prompts coherently. The original model refused all 10.

> [!NOTE]
>
> As in upstream, `trial_index = 0` in a non-interactive configuration selects the
> Pareto trial with the fewest refusals, which can be a broken model (trial 64
> above). Choose a trial by looking at the KL divergence as well.

## Requirements

* A Cloud TPU VM (developed and tested on v6e-1 with JAX 0.11.2 and libtpu 0.0.48).
  Models that do not fit on one chip are tensor-parallelised across all local
  devices. The code also runs, slowly, on CPU.
* Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).
* An unquantised safetensors checkpoint of a supported architecture (see below).

PyTorch is not needed at run time.

## Usage

```sh
git clone <this repository> heretic-tpu
cd heretic-tpu
uv sync --extra tpu
uv run heretic-tpu Qwen/Qwen3-4B-Instruct-2507
```

Everything else works as in upstream Heretic: run `uv run heretic-tpu --help`, or
copy [`config.default.toml`](config.default.toml) to `config.toml` in the working
directory and edit it. Upstream configurations keep working, including plugin names
such as `heretic.modifiers.ara.ARA`. The settings that differ are:

* `dtypes` defaults to `["auto", "bfloat16", "float32"]`; float16 is mapped to bfloat16;
* `quantization` only accepts `"none"` (bitsandbytes does not exist on TPUs);
* `device_map` and `max_memory` are gone. The new `parallelism` setting (`"auto"`,
  `"single"`, `"tensor"`) chooses between one chip and tensor parallelism;
* `compilation_cache_dir` (default `~/.cache/heretic-tpu/xla`) persists compiled
  XLA programs between runs, so later runs start much faster.

## Supported architectures

| Family | Model types |
| :-- | :-- |
| Llama (including Llama 3 RoPE scaling) | `llama` |
| Mistral, Mistral Small 3.1 | `mistral`, `mistral3` |
| Ministral 3 | `ministral3` |
| Qwen 2 / 2.5, Qwen 3 | `qwen2`, `qwen3` |
| Qwen 3 MoE, Mixtral | `qwen3_moe`, `mixtral` (attention is abliterated, as in upstream) |
| Gemma 3 (text and multimodal) | `gemma3_text`, `gemma3` |
| Phi-3, Phi-3.5, Phi-4-mini | `phi3` (including long-context RoPE) |

The following are rejected with an explanation:

* Gemma 1 and 2, whose chat templates refuse the system prompt (upstream fails on them too);
* pre-quantised checkpoints (FP8, GPTQ, AWQ, bitsandbytes);
* `trust_remote_code` models;
* hybrid layer stacks such as Qwen 3.5.

## How it works

Upstream wraps a Transformers model in PEFT adapters, reads internals with forward
hooks and generates with `model.generate`. That design defeats XLA compilation on
TPUs, so the port is built differently:

* Weights are read from safetensors on the host, stacked along a layer axis, and
  placed on the device once. The decoder is a single `lax.scan`, so it compiles
  once whatever the depth.
* The abliteration adapters are float32 arguments of the compiled programs. Trials
  change their values and never trigger a recompilation.
* Prompts are padded to power-of-two length buckets. Generation runs in compiled
  chunks over a donated, head-major KV cache, and end-of-sequence handling stays on
  the device.
* Float32 maths runs at full precision. TPUs otherwise emulate float32 matmuls with
  a single bfloat16 pass, which would silently stall ARA's optimisation.
* `torch.optim.LBFGS`, `torch.svd_lowrank` and `torch.cdist` are ported to JAX
  algorithm for algorithm. ARA's optimiser compiles once per weight shape.
* Benchmarks run through an lm-evaluation-harness adapter that reproduces HFLM's
  behaviour.

[`docs/DESIGN.md`](docs/DESIGN.md) is the full design. It ends with a list of every
behavioural divergence from upstream.

## Development

```sh
uv sync --extra tpu --group parity          # parity tests need CPU-only PyTorch, PEFT and Accelerate
uv run pytest -m "not slow"                 # unit and parity tests (CPU)
uv run pytest -m slow tests/e2e             # end-to-end CLI runs on tiny checkpoints
```

The parity tests compare the JAX engine with Transformers and PyTorch in float32.
They cover logits, hidden states, module inputs and outputs, greedy generations,
LoRA, exports, L-BFGS and lm-eval scores. Tests marked `tpu` run only on a TPU.

[`scripts/tpu.sh`](scripts/tpu.sh) syncs the working tree to a TPU VM over IAP and
runs commands there under a lock, because only one process can use the TPU at a time:

```sh
export TPU_NAME=my-tpu TPU_ZONE=europe-west4-a TPU_PROJECT=my-project
scripts/tpu.sh setup                                   # install uv and the environment on the VM
scripts/tpu.sh run 'pytest -m "not slow" tests/backend'
scripts/tpu.sh submit bench 'python scripts/bench_engine.py'   # detached; then:
scripts/tpu.sh logs bench
```

The upstream sources this port tracks are pinned in the `heretic/` submodule.

## Licence and acknowledgements

Heretic was written by Philipp Emanuel Weidmann and contributors. This port keeps
its licence, the GNU Affero General Public License v3.0 or later (see
[`LICENSE`](LICENSE)). If you use Heretic in research, please cite the original
project:

```bibtex
@misc{heretic,
  author = {Weidmann, Philipp Emanuel},
  title = {Heretic: Fully automatic censorship removal for language models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/p-e-w/heretic}}
}
```
