# heretic-tpu: design

This document is the implementation contract for heretic-tpu, the JAX/TPU port of
Heretic. Interfaces, shapes and rules stated here must be implemented exactly as
written. Every behaviour that differs from upstream is listed under
[Divergences from upstream](#divergences-from-upstream); anything not listed there
must behave as upstream does.

Upstream reference: p-e-w/heretic @ 662e4ba (optionally checked out, git-ignored, in
`heretic/`), which runs
on transformers 5.x. The rules below were checked against transformers 5.17,
lm-eval 0.4.13 and jax 0.11.2. heretic-tpu needs Python 3.12 or newer (jax 0.11
requires it). `pyproject.toml` requires `jax~=0.11.2` (releases before 0.11.2 compile
the abliteration kernel with more than one module in flight, see Testing → Memory)
and `lm-eval~=0.4.13` (whose `lm_eval.models.utils` helpers JaxLM imports), and adds
`immutabledict` and `langdetect`, which IFEval (a default benchmark) needs but lm-eval
lists only under an extra.

## Goals

* Run the complete Heretic workflow (batch-size tuning, residual analysis, Optuna
  study, scoring, export, chat, benchmarks, reproducibility) on Cloud TPUs with good
  utilisation. A single TPU chip is the primary target.
* Preserve upstream semantics: the same configuration format, plugin system,
  optimisation parameters (search space), scoring definitions and export formats
  (merged Hugging Face checkpoint or PEFT LoRA adapter). Where upstream behaviour
  can be reproduced cheaply, reproduce it.
* Exported models load and run in stock Hugging Face Transformers 5 and PEFT.
* PyTorch, PEFT and bitsandbytes are not runtime dependencies; they are used only by
  the parity tests.

## Non-goals (for now)

* Quantisation of any kind: runtime bitsandbytes quantisation, and pre-quantised
  checkpoints (fp8, compressed-tensors, GPTQ, AWQ, bitsandbytes).
* Accelerate `device_map`/`max_memory`, and data parallelism across chips. A model
  that fits on one chip runs on one chip (see [Sharding](#sharding)).
* `trust_remote_code` models, architectures outside the supported list, and
  heterogeneous layer stacks (for example Qwen3.5 linear attention), because the
  decoder must be homogeneous to be scanned.
* Bit-identical results to upstream on GPU (impossible across backends).

## Architecture overview

Upstream runs Transformers models in PyTorch, wraps them in PEFT LoRA adapters,
captures internals with forward hooks and generates with `model.generate`. On TPU
that approach is a poor fit (dynamic shapes, hooks and host round trips defeat XLA
compilation), so the port replaces the model backend with a purpose-built JAX
inference engine:

* Weights are read from the checkpoint's safetensors shards on the host, stacked
  along a leading layer axis on the host, and placed on the device once. The decoder
  is a single `jax.lax.scan` over layers and compiles once regardless of depth.
* The abliteration LoRA adapters are part of the forward pass as stacked float32
  arrays passed as jit arguments. Changing their values never recompiles.
* Captured quantities (logits, per-layer residuals, module inputs and outputs) are
  explicit outputs of jitted functions instead of hooks.
* Generation is chunked: jitted chunks of up to `C_chunk` decode steps carry a donated KV
  cache, EOS is handled on the device, and the host checks between chunks whether to
  stop early (all rows finished, lm-eval stop strings).
* Prompt lengths are padded to power-of-two buckets, and batch sizes follow the
  [engine shape policy](#engine-shape-policy). Generation packs shorter prompts into
  the batches of longer buckets, because a decode step costs about the same at any
  batch size.
* Everything above the model (config, plugins, Optuna study, evaluator, scorers,
  reproducibility, CLI) is upstream code, changed only as listed in
  [Changes to upstream modules](#changes-to-upstream-modules).

```
src/heretic_tpu/
  backend/                  JAX engine; never imports the rest of heretic_tpu
    __init__.py             sets jax_default_matmul_precision = "highest" at import
    errors.py               UnsupportedArchitectureError, UnsupportedCheckpointError,
                            DeviceMemoryError
    arch.py                 ArchConfig, built from the transformers-resolved config
    rope.py                 NumPy port of the transformers RoPE init functions
    weights.py              checkpoint resolution (online or from the Hugging Face cache),
                            quantisation checks, safetensors reading, the parameter pytree
                            layout, host stacking, placement, key and module-path mappings
    layers.py               norms, RoPE application, blocked attention, MLP and MoE primitives
    transformer.py          decoder forward (scan over layers), LoRA, captures, KV-cache primitives
    engine.py               Engine: jitted prefill/capture/generate/score entry points,
                            KV cache, shape policy, executable cache, batch helpers
    sharding.py             parallelism choice, mesh and partition specs
    linalg.py               normalize, svd_lowrank (port of torch's), k-NN mean distance
    lbfgs.py                jittable port of torch.optim.LBFGS with strong-Wolfe line search
    export.py               merged checkpoint export, PEFT adapter export and upload
    lm_eval_adapter.py      JaxLM: lm-evaluation-harness TemplateLM backed by the Engine
  model.py                  Model facade: the plugin-facing Model API
  modifiers/                Abliteration and ARA, rewritten on top of the facade
  scorers/                  KeywordRate, KLDivergence, BenchmarkScore
  config.py main.py evaluator.py plugin.py modifier.py scorer.py utils.py system.py
  reproduce.py progress.py
```

`backend/` may import JAX, NumPy, ml_dtypes, safetensors, huggingface_hub, httpx
(only for huggingface_hub's connection and timeout errors; huggingface-hub 1.x
depends on it), transformers (configs, tokenizers, `GenerationConfig`,
`TextStreamer`) and lm_eval (`lm_eval.api`, `lm_eval.models.utils`, `lm_eval.utils`).
No module of `heretic_tpu` imports torch, peft, bitsandbytes, accelerate or
`lm_eval.models.huggingface`: `tests/test_main.py` checks the sources statically and
imports the CLI, plugin and scorer modules with those modules blocked. Where torch is
installed (only with the `parity` dependency group), transformers imports it by itself
as soon as a config, `GenerationConfig` or tokenizer class is imported, which the port
cannot prevent.

### Changes to upstream modules

| Module | Change |
| :-- | :-- |
| `config.py` | See [Settings changes](#settings-changes). Built-in plugin defaults use `heretic_tpu.` names. |
| `plugin.py` | Upstream names `heretic.<...>` resolve to `heretic_tpu.<...>` (`resolve_plugin_name`). `Context.get_logits`/`get_residuals` return NumPy or JAX arrays. |
| `model.py` | Replaced by the [Model facade](#model-facade-heretic_tpumodelpy). |
| `modifiers/` | Rewritten as described under [Modifiers](#modifiers). Settings, parameter classes, `suggest_parameters`, serialisation, presentation and modifier names are unchanged. |
| `scorers/` | `KLDivergence` computes in JAX; `BenchmarkScore` uses `JaxLM`, imported inside `init`. `KeywordRate` is unchanged. |
| `main.py` | See below. |
| `utils.py`, `system.py`, `reproduce.py` | See [System information and reproducibility](#system-information-and-reproducibility). |
| `evaluator.py`, `scorer.py`, `modifier.py`, `progress.py` | Unchanged. |

`main.py`:

* No torch: drops `PYTORCH_ALLOC_CONF`, `torch.set_grad_enabled`, the TorchDynamo
  cache limit and the torch debug output. With `print_debug_information`, it prints
  `jax.print_environment_info()` and `jax.devices()` instead. It seeds with
  `random.seed` and `np.random.seed` instead of `transformers.set_seed`, which would
  import torch; JAX has no global random state, and the facade derives its keys from
  `settings.seed` (see PRNG keys).
* Enables the persistent compilation cache (`compilation_cache_dir`) before the
  `Model` is constructed, after settings have been restored from a study checkpoint.
* Batch-size tuning follows [Engine shape policy](#engine-shape-policy).
* The save and upload actions call `model.save_merged`, `model.save_adapter` and
  `model.push_to_hub`; `save_merged` writes the tokenizer and processor files itself
  (see [Export](#export)), so `main.py` no longer saves the tokenizer or processor.
  `obtain_export_strategy` loses its bitsandbytes branch and its unused `model`
  parameter. The calls to `reset_trial_model()` after a merged save or upload are
  removed, because the port's export does not modify the in-memory model;
  `reset_trial_model()` is inlined at its one remaining call site (restoring the
  chosen trial).
* The reproduce folder records the commit that was loaded (`model.checkpoint.sha`),
  see [System information and reproducibility](#system-information-and-reproducibility).
* The benchmark action builds one `JaxLM(model.tokenizer, model.lm_eval_state)`
  (imported in the action, so runs without benchmarks never load lm-eval's model
  API) and runs the original-model pass inside `with model.lora_disabled():`.
* User-facing names say heretic-tpu; the banner's version line reads
  `heretic-tpu v<version>`.

## Backend

### Matmul precision

On TPU, a JAX dot with float32 operands at the default precision is computed as a
single bfloat16 pass (precision settings have no effect on CPU). Upstream computes
float32 matmuls in full IEEE float32 (PyTorch's default; heretic never enables
TF32). Therefore:

1. `backend/__init__.py` calls
   `jax.config.update("jax_default_matmul_precision", "highest")` at import time,
   before anything is traced. Every module that traces JAX code imports
   `heretic_tpu.backend` first (`model.py` does so at the top). For an ordinary dot
   with bfloat16 operands the setting changes nothing (bf16×bf16 at HIGHEST equals
   DEFAULT bit for bit on TPU: exact products, float32 accumulation), so the decoder
   matmuls are unchanged. The exception is the ragged dot of the routed MoE MLP:
   XLA:TPU lowers it to a Mosaic kernel that fails to compile for bfloat16 operands at
   HIGHEST ("Bad lhs type"), so `layers._grouped_linear` passes its precision
   explicitly, DEFAULT for bfloat16 and HIGHEST for float32.
2. In addition, every dot, matmul and einsum in `modifiers/`, `backend/linalg.py` and
   `backend/lbfgs.py` passes `precision=jax.lax.Precision.HIGHEST` explicitly, so
   they are correct when imported without the CLI (tests, plugins).
3. `jax_enable_x64` stays disabled. Float64 work happens in NumPy on the host.

As a result the following run in true float32: the LoRA delta path, every matmul of
a float32-dtype model (the MoE grouped matmuls included), the abliteration kernels,
the user-written matmuls of `svd_lowrank`, the k-NN Gram matrix, the ARA objective
and its gradient, every dot in `lbfgs.py` (including `y·s`, `y·y`, `g·d` and the
two-loop recursion) and any device-side merge. ARA is the critical case. In upstream's
form of its objective, which multiplies the inputs by `W_eff = W_base + B @ A`, default
precision rounds `W_eff` to bf16 inside the dot, updates below half a bf16 ulp of
`W_base` vanish from the loss, and L-BFGS stops at B ≈ 0 for small
`steer_bad_behavior_weight`. The port's low-rank form (see [ARA](#ara)) adds the
adapter's contribution in float32 outside the dot, but its own products
(`in @ Aᵀ`, `W_base @ Aᵀ`, `B @ (A @ Aᵀ)`) have arbitrary float32 operands, which
default precision would round to bf16, so they run at HIGHEST as well.

### Supported architectures

**Contract.** `arch.check_config(config: PretrainedConfig) -> None` runs checks 1 to 3
below on the config alone. `ArchConfig.from_hf(config: PretrainedConfig, raw_config:
dict, tensors: TensorIndex) -> ArchConfig` calls `check_config` again and builds the
fields.

* `config = AutoConfig.from_pretrained(snapshot_dir, trust_remote_code=False)`, where
  `snapshot_dir` is the snapshot already resolved at the pinned commit (see
  [Weights](#weights)). The `KeyError` or `ValueError` AutoConfig raises for an
  unknown `model_type` or a remote-code config becomes
  `UnsupportedArchitectureError` (raised by `resolve_checkpoint`, which loads the
  config); other errors, for example an `OSError` from the network, propagate.
* `raw_config` is the parsed `config.json` of the same snapshot. Only `multimodal` is
  read from it.
* `t = config.get_text_config()`. Every other architectural field is read from the
  resolved objects `config` and `t`, never from raw config.json keys: released
  checkpoints rely on config-class defaults, and transformers 5 rewrites RoPE and
  sliding-window fields during config post-init.
* `tensors` is the `TensorIndex` built by `weights.py` (see [Weights](#weights)).
* `ArchConfig` is a frozen dataclass whose fields are all hashable: ints, floats,
  strings, bools, None, and tuples of these or of frozen dataclasses. It never holds
  NumPy or JAX arrays; arrays derived from it (RoPE tables, windows) are built by
  `rope.py` and `weights.py`. The default dataclass `==` and `hash` are used, and
  equality decides Engine reuse in `reset_model` (see the
  [Model facade](#model-facade-heretic_tpumodelpy)).

**Call sequence.** `Model.__init__` and the slow path of `reset_model` run exactly:

1. `ckpt = weights.resolve_checkpoint(model, model_commit)`: metadata only (see
   [Weights](#weights)).
2. The tokenizer is loaded (`__init__` only; see the
   [Model facade](#model-facade-heretic_tpumodelpy)).
3. `arch.check_config(ckpt.config)`. No shard is downloaded before it passes.
4. `weights.fetch_shards(ckpt)`: downloads the shard set (no-op for a local directory).
5. `tensors = weights.build_tensor_index(ckpt)`: reads headers, detects the text
   prefix, applies the quantisation backstop and finds the head key.
6. `arch = ArchConfig.from_hf(ckpt.config, ckpt.raw_config, tensors)`.

**Checks, in order.**

1. Pre-quantised checkpoints are rejected before anything else (in particular before
   any shard is downloaded and before text-prefix detection): if `config` or `t` has
   a non-null `quantization_config`, raise `UnsupportedCheckpointError` naming its
   `quant_method` and suggesting the unquantised repository. The tensor-level
   backstop is under [Weights](#weights).
2. Dispatch on the pair `(config.model_type, t.model_type)` using the table below.
   Wrappers are accepted only when their resolved text type is supported. A rejected
   pair raises `UnsupportedArchitectureError` naming the text type and listing the
   supported types.
3. Structural checks, in this order, each raising `UnsupportedArchitectureError`:
   per-layer overrides (transformers' heterogeneous configs, `is_heterogeneous`;
   checked first, because reading an overridden attribute raises); `layer_types`
   values other than `"full_attention"`/`"sliding_attention"`, or a `layer_types`
   list whose length is not L; for `qwen3_moe`, non-empty `mlp_only_layers`,
   `decoder_sparse_step != 1` or `num_experts == 0` (dense layers would make the stack
   heterogeneous); `use_bidirectional_attention` true (Gemma 3); an activation other
   than `silu`, `gelu` or `gelu_pytorch_tanh`; a RoPE type outside the list under
   RoPE below; `partial_rotary_factor != 1` outside `phi3` (transformers ignores it
   for default RoPE and fails with it for the other types, so the `rot` formula below
   would silently diverge); `longrope` or `dynamic` on some layer types but not all;
   layers with different `rot`; `longrope` with a different
   `original_max_position_embeddings` per layer type (there is one
   `rope_switch_len`).

| Text type `t.model_type` | Accepted `config.model_type` | Status |
| :-- | :-- | :-- |
| `llama` | `llama` | supported |
| `mistral` | `mistral`, `mistral3` | supported |
| `qwen2` | `qwen2` | supported |
| `qwen3` | `qwen3` | supported |
| `gemma3_text` | `gemma3_text`, `gemma3` | supported |
| `phi3` | `phi3` | supported |
| `qwen3_moe`, `mixtral` | same as text type | supported (only `attn.o_proj` is abliterable) |
| `ministral3` | `ministral3`, `mistral3` | supported (enabled after passing the parity tests; the released FP8 checkpoint is rejected as pre-quantised, the `-BF16` one is accepted) |
| `gemma`, `gemma2` | — | rejected: their chat templates reject the system role, so upstream cannot run them either |

`arch.PLANNED_ARCHITECTURES` (architectures rejected with a "planned" message) is
empty.

**Fields.** `ArchConfig` holds:

* `model_type`: the text type `t.model_type`, which selects the per-family rules;
* sizes: `num_hidden_layers` (L), `hidden_size` (D), `intermediate_size` (I),
  `num_attention_heads` (H), `num_key_value_heads` (KV), `vocab_size` (V),
  `rms_norm_eps`, `max_position_embeddings`;
* `head_dim = getattr(t, "head_dim", None) or hidden_size // num_attention_heads`;
* `activation = getattr(t, "hidden_activation", None) or t.hidden_act`;
* optional fields read with `getattr` and the default the transformers modelling code
  uses: `attn_logit_softcapping`, `final_logit_softcapping`, `query_pre_attn_scalar`,
  `norm_topk_prob`, `num_experts_per_tok`. `final_logit_softcapping` is None for the
  `gemma3` wrapper whatever its text config says, because transformers'
  `Gemma3ForConditionalGeneration` never applies it (only `Gemma3ForCausalLM`
  does);
* MoE sizes (None for dense models): `num_experts` (E; `t.num_experts` for
  `qwen3_moe`, `t.num_local_experts` for `mixtral`) and `moe_intermediate_size` (I_e;
  `t.moe_intermediate_size` for `qwen3_moe`, `t.intermediate_size` for `mixtral`);
* `biases: tuple[str, ...]`: the sorted per-layer parameter keys (a subset of `q`,
  `k`, `v`, `o_proj`, `gate`, `up`, `down_proj`) that have a bias, taken from
  `tensors` (a bias is present exactly when its key exists under the text prefix of
  layer 0);
* `tied_head`: true exactly when `tensors.head_key` is None (the head is the embedding
  matrix; see Output projection under [Weights](#weights)). It changes the parameter
  pytree, so it is part of equality;
* `multimodal`: `"vision_config" in raw_config` (the key is present, whatever its
  value; upstream's `get_model_class` rule); it selects the module-path root under
  [Weights](#weights);
* `layer_types: tuple[str, ...]` and `windows: tuple[int, ...]`, both of length L. Let
  `lt = getattr(t, "layer_types", None)`. If `lt` is None, every layer is
  `"sliding_attention"` when `getattr(t, "sliding_window", None)` is not None (mistral,
  phi3, mixtral) and `"full_attention"` otherwise (llama). The window is
  `t.sliding_window` for sliding layers and 2³⁰ (`NO_WINDOW`) for full layers and
  for sliding layers whose `sliding_window` is None (they attend to everything, as in
  transformers). The scan reads the windows as a traced int32 `[L]` buffer (see
  Parameter layout), so different windows never recompile;
* `rope: tuple[RopeSpec, ...]` of length L, `rope_per_layer_type: bool` (true when the
  RoPE parameters are given per layer type, as for Gemma 3; transformers' YaRN then
  reads `truncate` from the top-level `rope_parameters` and so always truncates),
  `rope_switch: str | None` and `rope_switch_len: int | None`, defined under RoPE
  below.

Derived quantities are properties, not fields: `rot`, `attention_scale`
(`query_pre_attn_scalar ** -0.5` if set, else `head_dim ** -0.5`), `is_moe`,
`has_qk_norm` (`qwen3`, `qwen3_moe`, `gemma3_text`) and `has_sandwich_norms`
(`gemma3_text`).

**RoPE.** One RoPE primitive serves every architecture. `backend/rope.py` is an exact
NumPy port, in float32, of the transformers RoPE init functions for the types
`default`, `linear`, `dynamic`, `llama3`, `yarn` and `longrope` (for `phi3`, the
resolved config has already mapped the legacy `su`/`yarn` names to `longrope`). Any
other `rope_type` is rejected. The one exception to the float32 port is
`base ** exponent`: PyTorch's float32 `pow` is a vectorised approximation, accurate to
about an ulp, that differs between platforms, so `rope.py` evaluates it in float64
and rounds to float32 (the correctly rounded value). `inv_freq` therefore matches
transformers to at most 1 ulp (bit for bit for about 99% of entries on CPU).
`number / tensor` is evaluated as `reciprocal(tensor) * number`, as PyTorch does.

* `RopeSpec` is a frozen dataclass `(rope_type: str, rot: int, params: tuple[tuple[str,
  Hashable], ...])`, where `params` is the sorted items of the layer's resolved
  `rope_parameters` dict with lists converted to tuples, and
  `rot = int(head_dim * partial_rotary_factor)` with `partial_rotary_factor =
  rope_parameters.get("partial_rotary_factor", 1.0)`. All layers must have the same
  `rot`.
* If `t.layer_types` exists and every layer type is a key of `t.rope_parameters`
  (Gemma 3), parameters are taken per layer type. Otherwise the flat
  `t.rope_parameters` applies to every layer.
* `rope.build_tables(arch) -> RopeTables` returns NumPy float32 `inv_freq [L, rot/2]`,
  `attention_scaling [L]` and, for `longrope` only, `long_inv_freq [L, rot/2]`.
  `weights.py` places them as buffers of the parameter pytree (see Parameter layout).
* `rope_switch` is `"long_factor"` for `longrope` (with `rope_switch_len =
  original_max_position_embeddings`), `"raise"` for `dynamic` (with `rope_switch_len =
  max_position_embeddings`, the transformers threshold) and None otherwise. Its use is
  defined under Forward pass semantics.
* `original_max_position_embeddings` is `rope_parameters.get(...)`, falling back to
  `getattr(t, ...)`.
* `linear`: `inv_freq / factor`. `llama3` and `yarn`: exactly as transformers
  (`yarn` includes `beta_fast`, `beta_slow`, the `truncate` default and its
  `attention_factor`).
* `dynamic`: the base is unchanged while the sequence-length measure (Forward pass
  semantics) is at most `max_position_embeddings`; beyond it the port raises
  `NotImplementedError`.
* `longrope`: `inv_freq = 1 / (ext * theta ** (arange(0, rot, 2) / rot))`, where
  `ext` is the short (`inv_freq`) or long (`long_inv_freq`) factor list (each of length
  rot/2). cos and sin are multiplied by `attention_factor` at every position, for
  short and long factors alike, so one `attention_scaling` table serves both.
  `attention_factor` is `rope_parameters["attention_factor"]` if given;
  otherwise, with `factor = rope_parameters.get("factor") or max_position_embeddings / original_max_position_embeddings`,
  it is `sqrt(1 + ln(factor) / ln(original_max_position_embeddings))` when
  `factor > 1`, else 1 (1.1902 for Phi-3-mini-128k, Phi-3.5-mini and Phi-4-mini).
* Application: cos/sin are computed in float32 from the elementwise product
  `position * inv_freq[l]` (not a matmul; for `longrope`, each row uses
  `long_inv_freq[l]` or `inv_freq[l]` according to its traced long-factor flag),
  `emb = concat(freqs, freqs)`, multiplied by
  `attention_scaling[l]` and cast to the model dtype. q and k are rotated on their
  first `rot` dims with `rotate_half` inside those dims; dims `rot:` pass through
  unchanged.

**Per-family rules.**

| Text type | Rules |
| :-- | :-- |
| `llama` | Llama RMSNorm; optional biases; any accepted RoPE type. |
| `mistral` | `sliding_window` (when non-null) applies to every layer. Inside `mistral3` it is the text model of Mistral-Small-3.1 (no RoPE scaling). |
| `qwen2` | q/k/v biases. Sliding windows only as resolved by the config (`use_sliding_window` false nulls the window, so every layer is full attention). |
| `qwen3` | Per-head q/k RMSNorm (Llama style, over `head_dim`) applied before RoPE. Windows as for `qwen2`. |
| `gemma3_text` | Gemma RMSNorm `(1 + w)`; embedding output multiplied by `sqrt(hidden_size)` cast to the model dtype; sandwich norms (`input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`); per-head Gemma-style q/k norm before RoPE; attention scale `query_pre_attn_scalar ** -0.5`; window only on sliding layers. RoPE per layer type: released 4B/12B/27B checkpoints use `linear` with factor 8 on base 1e6 for full-attention layers only and `default` on base 1e4 for sliding layers; legacy top-level `rope_scaling` is never applied to sliding layers. Soft-capping (both null in released checkpoints): `attn_logit_softcapping` is applied when non-null, although transformers 5.17 ignores it on every attention path, eager included, because `Gemma3Attention` never passes `softcap` (Gemma 2's does); see Divergence 2. `final_logit_softcapping` is applied when non-null for `gemma3_text` (as `Gemma3ForCausalLM` does) and never for the `gemma3` wrapper (as `Gemma3ForConditionalGeneration`). |
| `phi3` | Fused weights split on the host at load time: `qkv_proj` rows are q (H·hd rows), then k (KV·hd rows), then v (KV·hd rows); `gate_up_proj` rows are gate (I rows), then up (I rows), and the MLP computes `down(up * act(gate))`. Partial rotary (Phi-4-mini: hd 128, rot 96). `longrope` as above. `sliding_window` (when non-null, e.g. 2047 for Phi-3-mini-4k) applies to every layer. Factor switching under Forward pass semantics. |
| `qwen3_moe`, `mixtral` | Attention as `qwen3`/`mistral`; the MLP is a routed MoE (below). Only `attn.o_proj` is abliterable. |
| `ministral3` | `yarn` RoPE exactly as transformers' `_compute_yarn_parameters` (`beta_fast`, `beta_slow`, `factor`, `original_max_position_embeddings`, the `truncate` default, `attention_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)`, which is 1.0 for the released checkpoints); after RoPE, queries are multiplied by `1 + llama_4_scaling_beta * log(1 + floor(position / original_max_position_embeddings))` computed in float32 from the same position ids and cast to the model dtype (1 below 16384 tokens, but always applied); `llama_4_scaling_beta` and `original_max_position_embeddings` are read from `ArchConfig.rope[0].params`, which are already part of equality. The head follows the output-projection rule under Weights, which reads the top-level `config.tie_word_embeddings` as transformers 5 does when it ties the wrapper's weights (not `text_config.tie_word_embeddings`). For the released checkpoints both are true (the top-level key is absent, so the Mistral3Config default applies) and there is no `lm_head` tensor, so the head is the embedding matrix. |

**MoE.** The API keeps `[L, M, ...]` shapes with `M = 1`, so per-expert modules could
be added later; today only `attn.o_proj` is abliterable on these models, which is what
upstream finds on transformers 5, where experts are fused 3-D parameters. ARA runs on
`attn.o_proj`. Routing, per token `x` (model dtype):

* router logits `x @ gate.weightᵀ` in the model dtype (`gate.weight [E, D]`), then
  `probs = softmax(logits.astype(float32))`, then top-k over `probs`
  (`k = num_experts_per_tok`, ties broken towards the lower expert index);
* `qwen3_moe`: if `norm_topk_prob`, divide the k weights by their sum (float32); cast
  the weights to the model dtype; each contribution is
  `expert_e(x) * w_e` in the model dtype;
* `mixtral`: always divide the k weights by their sum and keep them float32; each
  contribution is `(expert_e(x) * w_e)` computed in float32 and cast to the model
  dtype;
* combination (both): per token, `out = 0` in the model dtype, then for each selected
  expert in **ascending expert index** (not top-k rank order),
  `out = (out + contribution_e)` rounded to the model dtype after every addition;
* `expert_e(x) = down_e(act(gate_e(x)) * up_e(x))` with expert width I_e
  (`moe_intermediate_size`). On disk, Qwen3-MoE stores
  `mlp.gate.weight` and `mlp.experts.{e}.{gate,up,down}_proj.weight`; Mixtral stores
  `block_sparse_moe.gate.weight` and `block_sparse_moe.experts.{e}.w1` (gate), `w3`
  (up) and `w2` (down). Fused `mlp.experts.gate_up_proj [E, 2·I_e, D]` (`[gate; up]`)
  and `mlp.experts.down_proj [E, D, I_e]` are accepted too.
* Numerics follow the eager per-expert loop of transformers, not its default
  grouped-matmul path. A combine by einsum or matmul with a `[T, E]` weight matrix
  (float32 accumulation) is not allowed.
* Implementation (`layers.routed_moe`): only the selected experts are computed. Each
  token's k experts are sorted by ascending index, the `(token, expert)` pairs are
  grouped by expert with a stable sort, and the gate, up and down projections are
  grouped matmuls (`lax.ragged_dot_general` with float32 accumulation, rounded to the
  model dtype, like `linear`; precision as under
  [Matmul precision](#matmul-precision)). The contributions are then summed in the
  sorted order, rounding after every addition.
* On the TPU, XLA's default `xla_allow_excess_precision` lets fused elementwise code
  skip some of these intermediate bf16 roundings, so bf16 results there can be a bf16
  ulp away from the rule in places. The bit-exact combination-order test therefore
  runs on the CPU; the TPU tests compare with transformers to a few bf16 ulps.
* For router probabilities that tie exactly at the top-k boundary (possible in bf16),
  the port selects the lower expert index (`lax.top_k`); `torch.topk` on the CPU was
  observed to pick the higher one (see Divergence 2).

### Weights

**Checkpoint resolution.** `weights.resolve_checkpoint(model, model_commit) ->
Checkpoint` downloads metadata only. `Checkpoint` is a frozen dataclass with
`model: str` (the Hub id or local directory as given, which `fetch_shards` and the
export need), `snapshot_dir: str`, `sha: str | None` (None for a local directory),
`offline: bool` (below; False for a local directory), `raw_config: dict` (parsed
config.json), `config: PretrainedConfig` (resolved, as under Supported
architectures), `index: dict | None` (parsed `model.safetensors.index.json`, None when
absent or stale), `shard_files: tuple[str, ...]` and
`generation_config: GenerationConfig` (resolved as under [Engine](#engine)).

* If `model` is a Hub id:
  1. Resolve the commit once and download the metadata:
     `snapshot_dir = snapshot_download(model, revision=model_commit,
     allow_patterns=["config.json", "generation_config.json",
     "model.safetensors.index.json"])` and `sha = basename(snapshot_dir)` (snapshot
     directories are named after their commit). `snapshot_download` also records the
     commit of a branch or tag under `refs/` in the cache, where a later offline run
     finds it. Then `HfApi().model_info(model, revision=sha)` lists the files of the
     commit. Every later download uses `revision=sha`, so the index and the shards come
     from the same commit.
  2. **Unreachable Hub.** If step 1 fails because the Hub cannot be reached
     (`OfflineModeIsEnabled`, i.e. `HF_HUB_OFFLINE`; an `httpx` connection error or
     timeout; an HTTP 5xx response; or HTTP 429 or 408, which huggingface_hub also
     treats as transient and which the Hub's API returns on its own when it
     rate-limits a machine), the commit the local cache records for `model_commit`
     (the default branch if unset) is used instead, as transformers (and so upstream)
     does: `snapshot_download(..., local_files_only=True)`, with a `UserWarning`
     saying that the cached commit could not be checked against the Hub, and
     `offline = True`. The cached snapshot must hold config.json and the index or
     model.safetensors, otherwise `LocalEntryNotFoundError` is raised. Missing
     repositories, revisions and gated access (`RepositoryNotFoundError`,
     `RevisionNotFoundError`, `GatedRepoError`, other client errors) are raised and
     never hidden by the cache. Unlike upstream, an unpinned offline run also works
     after the model was loaded by heretic-tpu alone, because step 1 records the
     branch.
  3. The shard set (`_shard_set`), where "a file exists" means that it is in the
     commit's file list, or, when `offline`, in the cached snapshot:
     * no index: `["model.safetensors"]` if it exists; if neither exists, raise
       `UnsupportedCheckpointError` saying that only safetensors checkpoints are
       supported;
     * an index all of whose shards exist: exactly
       `sorted(set(index["weight_map"].values()))`, even if a `model.safetensors`
       exists as well;
     * an index naming missing shards (a **stale index**): if `model.safetensors`
       exists, it is loaded instead with a warning saying the index is stale (or,
       offline, that its shards "are not cached"), and `index` is None; otherwise
       raise `UnsupportedCheckpointError` naming the missing shards (offline, the
       index is kept and `fetch_shards` reports the missing shards).

     transformers loads `model.safetensors` whenever it exists. The port prefers a
     complete index because `save_pretrained` leaves the other layout's files behind
     when it saves into a folder holding an earlier save of the other layout: an
     unsharded save deletes the old shards but keeps the old index, and a sharded save
     keeps the old `model.safetensors`. Missing shards therefore mark an index as
     stale, and a complete index as newer (Divergence 11).
  4. `weights.fetch_shards(ckpt)` (step 4 of the call sequence, after
     `check_config`) downloads exactly the shard set with
     `snapshot_download(..., revision=sha, allow_patterns=shard_set)`. When
     `ckpt.offline`, it only checks that the shard set is in the cached snapshot and
     raises `LocalEntryNotFoundError` naming the shards that are not cached.
* If `model` is a local directory, the same shard-set rule applies to its files; never
  glob.
* Any other file (for example `consolidated*.safetensors`, `original/`,
  `params.json`, `*.pth`, `*.pt`, `*.bin`) is never downloaded, opened or exported.
* The tokenizer is loaded after `resolve_checkpoint` (step 2 of the call sequence):
  `AutoTokenizer.from_pretrained(model, revision=ckpt.sha)` for a Hub id (the commit
  `model_commit` resolved to, so tokenizer and weights come from one commit), and
  `AutoTokenizer.from_pretrained(model, **revision_kwargs)` for a local directory. It
  fetches its own files. `AutoProcessor` is never instantiated (it needs Pillow and
  torchvision).
* The facade keeps the `Checkpoint` of the currently loaded model; its `sha` is used
  by the export and recorded in the reproduce folder.

**Tensor index.** `weights.build_tensor_index(ckpt) -> TensorIndex` reads only the
safetensors headers of the shard set:

```python
@dataclass(frozen=True)
class TensorInfo:
    shard: str                  # shard file name, relative to snapshot_dir
    shape: tuple[int, ...]
    dtype: str                  # safetensors dtype string, e.g. "BF16", "F8_E4M3"

@dataclass(frozen=True)
class TensorIndex:
    snapshot_dir: str
    tensors: Mapping[str, TensorInfo]   # every key of every shard in the shard set
    prefix: str                         # the text prefix P (below)
    head_key: str | None                # the head tensor key, None when tied (below)
```

It detects `P`, then runs the quantisation backstop, then resolves the head key, in
that order.

**Reading.** `weights.py` imports `ml_dtypes` at module top (this registers bfloat16
with NumPy, so reading does not depend on import order; `ml_dtypes` is declared as a
direct dependency). Tensors are read only with
`safetensors.safe_open(path, framework="np")`, which returns host NumPy arrays in the
stored dtype. Do not use `framework="flax"`/`"jax"` or `safetensors.flax.load_file`
(they create unsharded arrays on device 0) or `safetensors.numpy.load` (it cannot
read BF16). Storage dtypes and shapes come from `get_slice(name).get_dtype()` and
`.get_shape()` without reading data. Tensors outside the text model (vision tower,
projector) are never read for inference.

**Quantisation backstop.** After prefix detection, raise `UnsupportedCheckpointError`
if any tensor loaded as a linear or embedding weight has a storage dtype outside
{F32, BF16, F16}, or if any key under the text prefix ends in `weight_scale`,
`weight_scale_inv`, `qweight`, `qzeros`, `scales` or `g_idx`.

**Text prefix.** The text-model prefix `P` is the one of `model.`,
`model.language_model.` and `language_model.model.` for which
`P + "layers.0.self_attn.o_proj.weight"` exists. Exactly one must match, otherwise
`UnsupportedCheckpointError` is raised.

**Output projection.** The head key depends on the layout: `lm_head.weight` for
`P = "model."` or `"model.language_model."`, and `language_model.lm_head.weight` for
`P = "language_model.model."`.

* If that tensor exists, `head_key` is its key and it is used, whatever any
  `tie_word_embeddings` flag says (transformers also refuses to tie two present
  tensors that differ, and Mistral3Config defaults the flag to true although
  Mistral-Small-3.1 ships a distinct head).
* If it is absent and the top-level resolved `config.tie_word_embeddings` is true,
  `head_key` is None and the head is the embedding `P + "embed_tokens.weight"`. For
  Gemma the tied head is the unscaled embedding matrix; the `sqrt(hidden_size)` scale
  applies only to the lookup output.
* Otherwise raise `UnsupportedCheckpointError` (transformers would initialise a
  random head).

**Host stacking and placement.** For each stacked parameter, `weights.py` allocates
one host NumPy buffer `[L, ...]` (MoE experts `[L, E, ...]`) in the session dtype and
fills it layer by layer; Phi-3 splits and dtype casts (for example float16 to
bfloat16) happen on the host during the fill. The buffer is placed with a single
`jax.device_put(buffer, sharding)`, waited for and dropped; `gc.collect(0)` follows
each placement, because jaxlib defers releasing a transferred host buffer until Python
next calls into it or collects garbage. Non-stacked parameters (embedding, final norm,
head) and the buffers built from `ArchConfig` (RoPE tables from `rope.build_tables`,
windows) are placed the same way. `weights.py` never creates per-layer or unsharded
device arrays and never stacks or concatenates on the device. Peak device memory is
the placed model size; peak host memory is one stacked parameter plus the page cache.
Before reading any data, `load_params` checks that every tensor of every parameter
exists with the expected shape (`UnsupportedCheckpointError` otherwise) and runs the
pre-flight below. While it reads, a background thread reads every shard file that
holds text-model tensors sequentially into the page cache (stopped and joined before
`load_params` returns): the per-layer reads of each stacked parameter are spread over
the files and on their own reach only part of the disk's sequential speed. In shards
that mix text and vision tensors this also pulls vision bytes into the page cache,
although they are never parsed or loaded.

**Pre-flight.** Before reading tensor data, `weights.py` computes the per-device
footprint from the headers and the partition specs (sharded bytes divided by the
model-axis size plus replicated bytes, in the session dtype). If
`device.memory_stats()` is not None, a footprint above `bytes_limit` (not
`bytes_limit − bytes_in_use`) raises `DeviceMemoryError` naming the required and
available bytes, so the dtype loop in `Model.__init__` moves on. The check is skipped
when `memory_stats()` is None (CPU).

**Name mappings.** For every `(component, layer_index, module_index)`:

* `disk_key(...)`: the safetensors key (`P + f"layers.{i}.self_attn.o_proj.weight"` or
  `P + f"layers.{i}.mlp.down_proj.weight"`), its shard file and its storage dtype.
  Used only by the merged export.
* `module_path(...)`: the transformers 5 module path that upstream's
  `named_modules()` gives, from a static table (PyTorch is not available at run
  time). The root is `model.language_model.layers.{i}` when `ArchConfig.multimodal`
  (gemma3 and mistral3 wrappers, loaded by upstream through
  `AutoModelForImageTextToText`), otherwise `model.layers.{i}`. The suffix is
  `self_attn.o_proj` for `attn.o_proj` and `mlp.down_proj` for `mlp.down_proj`.
  Never derive it from the disk key: gemma3 and mistral3 checkpoints, including those
  written by transformers 5's `save_pretrained`, store `language_model.model.layers.*`,
  which transformers renames at load time.

### Parameter layout, components and LoRA

Parameters are a pytree of nested dicts defined in `weights.py` (`param_layout`,
tabulated in its module docstring, with `param_skeleton` giving the
`jax.ShapeDtypeStruct` tree for ahead-of-time lowering); `transformer.py` consumes it.
HF orientation `[out, in]` is kept for every linear layer. Per-layer tensors are
stacked on a leading `L` axis under `layers/`:

* `embed [V, D]`, `head [V, D]` (absent when `ArchConfig.tied_head`), `final_norm [D]`;
* per layer: `attn_norm [L, D]`; `q [L, H·hd, D]`, `k`, `v [L, KV·hd, D]` and their
  biases `q_bias`, `k_bias`, `v_bias` when present; `q_norm`, `k_norm [L, hd]`
  (`has_qk_norm`); `o_proj`; `attn_out_norm [L, D]` (Gemma 3); `mlp_norm [L, D]`
  (`post_attention_layernorm`, or Gemma 3's `pre_feedforward_layernorm`); dense
  `gate`, `up [L, I, D]` (and biases) and `down_proj`, or MoE `router [L, E, D]`,
  `expert_gate`, `expert_up [L, E, I_e, D]` and `expert_down [L, E, D, I_e]`;
  `mlp_out_norm [L, D]` (Gemma 3);
* `buffers`: `inv_freq [L, rot/2]` float32, `long_inv_freq [L, rot/2]` float32
  (`longrope` only), `attention_scaling [L]` float32 and `window [L]` int32, built on
  the host from `ArchConfig` and placed replicated. They are ordinary leaves of the
  params pytree, so they are traced jit arguments, owned by the facade and released
  with the parameters;
* abliterable components (upstream names `"attn.o_proj"` and `"mlp.down_proj"`, keys
  `o_proj [L, 1, D, H·hd]` and `down_proj [L, 1, D, I]`) are stored as
  `[L, M, d_out, d_in]` in the model dtype, with `M = 1` for every component of every
  supported architecture, so `get_base_weights` returns the stored array without a
  copy. Their biases `o_proj_bias` and `down_proj_bias`, when present, are
  `[L, M, d_out]`.

LoRA is a pytree `{component: (A [L, M, r, d_in], B [L, M, d_out, r])}`, both float32.
The adapted module computes, exactly like PEFT with an fp32 adapter and
`lora_alpha = r`:
`y = (base(x).astype(f32) + (x.astype(f32) @ Aᵀ) @ Bᵀ).astype(x.dtype)`,
where `base(x) = x @ Wᵀ (+ bias)` rounded to the model dtype. A forward pass without
adapters (`lora=None`, a separate compiled program) is used before any modifier has
run and inside `lora_disabled()`.

### Forward pass semantics

Numerics follow the transformers eager implementations:

* Linear layers: `x @ Wᵀ` in the model dtype with float32 accumulation, output
  rounded to the model dtype; biases added in the model dtype. Residual additions in
  the model dtype.
* Llama RMSNorm: normalise in float32, cast to the model dtype, then multiply by the
  weight in the model dtype. Gemma RMSNorm: normalise in float32, multiply by
  `(1 + w)` in float32, cast.
* q/k norms (`has_qk_norm`): `lax.optimization_barrier` is applied to the q and k
  projections before their norms, so that they are materialised in the model dtype,
  as the eager recipe rounds them. Without it, XLA:TPU fuses the norms into the
  projections, keeps their float32 accumulators and moves them into the attention
  layout in float32 (a tenth of Qwen3-4B's prefill time). Values are unchanged.
* Activations: `silu`; `gelu` is the exact erf GELU; `gelu_pytorch_tanh` is the tanh
  approximation.
* Position ids: `cumsum(mask) - 1`, with padding positions set to 0 (as
  transformers' generate). Decode steps use the next real position of each row.
* Attention: scores `(q @ kᵀ)` in the model dtype times the scale
  (`head_dim ** -0.5`, or `query_pre_attn_scalar ** -0.5` for Gemma 3); if
  `attn_logit_softcapping` is set, `scores = tanh(scores / cap) * cap`; then masked
  entries are replaced by `finfo(model dtype).min` (after the cap, so masked entries
  are never squashed to `-cap`); softmax in float32, cast to the model dtype; then
  `@ v` in the model dtype. Grouped-query attention shares each KV head across
  `H / KV` query heads. Fully masked padding rows produce finite values, never NaN.
* Mask: key slot `j` is visible from query slot `i` iff `mask[j]`, `j <= i` and,
  for windowed layers, `i - j < window`. Slots are cache slots; because padding is
  only on the left and generated tokens are contiguous, slot distance equals position
  distance for real tokens, as in transformers.
* Prefill attention, and any attention with more than one query, is blocked over
  queries: a `lax.map` over chunks of `q_blk` queries (`q_blk = min(T, 512)`, or 256
  when T > 8192). Prompt buckets are powers of two, so `q_blk` divides them; for any
  other length (the long-RoPE re-prefill over `T_cache` slots, below) the query axis
  is padded with masked queries to the next multiple of `q_blk` and the padded
  outputs are discarded. Each block uses
  the recipe above, so temporaries are `[B, H, q_blk, T]` and never `[B, H, T, T]`. The
  mask is built per block from slot indices and the padding mask; no `[B, T, T]` mask
  is materialised. Results are identical to unblocked eager attention. Pallas splash
  attention (an optional fast path for T ≥ 1024 that would have to pass the parity
  tests) is not implemented; blocked attention serves every length.
* **RoPE switching** (`ArchConfig.rope_switch` not None). Transformers decides per
  forward call from `max(position_ids) + 1` over the batch it is given, which upstream
  is its own batch. The port reproduces that batch per row: every `TokenBatch` and
  `ScoreBatch` row carries `ref_len`, the real token count of the longest real row of
  its reference batch (defined by the caller: the upstream batch for the facade, the
  JaxLM batch for JaxLM). The row's sequence-length measure is `ref_len` in a single
  forward call (capture, scoring, prefill) and `ref_len + s` at decode step `s` (the
  step that consumes generated token `s`, `s ≥ 1`). With `S = rope_switch_len`, the
  row's switch step is `s_switch = max(0, S − ref_len + 1)`, the first step whose
  measure exceeds S (0 means from prefill). The Engine computes `s_switch` on the
  host from `ref_len`.
  * `"long_factor"` (longrope): a row uses `long_inv_freq` from its switch step on,
    `inv_freq` before it. The choice is a traced per-row flag, so rows of different
    reference batches can share an engine batch.
  * `"raise"` (dynamic): in a single forward call the Engine raises
    `NotImplementedError` before running if any real row has `s_switch == 0`. In
    generation the chunk stops at `stop_at` exactly as below, and the host raises
    instead of re-prefilling if an unfinished row has reached its switch step, so a
    batch whose rows all finish earlier never raises (as upstream, whose positions
    stop growing when generation stops).
  * Generation with `"long_factor"`: the chunk function takes a traced `stop_at`, the
    earliest switch step among unfinished rows that is still ahead, and returns to
    the host before executing that step. The host then runs the **re-prefill**
    executable in place of that decode step: it runs the decoder over all `T_cache`
    slots (prompt tokens followed by the tokens generated so far, including the token
    being consumed; unwritten slots masked; positions `cumsum(mask) − 1`), with each
    row's factor choice for this step, rebuilds the whole KV cache (the old cache is
    donated) and selects the next token exactly as a decode step does (logits at each
    row's newest slot, repetition penalty, argmax or sampling, EOS, `pad_id` for
    finished rows). Rows that do not switch at this step are recomputed with their
    short factors, which changes them only at rounding level. Decoding then resumes
    with the chunk function. One re-prefill runs per distinct switch step among
    unfinished rows. This equals transformers' generation with `use_cache=False`.
* Final norm, then the head (or the tied embedding), giving logits in the model dtype;
  if `final_logit_softcapping` is set, `logits = tanh(logits / cap) * cap` in the model
  dtype. Logits are returned as float32.
* On the TPU, XLA's default `xla_allow_excess_precision` may keep intermediate values
  of fused elementwise bf16 code in float32, skipping some of the roundings these
  recipes specify, so bf16 results there differ from the recipes at rounding level.
  Bit-exact bf16 tests therefore run on the CPU.

Captured outputs, all at the **last prompt position** (the position from which the
first new token is generated, the last column under left padding):

* `logits`: `[B, V]` float32, raw (after final soft-capping, before any logits
  processor).
* `hidden`: `[B, L + 1, D]` in the model dtype, following the transformers
  `hidden_states` convention: index 0 is the (scaled) embedding output, index `i` for
  `1 <= i < L` is the output of layer `i - 1`, and index `L` is the final-norm-applied
  output of the last layer.
* `module_io[component]`: `(inputs [B, L, M, d_in], outputs [B, L, M, d_out])` in the
  model dtype. The input of `attn.o_proj` is the concatenated attention output; the
  input of `mlp.down_proj` is `act(gate) * up`. The output is that of the adapted
  module (base plus bias plus LoRA delta) when adapters are active and not disabled,
  as upstream's hooks on PEFT-wrapped modules see it.

### Engine

`backend/engine.py` owns the compiled entry points. Parameters and LoRA are always
explicit jit arguments; no jitted function closes over a device array (JAX would
embed it as a constant and recompile per value).

```python
StopCheck = Callable[[np.ndarray, np.ndarray], np.ndarray]

class Engine:
    adapters_allocated: bool    # set by the owner of the parameters, see `reserve`

    def __init__(self, arch: ArchConfig, plan: ShardingPlan, dtype: jnp.dtype, *,
                 offload_outputs_to_cpu: bool = True): ...

    def capture(self, params, lora, batch: TokenBatch,
                want: frozenset[str]) -> Captures
        # want ⊆ {"logits", "hidden", "module_io"}; prefill only, never [B, T, V] logits.
    def generate(self, params, lora, batch: TokenBatch, spec: DecodeSpec,
                 stop_check: StopCheck | None = None) -> Generated
    def stream(self, params, lora, batch: TokenBatch, spec: DecodeSpec) -> Iterator[np.ndarray]
        # B = 1, C_chunk = 1: yields each new token as it is produced.
    def score(self, params, lora, batch: ScoreBatch) -> ScoreOutputs
    def batch_limit(self, key: ShapeKey, cap: int) -> int        # B_eff, see shape policy
    def lower_limit(self, key: ShapeKey, failed_b: int) -> int   # after a runtime OOM
    def prepare(self, key: ShapeKey, batch_size: int) -> None    # compile and check fits
    def fits(self, key: ShapeKey, batch_size: int) -> bool
    def memory_need(self, key: ShapeKey, batch_size: int) -> int
    def memory_analysis(self, key: ShapeKey, batch_size: int) -> dict[str, Any]
    def rank(self, lora: Lora | None) -> int | None
        # the shape keys' LoRA rank; ValueError unless the adapters cover exactly the
        # abliterable components with one shared rank
    def release(self) -> None   # empties the executable and batch-limit caches

def decode_spec(generation_config: GenerationConfig, overrides: dict, *,
                max_new_tokens: int, chunk: int, pad_id: int,
                key: jax.Array | None) -> DecodeSpec
def is_out_of_memory(e: BaseException) -> bool
def bucket_length(length: int) -> int          # max(32, next_power_of_two(length))
def token_batch(rows, ref_len, *, pad_id, batch_size=None, length=None) -> TokenBatch
def score_batch(rows, ref_len, candidates, *, pad_id, C_score, G=None,
                batch_size=None, length=None) -> ScoreBatch
```

* The Engine holds only Python state: `arch`, `plan`, `dtype`,
  `offload_outputs_to_cpu`, `adapters_allocated`, the executable cache and the
  batch-limit cache. It never holds a `jax.Array`. The facade and JaxLM build batches
  with `token_batch` and `score_batch` (left padding to the bucket, filler rows that
  repeat the last real row) and derive shape-key ranks with `rank`.
* Every entry point runs exactly the batch it is given (`B = batch.tokens.shape[0]`).
  It never splits, shrinks or retries a batch; batching is the caller's job (see
  [Engine shape policy](#engine-shape-policy)). Before the first run of an executable
  at `(key, B)` it evaluates `fits(key, B)` and raises `DeviceMemoryError` if it is
  false. Out-of-memory errors propagate.
* `TokenBatch`: `tokens int32 [B, T]` left-padded to the bucket, `mask bool [B, T]`,
  `n_real` (rows from `n_real` on are filler rows whose results are discarded) and
  `ref_len int32 [B]`, the real token count of the longest real row of the row's
  reference batch (upstream batch for the facade, JaxLM batch for JaxLM; filler rows
  copy the value of the row they repeat). `ref_len` drives the repetition-penalty pad
  rule (`pad_penalised[row] = sum(mask[row]) < ref_len[row]`) and RoPE switching.
* `DecodeSpec`: static `max_new_tokens`, `C_chunk` (32 for batch generation, 1 for
  streaming) and `sampling: bool`; traced `eos_ids int32 [8]` (padded with −1; more
  than 8 ids raises), `pad_id`, `repetition_penalty` (1.0 when unset), `temperature`,
  `top_k`, `top_p`, `min_p` (neutral values 1.0, 0, 1.0, 0.0 when unset) and the PRNG
  key (None when not sampling).
* `decode_spec` is the one place that turns a generation config into a `DecodeSpec`.
  It merges `overrides` (the facade's `{"do_sample": False}` for greedy calls, or an
  lm-eval request's normalised kwargs) over the generation config as
  `GenerationConfig.update` would, sets `sampling = do_sample`, and warns once about
  unsupported processors and warpers (listed below). The facade and JaxLM both call
  it.
* `StopCheck(tokens, done) -> finish_now`: `tokens int32 [n_real, s]` are the tokens
  generated so far (`pad_id` after a row finished), `done bool [n_real]` the finished
  mask; it returns `bool [n_real]`, true for rows to mark finished now (entries for
  rows already finished are ignored). Filler rows are never passed.
* `Generated`: `tokens int32 [n_real, max_new_tokens]` (`pad_id` after a row
  finished) and `finish int32 [n_real]`: the number of tokens up to and including the
  row's first EOS; for a row marked by `stop_check`, the number of tokens generated
  when it was marked; `max_new_tokens` otherwise.

**Generation parameters** are resolved as transformers does for `generate()`:
`GenerationConfig.from_pretrained(snapshot_dir)` when generation_config.json exists,
else `GenerationConfig.from_pretrained(snapshot_dir, config_file_name="config.json",
_from_model_config=True)`, i.e. from the raw config.json, which keeps the token ids
and legacy generation parameters that the resolved config fills with class defaults
or drops (`GenerationConfig.from_model_config(config)` would lose them); fields left
unset take
`GenerationConfig._get_default_generation_params()` (so `top_k = 50` applies when
sampling and the checkpoint sets none). `eos_ids` is the generation config's
`eos_token_id` (an int or a list). `pad_id` is `tokenizer.pad_token_id`. The resolved
config is `Checkpoint.generation_config` of the currently loaded checkpoint.

**Greedy generation** (`DecodeSpec.sampling` false: `get_responses` and the smoke
test, which pass `do_sample = False` as upstream does, and lm-eval `generate_until`
requests whose normalised `do_sample` is false). Each step takes the float32 logits
of every row, applies the repetition penalty, takes `argmax` (first maximal index),
writes `pad_id` for rows that have already finished, and marks a row finished when
its token is in `eos_ids`. Other processors active in the generation config
(`no_repeat_ngram_size > 0`, `bad_words_ids`, `min_length`/`min_new_tokens > 0`,
forced BOS/EOS, `suppress_tokens`, `begin_suppress_tokens`, `sequence_bias`,
`exponential_decay_length_penalty`, `renormalize_logits`, `remove_invalid_values`)
and beam search (`num_beams > 1`) are not applied and produce one warning each (once
per process). An override that is not a generation-config field is ignored with a
warning.

**Repetition penalty.** The penalty p is a traced scalar, always applied (p = 1.0
when the generation config leaves it unset, which is an exact identity), to the
float32 logits of each row after final soft-capping and before argmax or sampling,
once per distinct token: `s < 0 ? s * p : s / p`. The penalised set of a row is kept
as a `[B, V]` boolean mask and is:

* every real prompt token, plus
* every token generated so far, plus
* `pad_id`, only if `pad_penalised[row]`, i.e. only if upstream would have
  left-padded that row: the row is shorter than `ref_len[row]`, the longest row of its
  upstream batch (defined under the [Model facade](#model-facade-heretic_tpumodelpy)).

Bucket padding and filler rows never add `pad_id`. For Qwen2.5 (pad and one EOS are
the same id, and the prompt contains the other EOS) this reproduces upstream's EOS
penalties exactly.

**Chunked decoding.** Prefill builds the KV cache and selects the first token. Then
the host calls a jitted chunk function repeatedly. A chunk is a `lax.while_loop` of
at most `C_chunk` steps that stops early when every row is finished, when
`max_new_tokens` tokens exist or when the step count reaches the traced `stop_at`
(RoPE switching; `max_new_tokens` when there is none); the decode step count,
positions, the done mask, `stop_at` and the penalised-set mask are traced, so there is
one chunk compile per `(B, T_cache, C_chunk, sampling, rank)`. The KV cache and the
other carried buffers are donated to every chunk call. Between chunks the host stops
when all rows are finished, runs the re-prefill when `stop_at` was reached (see
Forward pass semantics), and, if `stop_check` is given, calls it with the real rows'
tokens generated so far and done mask and marks the rows it returns as finished (used
for lm-eval stop strings; their later tokens are `pad_id`). Output is truncated at
`max_new_tokens`; the cache has `T_cache = T_bucket + max_new_tokens` slots. The host
synchronises once per device call (it fetches the step count and done mask, plus the
tokens when streaming or when `stop_check` is given). With `max_new_tokens = 1` (the
smoke test) the prefill alone produces the token: no cache is built and no chunk is
compiled.

**Sampling** (`DecodeSpec.sampling` true: streaming chat when the generation config
has `do_sample`, and lm-eval requests whose normalised `do_sample` is true): over the
whole templated conversation plus generated tokens, apply in this order: repetition
penalty; temperature (if ≠ 1.0); top-k (if ≠ 0, keeping at least one token); top-p (if
< 1.0, keeping at least one token); min-p (if > 0); then softmax and a categorical
draw. Sampling is its own compiled program (static `sampling`). Its parameters are
traced, so k and p are never static: temperature and min-p are applied under
`jnp.where`, and top-k and top-p under `lax.cond` on their values. Top-k keeps every
logit ≥ the k-th largest and top-p removes the tokens (in ascending order) whose
cumulative probability is at most `1 − top_p` except the most likely one, as
transformers does; tokens with equal scores are kept or removed together. Both
thresholds are found by bisection (32 masked reductions) over int32 keys that order
like the float32 scores, not by sorting the vocabulary, which on TPU costs more than
the decode step itself. `1 − top_p` is computed in double precision on the host and
rounded to float32, as transformers' threshold is. Other warpers (`top_h`,
`typical_p < 1`, `epsilon_cutoff`, `eta_cutoff`) produce one warning. The PRNG key
comes from the facade (see PRNG keys under the
[Model facade](#model-facade-heretic_tpumodelpy)); step `s` uses
`jax.random.fold_in(key, s)`.

**Scoring** (for JaxLM): `ScoreBatch` holds left-padded `tokens [B, T]`, `mask`,
`n_real`, `ref_len int32 [B]` (RoPE switching only), `cand int32 [B, G]` with
`cand_mask`, a static `C_score` (power-of-two bucket of the longest continuation) and
a static `G` (power-of-two bucket of the largest candidate count; JaxLM chooses both
per group of rows, see [JaxLM](#lm-eval-adapter-jaxlm)). Prefill runs the decoder;
final norm and unembedding are applied only to the
last `C_score` columns, in a `lax.map` over those columns, each producing in float32
the log-softmax normaliser, the target log-probability and the argmax.
Outputs (NumPy, real rows only): `next_lp [n_real, C_score]` and `next_greedy
[n_real, C_score]` (column `T − C_score + j` scored against the input token at column
`T − C_score + j + 1`; the last column gives 0 and true), and `last_lp [n_real, G]`
and `last_greedy [n_real, G]` for the candidate tokens at the last column.
`[B, T, V]` (or `[B, C_score, V]` as one buffer) is never materialised.

**KV cache.** A pair `(keys, values)` of `[L, B, KV, T_cache, hd]` arrays in the
model dtype: head-major, with the slot axis next to the head dimension, which is the
layout XLA:TPU's decode attention reads. (With a slot-major `[L, B, T_cache, KV, hd]`
cache, every chunk call copied the whole cache into that layout and back: 3.2 GB of
scratch at B = 128 for Qwen3-4B, and far worse for streaming with `C_chunk = 1`.) The
cache is carried by the decode loop and sharded on the KV axis (axis 2,
`ShardingPlan.kv_cache_sharding`) when attention is sharded.

* Prefill builds the cache in its final shape: each layer's scan step returns its K/V
  (keys after RoPE, values as projected) transposed to `[B, KV, T, hd]` and padded to
  `T_cache` as `ys`, so the stacked `ys` is the cache and no second copy is made.
* Layout pinning (`engine.kv_cache_layout`): when `head_dim` is a multiple of 128, the
  cache crosses executable boundaries in row-major layout, set with
  `jax.experimental.layout.Format` on the outputs of the prefill, chunk and
  re-prefill, and the prefill and re-prefill constrain each layer's K/V scan output to
  it with `with_layout_constraint`, so that XLA stacks it in place. Otherwise XLA:TPU
  chooses an entry layout that minimises padding, which for a `T_cache` that is not a
  multiple of 8 puts the slots outside the heads and copies the cache on every call.
  For narrower heads the compiler's default is kept: row-major would pad them to 128
  lanes, and the default measured faster for `head_dim` 64.
* The per-token layer scan never returns a full per-layer cache slice as `ys`: each
  layer reads its slice as `xs` and returns only the new `[B, KV, hd]` K/V as `ys`,
  and one `dynamic_update_slice` writes `[L, B, KV, 1, hd]` at the current slot after
  the scan, which updates the donated cache in place.
* Decode attention covers the cached slots before the current one plus the current
  token's K/V (cast to the cache dtype) as an extra column. The extra column gets the
  same scaling and soft-capping, is always inside the window, and the unwritten
  current slot and later slots are masked. The cached-slot scores and the extra
  column's score are computed separately and concatenated before the softmax; the
  extra column's value contribution is added to the cached-slot product in float32
  before one rounding, which equals a single matmul up to summation order.

### Engine shape policy

**Shape keys.** Everything static except the batch size is in one key:

```python
class ShapeKey(NamedTuple):
    entry: str                        # "capture", "generate", "stream" or "score"
    T: int                            # prompt bucket
    rank: int | None                  # LoRA rank; None is the lora=None program
    want: frozenset[str] = frozenset()   # capture only
    max_new_tokens: int = 0           # generate, stream
    C_chunk: int = 0                  # generate (32), stream (1)
    sampling: bool = False            # generate, stream
    C_score: int = 0                  # score: continuation bucket
    G: int = 0                        # score: candidate bucket
```

Executables are compiled ahead of time (`jit(...).lower(...).compile()`) and cached by
`(ShapeKey, B)`; one key may own several executables (`"generate"`: prefill, chunk
and, for `longrope`, re-prefill). `T_cache = T + max_new_tokens` and `q_blk` are
derived from the key. The parameter pytree structure, the dtype and the shardings are
fixed for the life of an Engine (they follow from `ArchConfig`, dtype and plan), so
they are not part of the key; a different combination gets a new Engine. Everything
else is traced: token ids, masks, positions, `ref_len`, LoRA values, all trial
parameters, the params `buffers` (RoPE tables, windows), `eos_ids`, `pad_id`, the
repetition penalty and sampling parameters, the PRNG key, the decode step, `stop_at`
and the done mask. The persistent XLA compilation cache is enabled (see
`compilation_cache_dir`).

**Prompt lengths** are left-padded to `T = max(32, next_power_of_two(longest real row))`.

**Batch sizes.**

* Auto mode, `settings.batch_size == 0` (before tuning has fixed the batch size: the
  `Model.__init__` smoke test and the tuning loop): the facade sends the whole call as
  one batch, `B = n`, `T` the bucket of its longest row, no filler rows. It never
  calls `batch_limit` or `lower_limit`, never halves and never retries: the
  `DeviceMemoryError` of the `fits` check and XLA out-of-memory errors propagate to
  the caller.
* `settings.batch_size > 0` (cap = `settings.batch_size`), and JaxLM (cap = 64): the
  caller groups rows (facade captures: by bucket T, stable order; JaxLM: as under
  [JaxLM](#lm-eval-adapter-jaxlm); facade generation: packed, below). For a group of
  `n_g` rows, `B_eff = engine.batch_limit(key, min(cap, next_power_of_two(n_g)))`; the
  group is split into chunks of `B_eff` rows, and a chunk with fewer rows is padded to
  `min(B_eff, next_power_of_two(rows))` with filler rows repeating its last real row,
  so the number of compiled batch sizes stays logarithmic. Results are returned in the
  original order. Each row's computation is independent of the other rows (up to
  floating-point rounding), so grouping does not change results; the upstream batch
  membership needed by the repetition penalty, RoPE switching and response lengths
  is computed before grouping and carried in `ref_len`.
* Facade generation (`get_responses*`, `max_new_tokens` per row) packs rows across
  buckets. A decode step costs about the same at any batch size, because it is
  dominated by reading the weights, so fewer, fuller batches save whole decodes (for
  the default refusal prompts on Qwen3-4B, one decode per call instead of two). Rows
  are taken longest bucket first, in their original order within a bucket. Each batch
  takes its bucket T from its first row and is filled with the following rows,
  left-padded to T, as long as `T + max_new_tokens ≤ 2 (T_row + max_new_tokens)`
  (a row's KV-cache length at most doubles); B_eff is
  `batch_limit(key(T), min(cap, next_power_of_two(eligible rows)))`, recomputed for
  every batch (which gives the same sizes as computing it once per group). Captures
  keep per-bucket grouping, because prefill cost scales with B × T. Packing changes
  results only through the rounding of the bucket a row runs at (Divergence 7).
* If running a chunk raises an error satisfying `is_out_of_memory`, the caller calls
  `engine.lower_limit(key, B)`, which records B as failing and returns the halved
  `B_eff` (B = 1 raises `DeviceMemoryError`), and re-runs that chunk's rows from
  prefill in chunks of the new size (donated buffers are not reused). The rest of the
  key's rows also run at the lowered size, because `batch_limit` skips sizes recorded
  as failing.
* Streaming chat always runs at B = 1.

**B_eff.** `Engine.batch_limit(key, cap)` returns `cap` if `cap` is at most the
largest B recorded as fitting for `key`. Otherwise it tries `B = cap, cap // 2, …, 1`,
skipping sizes at or above the smallest B recorded as failing: it compiles the key's
executables at B, and B fits if compiling raises no out-of-memory error and
`fits(key, B)` is true. The first fitting B is recorded and returned; B = 1 failing
raises `DeviceMemoryError`. The records live for the life of the Engine.

* `is_out_of_memory(e)`: `isinstance(e, jax.errors.JaxRuntimeError)` and
  `"RESOURCE_EXHAUSTED"` in `str(e)`, or `isinstance(e, DeviceMemoryError)`. XLA:TPU
  usually reports HBM overflow at compile time.
* `fits(key, B)`: true when `memory_stats()` is None (CPU). Otherwise
  `need ≤ free − reserve`, where `free = min over devices (bytes_limit − bytes_in_use)`;
  `need = temp_size + output_size − alias_size + transient_argument_bytes` from
  `compiled.memory_analysis()` (for generation: the prefill output including the
  cache plus the largest of the prefill, chunk and re-prefill temporaries);
  `transient_argument_bytes` counts only per-call inputs (tokens, masks, positions,
  candidates), never params or adapters, which are already in `bytes_in_use`.
  `peak_memory_in_bytes` is never used, because it includes resident arguments.
* `reserve = 0.05 × bytes_limit`, plus, for a key without adapters (`rank` None)
  while no adapters exist, the per-device float32 adapter bytes of every component at
  rank 50 (the ARA default, the largest default rank), so that batch sizes found
  before `apply_lora` still fit afterwards, plus `0.10 × bytes_limit` when
  `offload_outputs_to_cpu` is false. The Engine learns whether adapters exist from
  its public attribute `adapters_allocated`, which the facade sets in `apply_lora`
  and clears in `_release_device_state`. Resident adapters are part of
  `bytes_in_use`, so calls without adapters while they are allocated (inside
  `lora_disabled()`, or lm-eval on the original model) do not reserve them a second
  time.

**Batch-size tuning** (`main.py`, `batch_size == 0`). Candidates 1, 2, 4, … ≤
`max_batch_size` call `model.get_responses(prompts)` with exactly that many prompts
(auto mode: `B = n`, no halving), with a warm-up run as upstream. A candidate fails
iff the call raises an error satisfying `is_out_of_memory` (re-exported by
`heretic_tpu.model`, so `main.py` imports only the facade module); that includes the
`DeviceMemoryError` the Engine raises when `fits` is false. Any other exception
propagates. The first failure above B = 1 ends the loop; a failure at B = 1
is re-raised, as upstream. (An optional warm-up compile of the LoRA executables in
`apply_lora` was considered and not implemented.)

### Sharding

`backend/sharding.py` chooses a plan from the `parallelism` setting:

* `"single"`: everything on `jax.local_devices()[0]` with `SingleDeviceSharding`; no
  mesh and no collectives. This is the fast path.
* `"tensor"`: a 1-D mesh over all local devices with axis `"model"` (size n).
  Attention is sharded only when both `num_attention_heads % n == 0` and
  `num_key_value_heads % n == 0`; then q/k/v output rows, the `attn.o_proj` input axis
  (and its LoRA `A` input axis) and the KV-cache head axis are sharded, otherwise all
  of attention is replicated. The MLP intermediate axis (`gate`/`up` rows,
  `mlp.down_proj` input axis and its LoRA `A`) is sharded when `I % n == 0`; for MoE,
  the expert intermediate axis is sharded when `I_e % n == 0`. Embedding and head rows
  are sharded when `V % n == 0`. Everything else, LoRA `B`, norms, the params
  `buffers` and captured outputs, is replicated.
* `"auto"` (default): `"single"` when there is one local device, when
  `memory_stats()` is None, or when the model's footprint in the session dtype plus
  `0.05 × bytes_limit` and the float32 rank-50 adapters fits within one device's
  `bytes_limit`; otherwise `"tensor"`. The `0.10 × bytes_limit` output reserve of
  `fits` is not included, because `choose_plan` does not see
  `offload_outputs_to_cpu`.

`sharding.choose_plan(arch, dtype, parallelism) -> ShardingPlan`. `ShardingPlan` is a
frozen dataclass compared by value: `kind`, `device_ids`, `mesh_shape`, `specs` (the
partition spec of every parameter path) and the derived flags `attention_sharded` and
`mlp_sharded`, which give the LoRA shardings (`lora_shardings(component)`: `A` sharded
like the module's input axis, `B` replicated) and the KV-cache sharding
(`kv_cache_sharding`); the plan depends only on `ArchConfig`, dtype, the setting and
the devices.

Correctness of `"tensor"` is tested on CPU with
`--xla_force_host_platform_device_count`.

## Model facade (`heretic_tpu/model.py`)

Plugins and `main.py` talk only to the facade. When
`settings.offload_outputs_to_cpu` is true (the default) inference methods return
NumPy arrays and free the device buffers at once; otherwise they return
`jax.Array`s. Plugins should use `jax.numpy` operations, which accept both.

```python
ModuleIO = dict[str, tuple[Array, Array]]
# component -> (inputs [L, M, N, d_in], outputs [L, M, N, d_out]), model dtype, M = 1

class LMState(NamedTuple):
    engine: Engine
    params: Params
    lora: Lora | None
    hf_config: PretrainedConfig            # top-level resolved config of the current model
    generation_config: GenerationConfig    # resolved from the current model's checkpoint
    pad_id: int                            # tokenizer.pad_token_id
    next_key: Callable[[], jax.Array]      # the facade's sampling-key factory (PRNG keys)

def is_out_of_memory(e: BaseException) -> bool   # re-exported from backend.engine

class Model:
    settings: Settings
    tokenizer: PreTrainedTokenizerBase     # pad_token defaults to eos_token, padding_side="left"
    dtype: jnp.dtype                       # session parameter/compute dtype
    revision_kwargs: dict[str, str]        # {"revision": model_commit} if set
    trusted_models: set[str]               # always empty (remote code is unsupported)
    lora_rank: int | None                  # None until apply_lora() and after a slow-path reload

    # The loaded model; hf_config and the generation config are read from checkpoint.
    checkpoint: Checkpoint
    tensors: TensorIndex
    arch: ArchConfig
    plan: ShardingPlan
    # The device state, owned only by the facade (see Device memory lifetime).
    engine: Engine | None
    params: Params | None
    adapters: Lora | None

    def __init__(self, settings: Settings): ...

    # Structure
    def get_layers(self) -> list[int]                      # list(range(L)); len() is the layer count
    def get_abliterable_components(self) -> list[str]      # sorted
    def get_module_count(self, component: str) -> int      # M (always 1)
    def get_base_weights(self, component: str) -> jax.Array  # [L, M, d_out, d_in], the stored array

    # Adapters
    def apply_lora(self, lora_rank: int) -> None
    def get_lora(self, component: str) -> tuple[jax.Array, jax.Array]   # (A, B)
    def set_lora(self, component: str, A: ArrayLike, B: ArrayLike) -> None
    def reset_model(self) -> bool
    def lora_disabled(self) -> ContextManager[None]

    # Inference (prompts are heretic_tpu.utils.Prompt)
    def get_responses(self, prompts, skip_special_tokens=False) -> list[str]
    def get_responses_batched(self, prompts, skip_special_tokens=False) -> list[str]
    def get_logits(self, prompts) -> Array                   # [N, V] float32
    def get_logits_batched(self, prompts) -> Array
    def get_residuals(self, prompts, winsorization_quantile=1.0) -> Array          # [N, L+1, D] float32
    def get_residuals_batched(self, prompts, winsorization_quantile=1.0) -> Array
    def get_residuals_mean(self, prompts, winsorization_quantile=1.0) -> np.ndarray  # [L+1, D] float32
    def get_module_io(self, prompts) -> ModuleIO
    def get_module_io_batched(self, prompts) -> ModuleIO
    def stream_chat_response(self, chat: list[dict[str, str]]) -> str

    # Benchmarks
    def lm_eval_state(self) -> LMState

    # Export (see "Export")
    def save_merged(self, directory: str) -> None
    def save_adapter(self, directory: str) -> None
    def push_to_hub(self, repo_id: str, *, private: bool, token: str, strategy: ExportStrategy) -> None
```

**`__init__`.** Runs the call sequence under
[Supported architectures](#supported-architectures): resolve the checkpoint (giving
`sha`), load the tokenizer from that commit (pad token and left padding as upstream),
check the config, fetch the shards, build the `TensorIndex` and the `ArchConfig`. The
generation config is `Checkpoint.generation_config`. It then tries `settings.dtypes`
in order, choosing the sharding plan for each attempt with `sharding.choose_plan`
(the `"auto"` choice depends on the dtype):

* `"auto"` resolves as transformers' `_get_dtype`: the top-level config `dtype` (or
  legacy `torch_dtype`), which transformers applies to every sub-config; if unset,
  the first floating-point storage dtype that is not float8 or float4. float16 is
  then mapped to bfloat16.
* An explicit `"float16"` entry is mapped to `"bfloat16"` with a warning. An entry
  whose effective dtype was already tried is skipped. Any entry other than `auto`,
  `float16`, `bfloat16` and `float32` (for example `float64`) is a failed attempt:
  printed, then the next entry is tried. Settings keep the entries exactly as given
  (restored upstream settings contain `["auto", "float16", "bfloat16", "float32"]`).
* Each attempt loads into local variables and runs a smoke test: greedy generation of
  one token for `Prompt(settings.system_prompt, "What is 1+1?")` at B = 1 (prefill,
  processors and argmax; no decode chunk is compiled). Only when both succeed is the state
  assigned to `self`. On any exception the attempt's state is released (see
  [Device memory lifetime](#device-memory-lifetime)), the failure is printed as
  upstream, and the next dtype is tried. A dtype whose parameters do not fit fails
  before any data is read, through the pre-flight of `weights.load_params` (under
  [Weights](#weights)), so no separate size check is made. When every entry fails,
  `RuntimeError` is raised (upstream raises a bare `Exception` with the same
  message).
* Prints the layer count and the abliterable components with `L × M` modules each,
  as upstream.

**Adapters.**

* `apply_lora(r)` allocates adapters for every component: `A` initialised like PEFT
  (kaiming-uniform with `a = sqrt(5)`, i.e. `U(-1/sqrt(d_in), 1/sqrt(d_in))`) from
  `jax.random.fold_in(jax.random.fold_in(jax.random.key(settings.seed), 1), i)` for
  the i-th component in sorted order, and `B` zeros. It sets
  `engine.adapters_allocated` (see `reserve` under
  [Engine shape policy](#engine-shape-policy)).
* `set_lora` validates shapes, casts to float32, places with the plan's sharding and
  stores a new adapters dict, so that an `LMState` handed out earlier is unchanged.
* `get_lora`, `set_lora`, `save_merged`, `save_adapter` and `push_to_hub` raise
  `RuntimeError("The model has no adapters. Call apply_lora() first.")` before
  `apply_lora` (and after a slow-path reload).
* `lora_disabled()` makes every forward pass use `lora=None` inside the block (it
  nests).
* `reset_model()`, fast path (`settings.model` equals `checkpoint.model`, the model
  that is loaded): `B = 0` and `A` re-initialised exactly as in `apply_lora` (nothing
  to do before `apply_lora`); returns True.
* `reset_model()`, slow path (for example `--evaluate-model`, whose checkpoint may use
  a different wrapper, architecture or storage dtype), in this order:
  1. Run steps 1 and 3 to 6 of the call sequence for the new `settings.model` with
     `model_commit` from `revision_kwargs` (as upstream): new `Checkpoint` (with its
     `sha`, config, raw config and generation config), `TensorIndex` and `ArchConfig`.
     The tokenizer is not reloaded (as upstream).
  2. `new_plan = sharding.choose_plan(new_arch, self.dtype, settings.parallelism)`.
  3. `_release_device_state(keep_engine=(new_arch, new_plan) == (old_arch, old_plan))`.
  4. Load the weights in `self.dtype` (no dtype loop and no smoke test, as upstream).
  5. If the Engine was released, build `Engine(new_arch, new_plan, self.dtype)`.
  6. Replace the facade's `Checkpoint` (so the export and the adapter's `revision`
     use the new `sha`; `hf_config` and the generation config are read from it),
     `TensorIndex`, `ArchConfig` and plan, so `eos_ids`, `repetition_penalty` and the
     sampling defaults of every later generation (KeywordRate, chat, JaxLM) and
     `JaxLM.max_length` come from the evaluated checkpoint, as with upstream's
     reloaded model. Set `lora_rank = None` and the adapters to None.
  7. Return False.

**Prompt formatting** is identical to upstream: `apply_chat_template` with a system
and a user message, `add_generation_prompt=True`, `tokenize=False`, then
`settings.response_prefix` appended, then the tokenizer called on the strings with
default special-token handling. `stream_chat_response` templates the whole chat the
same way without the response prefix.

**Upstream batches.** For the repetition-penalty pad rule, RoPE switching and
response lengths, the upstream batch of a prompt is the whole call for the
non-batched methods (`get_responses`, `get_logits`, `get_residuals`,
`get_module_io`) and its chunk of `batchify(prompts, settings.batch_size)` for the
`*_batched` methods. In auto mode (`settings.batch_size == 0`, before tuning), the
`*_batched` methods and `get_residuals_mean` treat the whole call as one upstream batch
(upstream's `batchify(prompts, 0)` would raise). A row's `ref_len` is the largest
token count among the prompts of its upstream batch, computed before grouping; so
`pad_penalised[row]` is true iff the row's token count is below the longest in its
upstream batch, and the Phi-3 factor choice matches the upstream batch.

**PRNG keys.** `settings.seed` must be an int in `[0, 2**32 − 1]`; otherwise the
facade raises `ValueError` before deriving any key (`jax.random.key` silently
truncates larger seeds without x64, and upstream's `transformers.set_seed` already
rejects them through `np.random.seed`). The seed is validated when a key is derived
(`apply_lora`, a fast-path reset, `_next_key`), not in `__init__`, so a facade can be
built with `seed=None` (`main.py` always sets a seed first). Keys are derived from
`root = jax.random.key(settings.seed)` with distinct first-level tags: 1 for adapter
initialisation, 2 for abliteration's `svd_lowrank`, 3 for sampling. The facade owns a
sampling-call counter `c` (starting at 0); every sampling batch (each
`stream_chat_response` call, even when the generation config is greedy, and each
sampling JaxLM batch) takes `jax.random.fold_in(jax.random.fold_in(root, 3), c)` from
`Model._next_key()` and increments `c`. `LMState.next_key` is that bound method.

**Responses.** For each upstream batch b, `L_b` is the largest `Generated.finish`
among its rows. Each row's response is the decode of its first `L_b` generated tokens
(positions after its EOS hold `pad_id`), with the requested `skip_special_tokens`.
This reproduces the trailing pad tokens that upstream's `batch_decode` returns when
`skip_special_tokens` is false.

**Residuals.** `get_residuals` casts `hidden` to float32, applies winsorisation as
upstream (`quantile(|r|, q, axis=-1, method="linear")` per prompt and layer, then
clamp to ±threshold) and returns real rows only. `get_residuals_mean` processes
`batchify(prompts, settings.batch_size)` batches, fetches each batch's winsorised
float32 residuals for real rows only to the host, accumulates
`np.sum(..., axis=0, dtype=np.float64)` in NumPy, divides by the number of real
prompts and returns float32. No x64 JAX arrays are used.

**Module I/O.** `get_module_io` and `get_module_io_batched` return the stacked
`ModuleIO` layout above, with semantics as defined under Forward pass semantics. The
Engine returns `[B, L, M, d]` per engine batch. With `offload_outputs_to_cpu` true,
each batch's real rows are fetched to the host at once (and the device buffers
deleted), concatenated in NumPy in the original prompt order to `[N, L, M, d]` and
transposed with `np.ascontiguousarray(np.transpose(x, (1, 2, 0, 3)))`, giving NumPy
arrays. With it false, the real rows are concatenated and transposed on the device
(`jnp.concatenate`, `jnp.transpose`), giving `jax.Array`s.

**Chat.** `stream_chat_response` tokenises the templated chat with default special
tokens, runs `Engine.stream` at B = 1 with `max_new_tokens = 4096` (the module
constant `CHAT_MAX_NEW_TOKENS`, which tests shorten) and the resolved
generation parameters (sampling if the generation config says so, repetition penalty
over every token of the conversation), feeds each new token as a NumPy array to
`transformers.TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)`
and returns the decode of the generated tokens with `skip_special_tokens=True`.

**`lm_eval_state()`** returns the current Engine, params, LoRA (None before
`apply_lora`, inside `lora_disabled()` and after a slow-path reload), resolved config,
generation config, `pad_id` and `next_key`. It is called at request time, never
cached by callers.

### Device memory lifetime

JAX frees device memory only when the last reference to an array dies, so ownership
is explicit:

1. The facade is the only long-lived owner of the param pytree and the adapter
   arrays. Engine entry points take them as arguments and never close over them,
   capture them as constants or store them. Modifiers and JaxLM do not retain them
   across trials or calls; re-fetching through `get_base_weights`, `get_lora` or
   `lm_eval_state` is the supported route. Callers pass `get_base_weights` results
   straight into jitted functions and never convert the full stack to float32 or keep
   derived copies across trials.
2. `Model._release_device_state(keep_engine: bool = False)` does, in order:
   1. `leaves = [x for x in jax.tree.leaves((self.params, self.adapters)) if isinstance(x, jax.Array)]`;
   2. `self.params = None`, `self.adapters = None`;
   3. if `self.engine` is not None: clear `engine.adapters_allocated` (a kept Engine
      reserves room for adapters again until they are applied), and, if `keep_engine`
      is false, `self.engine.release()` and `self.engine = None` (if `keep_engine` is
      true, the Engine and its executables are kept);
   4. `x.delete()` for each leaf that is not already deleted (tied leaves appear
      twice; stale references then fail loudly);
   5. `del leaves`, then `gc.collect()`.

   The Engine holds no device arrays, so keeping it costs only its executables.
3. The slow path of `reset_model` calls it before reading any checkpoint tensor.
4. In the `__init__` dtype loop, any exception calls it before the next attempt. The
   loader builds into local variables and never stores partial results on `self` or
   in a cache, so no try/finally deletion of partial arrays is needed.

## Modifiers

Both modifiers keep upstream's settings, `suggest_parameters`, parameter
serialisation and presentation, `reset_model` and modifier names. Their float32 work
runs at `Precision.HIGHEST` (see [Matmul precision](#matmul-precision)).

### Abliteration

* `init`: as upstream. Residual means come from `get_residuals_mean` for the good and
  bad prompts; the residual directions (normalised difference, projection orthogonal
  to the good direction when `orthogonalize_direction`, renormalised) are computed in
  NumPy float32 on the host. `lora_rank` is 1 unless `row_normalization = "full"`, in
  which case it is `full_normalization_lora_rank`. Then `model.apply_lora(rank)`.
* `modify_model`, host part: for each component and layer l,
  `distance = |l - max_weight_position|`; `apply[l] = distance <=
  min_weight_distance and weight != 0`, where `weight = max_weight + (distance /
  min_weight_distance) * (min_weight - max_weight)`, all in Python floats as upstream.
  The direction array `v [L, D]` float32 is the interpolated, normalised global
  direction broadcast to every layer when `direction_index` is set (index shifted by
  1 for the embedding entry, `lerp` by the fractional part, as upstream), otherwise
  `residual_directions[l + 1]` per layer. Both cases share one executable.
* `modify_model`, device part: one jitted function per (component shape,
  `row_normalization`, rank). It iterates sequentially over the flattened
  `(layer, module)` axis of length `L·M` with
  `jax.lax.map(body, (W, v, weight, apply, A_reset))` **without** `batch_size` (with
  any `batch_size`, `lax.map` vmaps the body, which turns `lax.cond` into a select
  that runs both branches), or an equivalent `lax.scan`. `W` is
  `get_base_weights(component)` reshaped to `[L·M, d_out, d_in]` (model dtype, not
  donated), `A_reset` is the current (reset) `A` from `get_lora(component)` reshaped
  the same way, and `v`, `weight`, `apply` are repeated over M. It never vmaps over
  layers or modules, so at most one float32 `[d_out, d_in]` working set is live and
  the bf16 stack is never converted to float32. `body` is `lax.cond(apply, compute, skip)`;
  `skip` returns the reset `A` and `B = 0` and does no float32 conversion,
  normalisation or `svd_lowrank` (never multiply by the weight instead: the full path
  produces a non-zero, noise-only delta at weight 0). Outputs are reshaped to
  `[L, M, r, d_in]` / `[L, M, d_out, r]` and written with one `set_lora`.
* `compute`, line by line as upstream with `W = W_module.astype(f32)` and
  `W_org = W`: if `row_normalization != "none"`: `W_row_norms = ‖W‖` per row and
  `W = normalize(W)`; `A = (v @ W)[None, :]` and `B = (-weight * v)[:, None]`;
  `"pre"`: `B = W_row_norms * B`; `"full"`: `W' = normalize(W + B @ A) * W_row_norms`,
  `Δ = W' - W_org`, `U, S, V = linalg.svd_lowrank(Δ, q=2r+4, niter=6, key)`,
  `B = U[:, :r] · diag(sqrt(S[:r]))`, `A = diag(sqrt(S[:r])) · V[:, :r]ᵀ`. The key is
  `jax.random.fold_in(jax.random.key(settings.seed), 2)` (see PRNG keys under the
  [Model facade](#model-facade-heretic_tpumodelpy)), the same for every module,
  mirroring upstream's reseed before every call.

### ARA

* `init`: as upstream: module I/O for the good and bad prompts via
  `get_module_io_batched` before adapters exist, then `model.apply_lora(lora_rank)`.
* `suggest_parameters` is unchanged; its upper bound for `neighbor_count` is the
  shared constant `NEIGHBOR_COUNT_MAX = 15`.
* One module-level function, `modifiers/ara.py::ara_optimise(A, B, W_stack, layer,
  module, good_in, good_out, bad_in, bad_out, loss_weights, k, state)`, jitted with
  static `preserve_row_magnitudes`, `max_iter`, `history_size`, `learning_rate`,
  `k_max_good` and `k_max_bad`. Everything else is traced: `W_stack` is
  `get_base_weights(component)` (not donated, sliced inside the function with the
  traced `layer`, `module`), the module I/O arrays are float32,
  `loss_weights = f32[3]` (preserve, steer, overcorrect) and `k` is int32. Nothing is
  closed over, so there is one compile per component shape (two per run for dense
  models), independent of modules, trials and `neighbor_count`.
* Inside: `W_base = W_stack[layer, module].astype(A.dtype)` (float32 in production;
  following `A`'s dtype lets the float64 parity test run under `jax.enable_x64`), then
  `data = objective_data(W_base, good_in, good_out, bad_in, bad_out, loss_weights, k)`
  computes once per call what does not depend on the adapter: `‖w_i‖²` and `‖w_i‖`
  per row `i` of `W_base` (`[d_out]` each) and the base outputs
  `good_in @ W_baseᵀ`, `bad_in @ W_baseᵀ` (fields of the `ara.ObjectiveData` pytree).
  Then one outer L-BFGS step (`lbfgs.step`) on the objective over
  `x = concat(A.ravel(), B.ravel())`.
* The objective is upstream's (`W_eff = W_base + B @ A`; if `preserve_row_magnitudes`,
  `W_eff = normalize(W_eff) * W_row_norms`; `new_good = good_in @ W_effᵀ`,
  `new_bad = bad_in @ W_effᵀ`;
  `loss = w_p · mean((new_good − good_out)²) + w_s · (mean(knn(new_bad, good_out, k))
  − w_o · mean(knn(new_bad, bad_out, k)))`), but it never forms `W_eff`, which would
  cost `O(N d_in d_out)` per evaluation and gradient (about 90 GFLOP for a 4B model's
  `down_proj`) instead of `O((N + d_out) d_in r)` (about 7). It computes
  `new_out = (in @ W_baseᵀ + (in @ Aᵀ) @ Bᵀ) * s`, where, with
  `preserve_row_magnitudes`, `s_i = ‖w_i‖ / max(‖w_eff_i‖, 1e-12)` (F.normalize's
  clamp) with `‖w_eff_i‖² = ‖w_i‖² + B_i · (2 W_base Aᵀ + B (A Aᵀ))_i`, and `s = 1`
  otherwise. A guarded square root gives a zero rather than NaN gradient where the
  squared norm is not positive, as `linalg.normalize` does at a zero row. Every dot
  passes `Precision.HIGHEST`. The mathematics is upstream's; only the float32
  rounding changes (Divergence 12): in float64 the two forms agree to 1e-12 (loss)
  and 1e-10 (gradient). The expanded squared row norm loses accuracy only where a
  row of `W_eff` nearly cancels; on real trial adapters the largest
  `(‖w‖ + ‖v‖) / ‖w + v‖` observed was 1.49.
* `ara_optimise` returns `(A, B, state, loss)`, where `loss` is the value at the start
  of the step (what torch's `step` returns, used by `print_loss`).
* `modify_model`: for each layer in `[start_layer_index, end_layer_index)`, component
  and module: check on the host that `1 <= k <= min(N_good, N_bad)` (raise otherwise,
  as `torch.topk` would); move the module's I/O to the device as float32; call
  `ara_optimise` `n_optimization_steps` times, donating `A`, `B` and `state`; block on
  the result before starting the next module. `k_max_good = min(max(k, 15), N_good)`
  and `k_max_bad = min(max(k, 15), N_bad)`: identical to `min(15, N)` for the whole
  search space (k ≤ 15), while a hand-written k > 15 (which upstream accepts up to N)
  averages k neighbours, at the cost of a second compile. Results are collected per
  component; the component's stacks are assembled once from every module (modules
  outside the range keep the reset `A` and `B = 0`), so that their shapes, and the
  assembly's compiled program, are the same in every trial, and written with one
  `set_lora`. A component with no module in the range is not written (its adapters
  are already the reset values).
* The initial `A` and `B` of a module are the reset values (`A` re-initialised, `B`
  zero). They and the fresh `lbfgs.init` state are committed to `B`'s (replicated)
  sharding before the first step, because later steps receive committed arrays and
  JAX compiles separate programs for committed and uncommitted arguments.

### linalg.py and lbfgs.py

* `normalize(x, axis)`: `x / max(‖x‖₂, 1e-12)` (as `F.normalize`), with the norm
  computed so that its gradient at an all-zero row is zero rather than NaN, as
  PyTorch's is (otherwise a zero row of W would turn a whole L-BFGS run into NaN).
* `svd_lowrank(A, q, niter, key)`: port of `torch.svd_lowrank` with `M = None`: if
  `m < n`, operate on `Aᵀ`; `R = jax.random.normal(key, (n', q), float32)`;
  `Q = qr(A R)`; `niter` times: `Q = qr(Aᴴ Q)`, `Q = qr(A Q)`; `B = Qᴴ A`;
  `U, S, Vh = svd(B, full_matrices=False)`; `U = Q U`, `V = Vhᴴ`; swap U and V back if
  transposed. Random draws differ from PyTorch's.
* `knn_mean(a, b, k, k_max)`: `d2 = ‖a‖²[:, None] + ‖b‖²[None, :] − 2 a bᵀ`,
  `d = sqrt(maximum(d2, 1e-30))` (torch.cdist's matrix-multiply path, which it uses
  for more than 25 rows; the gradient is zero where the clamp applies);
  `vals = −lax.top_k(−d, k_max)[0]`;
  `mean = sum(where(arange(k_max) < k, vals, 0), −1) / k`. Value and gradient equal
  `cdist(a, b).topk(k, largest=False)[0].mean(1)`. Broadcast differences
  `a[:, None, :] − b[None, :, :]` are never formed.
* `lbfgs.py`: `init(n_params, history_size) -> LBFGSState` and
  `step(fun, x, state, data, *, lr, max_iter, history_size) -> (x, state, orig_loss)`.
  `step` is a plain traceable function and is never jitted on its own; it is called
  while tracing an enclosing jitted function (`ara_optimise`), and `lr`, `max_iter`
  and `history_size` are Python values there. `fun(x, data) -> loss` is any Python
  callable available at trace time: it may close over static Python values (for ARA,
  a `functools.partial` of the module-level objective with `preserve_row_magnitudes`,
  `k_max_good` and `k_max_bad`, built inside `ara_optimise` from its static
  arguments), but never over arrays. Every array, including `W_base`, its row norms,
  the base outputs, the module I/O, `loss_weights` and `k`, goes through the traced
  pytree `data` (`ObjectiveData`). It is a faithful port of `torch.optim.LBFGS.step`
  with `line_search_fn="strong_wolfe"`: `tolerance_grad = 1e-7`,
  `tolerance_change = 1e-9`,
  `max_eval = max_iter * 5 // 4`; history update only when `y·s > 1e-10`,
  `H_diag = y·s / y·y`, two-loop recursion over a ring buffer of `history_size`;
  first step length `min(1, 1/‖g‖₁) · lr` on the very first iteration, `lr` after;
  break when `g·d > −tolerance_change`; the strong-Wolfe search (cubic
  interpolation, `c1 = 1e-4`, `c2 = 0.9`) is called with
  `max_ls = max_eval − current_evals`, as torch does; the end-of-iteration checks
  (`max_iter`, `max_eval`, `‖g‖∞ <= tolerance_grad`, `‖d·t‖∞ <= tolerance_change`,
  `|loss − prev_loss| < tolerance_change`) are torch's. `LBFGSState` carries torch's
  state across the outer steps of a module: `func_evals`, the global iteration count
  `n_iter`, `d`, `t`, the history (`old_dirs`, `old_stps`, `ro`, as a ring buffer of
  `history_size` rows with `num_old` entries starting at row `head`), `H_diag`,
  `prev_flat_grad` and `prev_loss`. The flat parameter vector is
  `concat(A.ravel(), B.ravel())`. Everything is `lax.while_loop`/`lax.cond`; no host
  synchronisation inside a step.
* Tiled history: `old_dirs` and `old_stps` are
  `[history_size, *history_row_shape(n)] = [history_size, ceil(n / 1024) · 8, 128]`.
  Each row holds a vector of `n` parameters zero-padded to whole 8 × 128 TPU tiles
  (`TILE = 8 · LANES`, `LANES = 128`) and is read and written as a flat `[n]` vector
  (`_read_row`, `_write_row`); the padding is never read, so results are bit-identical
  to a `[history_size, n]` history. In that layout a row would be spread over one
  sublane of each of `n / 128` tiles, so every row access moved eight times its
  bytes, which made the optimiser's vector work about 2.5 times slower on TPU for a
  4B model.
* `step` computes in the dtype of `x` (float32 in production; the tests cast the state
  to float64 under `jax.enable_x64` to compare the algorithm with torch in float64),
  and raises `ValueError` when the state was initialised for another `history_size`
  or another number of parameters than `x` has.

## Scorers

`KeywordRate` is unchanged. `KLDivergence` computes
`sum(exp(t) * (t − x)) / N` with `x`, `t` the float32 log-softmax of the current and
baseline logits (`F.kl_div(..., reduction="batchmean", log_target=True)`).
`BenchmarkScore.init` builds one `JaxLM(model.tokenizer, model.lm_eval_state)` and
`get_score` calls `lm_eval.simple_evaluate(model=self.lm, tasks=[task])`; upstream's
per-call `hflm._model` reassignment is unnecessary because JaxLM resolves the model
at call time.

### lm-eval adapter (JaxLM)

`backend/lm_eval_adapter.JaxLM(tokenizer, state_fn: Callable[[], LMState])` subclasses
`lm_eval.api.model.TemplateLM`, calls `LM.__init__(self)`, and reproduces `HFLM` as
upstream constructs it (`HFLM(pretrained, tokenizer, batch_size="auto")`: no
`add_bos_token`, `max_length` or `truncation` override, no chat template). It may
import `Collator`, `resolve_max_length`, `handle_stop_sequences`,
`normalize_gen_kwargs`, `postprocess_generated_text`, `has_bos_prefix` and
`_add_special_kwargs` from `lm_eval.models.utils`, and `get_rolling_token_windows` and
`make_disjoint_window` from `lm_eval.utils`. `Collator` is used only for grouping,
ordering and `get_original`; `Collator.get_cache` (which needs torch tensors) is never
called.

**State.** JaxLM calls `state_fn()` at the start of every `loglikelihood`,
`loglikelihood_rolling` and `generate_until` call, and on every read of
`max_length`, and keeps no reference to an Engine, params or LoRA between calls.
Later `set_lora`, `reset_model` and `lora_disabled` changes are therefore seen without
rebuilding it, and a JaxLM built before `apply_lora` scores the adapted model
afterwards.

**Members.**

* `eot_token_id = tokenizer.eos_token_id`.
* `prefix_token_id = tokenizer.bos_token_id` if not None, else `eos_token_id`
  (overrides TemplateLM's EOS default).
* `max_length` is a property, evaluated on every access as HFLM's is:
  `resolve_max_length(state_fn().hf_config, tokenizer, default=2048)` (text config
  first, typically `max_position_embeddings`). After a slow-path `reset_model` it
  therefore reflects the evaluated checkpoint.
* `max_gen_toks = 256`.
* `tok_encode(s, add_special_tokens=None, left_truncate_len=None)`: HFLM's logic:
  explicit `add_special_tokens` wins; otherwise `add_special_tokens=False` when
  `has_bos_prefix(s, tokenizer.decode(prefix_token_id))`; otherwise tokenizer defaults
  (BOS is added for Gemma 3, Llama 3 and Mistral); then left truncation.
* `tok_decode(t, skip_special_tokens=True)`.
* `_encode_pair` and `loglikelihood` are inherited unchanged (empty contexts use
  `prefix_token_id`).

**`_loglikelihood_tokens`.** Group requests by key `tuple(ctx + cont[:-1])` (do not
use `Collator.get_cache`, which needs torch tensors). Each group's representative is
its member with the longest continuation; representatives are sorted by
`(−len(ctx + cont), tuple(ctx + cont))`. For a representative,
`inp = (ctx + cont)[−(max_length + 1):][:−1]`, which depends only on the key.
Representatives are grouped, in sorted order, by `(T bucket, C_score bucket)` and
batched as under [Engine shape policy](#engine-shape-policy) (cap 64, filler rows),
then scored with `Engine.score`, the candidates of a row being the last tokens of its
members' continuations. `ref_len` of every row of a JaxLM batch is the longest `inp`
among its real rows (HFLM pads its batch to the longest row and uses positions
`0…len−1`). For a member with continuation length K and candidate g:
`logprob = sum(next_lp[row, C_score−K : C_score−1]) + last_lp[row, g]` and
`is_greedy = all(next_greedy[row, C_score−K : C_score−1]) and last_greedy[row, g]`.
Results are returned in request order and `cache_hook.add_partial("loglikelihood",
...)` is called for each.

**`loglikelihood_rolling`.** Windows are
`map(make_disjoint_window, get_rolling_token_windows(tok_encode(s),
prefix_token=prefix_token_id, max_seq_len=max_length, context_len=1))`, scored with
`_loglikelihood_tokens` and summed per string.

**`generate_until`.**

1. `Collator([r.args for r in requests], sort_fn=lambda x: (−len(tok_encode(x[0])),
   x[0]), group_by="gen_kwargs", group_fn=lambda x: x[1])`, exactly as HFLM; each
   group is taken whole with `get_batched(n=0)` (longest context first), and results
   are restored with `get_original`.
2. Per group: `kwargs = normalize_gen_kwargs(gen_kwargs, max_gen_toks)`;
   `until = handle_stop_sequences(kwargs.pop("until"),
   eos=tokenizer.decode(eot_token_id, skip_special_tokens=False))`;
   `max_gen_toks = kwargs.pop("max_gen_toks")` (per task: 256 by default, 1024 for
   bbh, 2048 for mmlu_pro).
3. Encode each context with `add_special_tokens=False` when it starts with
   `tokenizer.bos_token`, otherwise with tokenizer defaults, and left-truncate to
   `max_length − max_gen_toks`. Group the encoded rows by T bucket and batch them as
   under [Engine shape policy](#engine-shape-policy) (cap 64, filler rows; ShapeKey
   `("generate", T, rank, max_new_tokens=max_gen_toks, C_chunk=32, sampling)`).
   `ref_len` of every row is the longest real row of its JaxLM batch (HFLM pads to the
   longest row of its batch), which gives `pad_penalised` and the Phi-3 factor choice.
4. `spec = decode_spec(state.generation_config, kwargs, max_new_tokens=max_gen_toks,
   chunk=32, pad_id=state.pad_id, key=state.next_key() if kwargs["do_sample"] else
   None)`, one call (and so one sampling key) per JaxLM batch. Greedy when the
   normalised `do_sample` is false (the generation config's EOS list and repetition
   penalty apply; temperature, top-k and top-p are ignored, as transformers does);
   sampling when it is true, with the request's kwargs merged over the resolved
   generation config.
5. `stop_check` decodes each unfinished real row's whole generated sequence with
   `skip_special_tokens=True` and returns true for rows containing any non-empty stop
   string. This only stops batches earlier; outputs are unchanged, because rows are
   independent and post-processing discards everything after the first stop string.
6. Decode each row's first `Generated.finish` tokens with `skip_special_tokens=True`,
   apply `postprocess_generated_text(s, until, None)`, and call
   `cache_hook.add_partial("generate_until", (context, gen_kwargs), s)`.

Log-probabilities are computed in float32 (HFLM uses the logits dtype), and their
sums over a continuation's tokens are accumulated in float64 (HFLM sums in float32).

Further rules:

* `G` (the candidate bucket) is computed once per `(T bucket, C_score bucket)` group
  of representatives, not per engine batch, so that the `ShapeKey` handed to
  `batch_limit` is fixed for the group; some batches therefore use a `G` one power of
  two larger than they need. A row's candidates are de-duplicated (members with the
  same last token share one).
* After an out-of-memory error, the failed batch and the rest of its group run at
  the lowered `B_eff`.
* A `max_length` key in a request's generation kwargs is removed with a warning and
  ignored: HFLM would read it as the padded batch length plus the generated tokens,
  which depends on its own batches.
* HFLM's assertions become exceptions: `ValueError` for an empty context or
  continuation, a continuation longer than `max_length` and `max_gen_toks ≥
  max_length`; `TypeError` for generation kwargs that are not a dict.
* JaxLM also has HFLM's `tokenizer_name` property; it has no `get_model_info` or
  `apply_chat_template`.
* Truncation warnings use HFLM's wording on the logger `lm_eval.models.heretic_tpu`,
  a child of lm-eval's logger, so the log level heretic-tpu sets for lm-eval applies
  to them.

## Export

The port's export never modifies the in-memory model, so `main.py` does not call
`reset_trial_model()` after a save or upload. Both exports run on the host with NumPy,
from the checkpoint files and host copies of the adapters:
`export.save_merged(directory, ckpt, tensors, arch, adapters, tokenizer)` and
`export.save_adapter(directory, ckpt, arch, adapters, *, base_model, model_commit)`.
Both first validate the adapters against `arch` (components, L, M = 1, d_out, d_in
and one shared rank) and raise `ValueError` before writing anything. Host copies are
made with `np.ascontiguousarray`: a host copy of a TPU array with a narrow minor
dimension can come back column-major, and safetensors writes the raw buffer whatever
its strides, which once scrambled the `lora_B` weights of adapter exports.

### Merged

* Only the shard set defined under [Weights](#weights) is processed. A shard that
  contains no abliterated tensor (no module with a non-zero `B`) is copied byte for
  byte (following Hugging Face cache symlinks). Other shards are rewritten: every
  tensor is read with `safe_open(framework="np")` in its storage dtype; each
  abliterated tensor is replaced by `(W.astype(f32) + B @ A).astype(storage_dtype)`,
  computed in NumPy float32 on the host from the on-disk `W` (not from the device);
  all other tensors, including vision tensors, embeddings and the head, are written
  unchanged; the file keeps its name and the source's `__metadata__` (for example
  `{"format": "pt"}`); it is written to `<name>.tmp` and then `os.replace`d.
* `model.safetensors.index.json` is copied unchanged when the source has one. An
  `lm_head` tensor absent from the source is never written.
* Every file the merged export writes, rewritten shards, byte copies, metadata and
  tokenizer files alike, is written to `<name>.tmp` and then `os.replace`d, so an
  interrupted export never leaves a truncated file, and a symlink in the target (for
  example into a Hugging Face cache) is replaced rather than written through.
* Export refuses, with `ValueError`, a target directory that is the source checkpoint
  directory or lies inside it (symlinks resolved first, then each ancestor compared
  with `os.path.samefile`), because the shards are read from the source while the
  target is written. Before writing, files in the target that match
  `model(-\d{5}-of-\d{5})?\.safetensors` and are not in the shard set are deleted, and
  so is a `model.safetensors.index.json` in the target when the source has no index,
  so that no stale index points at deleted shards. No other source file is copied; in
  particular `consolidated*`, `original/`, `params.json`, `*.pth`, `*.pt` and `*.bin`
  never are.
* Non-weight files: `config.json` and `generation_config.json` are copied verbatim.
  Whichever of `preprocessor_config.json`, `processor_config.json`,
  `video_preprocessor_config.json`, `chat_template.json`, `chat_template.jinja`,
  `tokenizer.json`, `tokenizer_config.json`, `tokenizer.model`,
  `special_tokens_map.json`, `added_tokens.json`, `vocab.json` and `merges.txt` exist
  at the pinned commit are fetched with `snapshot_download(revision=sha,
  allow_patterns=[...])` (from the cache only, `local_files_only=True`, when the
  checkpoint was resolved offline; read from the local directory for a local model)
  and copied; then `tokenizer.save_pretrained` runs over them. In transformers 5.17
  that always persists `pad_token`, and persists `padding_side` only when the source
  `tokenizer_config.json` already has the key, as upstream's
  `tokenizer.save_pretrained` does; the export does not force it, which would make
  exported tokenizers pad on the left by default.

### Adapter

* `adapter_config.json`: `peft_type = "LORA"`, `task_type = "CAUSAL_LM"`,
  `r = lora_alpha = rank`, `lora_dropout = 0.0`, `bias = "none"`,
  `target_modules` = the sorted `module_path(...)` of every layer and module of every
  component, `base_model_name_or_path = settings.model`,
  `revision = settings.model_commit` if set, else the resolved commit `sha` (null for
  a local directory; `save_adapter` applies `model_commit or ckpt.sha` itself),
  `inference_mode = true`, `fan_in_fan_out = false`, `init_lora_weights = true`,
  `use_rslora = false`, `use_dora = false`. It is written as PEFT writes it
  (`json.dumps(..., indent=2, sort_keys=True)`). Only the two adapter files are
  written: no tokenizer files (upstream saves none with an adapter either) and no
  PEFT README (Divergence 8).
* `adapter_model.safetensors`: keys `base_model.model.<module_path>.lora_A.weight`
  (`[r, d_in]`) and `.lora_B.weight` (`[d_out, r]`), float32, for every target module
  (`B = 0` where the trial did not modify it). For gemma3 this gives, for example,
  `base_model.model.model.language_model.layers.0.self_attn.o_proj.lora_A.weight`,
  exactly what upstream's `PeftModel.save_pretrained` produces.
* Disk keys are used only by the merged export. On MoE models only `attn.o_proj` is a
  target, with the dense paths; loading such adapters into transformers 5 needs
  `peft >= 0.20` (0.19.x fails for every Qwen3-MoE and Mixtral adapter).

### Upload

`push_to_hub` exports into a fresh temporary directory, creates the repository with
the requested visibility, and uploads that directory with
`HfApi.upload_folder`. Only files produced by the export are uploaded, so the
`.safetensors` hashes that `upload_reproduce_folder` records cover exactly the indexed
shards. The model card and the reproduce folder follow upstream's flow in `main.py`,
branded as described under
[System information and reproducibility](#system-information-and-reproducibility).

## Settings changes

* `dtypes` defaults to `["auto", "bfloat16", "float32"]`; `"auto"` and explicit
  `"float16"` entries resolve as described under `Model.__init__`.
* `quantization` only accepts `"none"`; `"bnb_4bit"` fails validation with a message
  explaining that bitsandbytes quantisation is unsupported on TPU.
* `device_map` and `max_memory` are removed. When present in a restored upstream
  settings object (study checkpoint, reproduce.json, config.toml) they are ignored
  with a warning.
* New `parallelism` (`"auto"`, `"single"` or `"tensor"`; default `"auto"`): see
  [Sharding](#sharding).
* New `compilation_cache_dir` (default `~/.cache/heretic-tpu/xla`, `exclude=True`):
  the persistent XLA compilation cache, set with
  `jax.config.update("jax_compilation_cache_dir", ...)` before the first compile; an
  empty string disables it.
* `max_shard_size` is accepted but ignored by the merged export.
* `batch_size`, `max_batch_size` and `offload_outputs_to_cpu` keep their meaning; see
  [Engine shape policy](#engine-shape-policy) and the
  [Model facade](#model-facade-heretic_tpumodelpy). The environment prefix stays
  `HERETIC_`.

## System information and reproducibility

* `system.py`: `empty_cache()` is `gc.collect()`. `configure_compilation_cache(dir)`
  sets `jax_compilation_cache_dir` (None for an empty string).
  `get_accelerator_info_dict()` returns, on TPU, `{"type": "TPU", "api_name":
  "libtpu", "api_version": <libtpu version or None>, "driver_version": None,
  "devices": [{"name": device_kind, "vram_gb": round(bytes_limit / 2**30, 2)}, ...]}`
  (rounded as upstream rounds); on another JAX accelerator the platform name
  upper-cased with `api_name` and `api_version` None and the same keys otherwise; on
  CPU `{"type": None}` (the keys upstream's `check_environment` reads).
  `get_accelerator_info()` prints the device count and type, total HBM, the libtpu
  version and each device's kind and HBM; it drops upstream's "Driver Version:
  Unknown" line, because JAX exposes no driver version. `print_memory_usage()` prints
  host RSS and the summed device `bytes_in_use` and `peak_bytes_in_use`.
* `get_heretic_version_info()` gives installations from other sources (archive URLs,
  other version-control systems) no origin `type`, which both programs' readers take
  as an unknown origin; upstream writes `"type": "unknown"`, which its own reader
  rejects. The port's `format_version_information` accepts `"unknown"` (as a version
  that cannot be told apart, like a local one), so `--reproduce` reads such files.
* `get_requirements_dict()` seeds `packages_to_check` with
  `["heretic-tpu", "jax", "jaxlib", "libtpu"]` (libtpu is reachable only through
  `extra ==` markers, which the walk skips). A package that is not installed is
  skipped, so CPU runs omit libtpu.
* Every distribution-name lookup uses `heretic-tpu`. `get_readme_intro` uses
  `version("heretic-tpu")` and states that the model was made with heretic-tpu, a TPU
  port of Heretic.
* The reproduce README lists the JAX, jaxlib and libtpu versions instead of PyTorch;
  its install step is `pip install -r requirements.txt` (which pins `jax==`,
  `jaxlib==` and `libtpu==` exactly); its commands are
  `heretic-tpu --reproduce reproduce.json` and `heretic-tpu`. Its
  heterogeneous-device warning is kept, triggered by differing device names in the
  accelerator dict and referring to tensor parallelism instead of `device_map`.
* The reproduce folder names the commit that was loaded. Upstream's
  `create_reproduce_folder` records `model_info(settings.model).sha`, the current head
  of the default branch, which differs from the loaded commit when `model_commit`
  pins another one or the repository has changed since, and it overwrites the live
  settings, so that adapters saved later in the session recorded that revision. The
  port's `create_reproduce_folder` and `upload_reproduce_folder` take
  `model_commit` (`main.py` passes `model.checkpoint.sha`; when it is None, the commit
  `settings.model_commit` or the default branch resolves to is looked up) and write
  the files from a copy of the settings with that commit.
* `reproduce.json` keeps format version 4 and stays readable by both programs. The
  port writes `environment.heretic` (heretic-tpu's version information),
  `environment.pytorch_version = null` (the key upstream's reader requires),
  `environment.backend = "jax"`, `environment.jax_version`,
  `environment.jaxlib_version`, `environment.libtpu_version` (null if absent) and
  `environment.requirements`; in `settings`, and in the reproduce folder's
  config.toml, built-in plugin names are written in upstream form (`heretic.<...>`,
  which the port resolves back), so that both programs can read both files.
* `check_environment` stores the local heretic-tpu version information under the key
  `heretic-tpu`. For a file with `environment.backend == "jax"` the original
  information goes under `heretic-tpu`; for an upstream file it goes under
  `heretic-llm` and `pytorch_version` is compared as `torch`, so those entries show up
  as mismatches. `get_package_mismatch_severity`: `heretic-tpu` and `heretic-llm`
  critical; `jax`, `jaxlib`, `libtpu`, `torch` and `transformers` high; the rest as
  upstream.

## Testing

`tests/backend/` runs on CPU in float32 against transformers + PyTorch (the `parity`
dependency group: `torch`, `peft>=0.20`, `accelerate`), with references loaded with
`attn_implementation="eager"` and, for MoE models, eager experts
(`set_experts_implementation("eager")`). Since transformers ignores Gemma 3's
`attn_logit_softcapping` on every attention path, references for soft-capped
configurations register transformers' own Gemma 3 `eager_attention_forward` with the
cap passed through as the attention implementation `"eager_softcap"`. Tiny
random-weight checkpoints (`tests/backend/tiny.py`) are built from the resolved
configs of real models at commits pinned in `tiny.REVISIONS` (every Hub model the
helpers read must be pinned there), shrunk and written with `save_pretrained`. Two
exceptions: tests marked `tpu` run only on a TPU (in the dtype they state; a
collection hook in `tests/conftest.py` skips them elsewhere, and the `device` fixture
runs a test on the CPU and, marked `tpu`, on the TPU), and the memory tests compile
`jax.ShapeDtypeStruct` inputs (bf16 where stated) for the default backend without
allocating them. Tests marked `slow` download models or run end to end; the default
selection is `-m "not slow"`.

The tests that build an `ArchConfig` from real checkpoints read the configs (cached by
huggingface_hub) and the safetensors headers (`tiny.real_tensor_index`: fetched once
with HTTP range requests, no weights) at the pinned commits. The tensor indexes are
cached under huggingface_hub's assets cache
(`$HF_ASSETS_CACHE/heretic-tpu-tests/<repo>/tensor-index/<sha>.json`, about 2.6 MB
for all models, written atomically), so after one run with Hub access the default
selection runs with `HF_HUB_OFFLINE=1`; without a cached index and without the Hub,
those tests fail with a `RuntimeError` saying so.

Tests that compare with upstream Heretic import it from a git-ignored checkout in
`heretic/` and skip when it is not present, except where
`HERETIC_TPU_REQUIRE_UPSTREAM=1`, which `scripts/tpu.sh` sets on the VM when the synced
working tree has that checkout: there such a skip is reported as a failure.

* **Forward parity**, per family: logits at all positions, hidden states, module I/O
  and greedy generations. Cases: llama (llama3 RoPE; tied, no `lm_head`; and a copy
  with biases), mistral (window shorter than the prompt), qwen2 (biases;
  `use_sliding_window` false), qwen3, gemma3_text (linear factor 8 on full layers
  only; window shorter than the prompt; `attn_logit_softcapping` set with amplified
  q/k weights so the cap is active, against `"eager_softcap"`), gemma3 and mistral3
  wrappers saved by transformers 5 (legacy `language_model.model.*` layout; one
  mistral3 with an untied head and no tie key), ministral3 in the mistral3 wrapper
  (yarn RoPE; the Llama 4 query scaling made active), qwen3_moe (per-expert and fused
  expert tensors) and mixtral,
  phi3 shaped like Phi-4-mini (partial rotary 0.75, longrope with
  `max_position_embeddings > original_max_position_embeddings`, tied) and like
  Phi-3-mini-4k (window shorter than the sequence); logits on both sides of the
  long-factor switch; generation across it compared with transformers'
  `use_cache=False`; a `*_batched` call whose upstream batches have longest rows on
  either side of the threshold, so rows sharing an engine batch use different factors,
  compared with transformers run per upstream batch. Synthetic cases for accepted
  paths no released family exercises: a llama config with `hidden_act = "gelu"`
  (exact erf GELU); a qwen3 config with `rope_parameters` `{rope_type: "yarn",
  factor: 4, original_max_position_embeddings: 32768}`; a llama config with
  `{rope_type: "dynamic", factor: 2}` and a small `max_position_embeddings`, whose
  logits match below the threshold, whose generation raises `NotImplementedError`
  at the step that crosses it, and which does not raise when every row reaches EOS
  first. Also: the LoRA path against PEFT; prefill plus decode steps against the full
  forward; blocked attention against eager; the KV cache is never copied between
  executables; tensor parallelism on a forced 4-device CPU mesh. MoE: the routed MLP
  in bf16 equals transformers' eager expert loop bit for bit on inputs where every
  matmul is exact and rank-order combination would differ (on the CPU only, see
  [MoE](#supported-architectures)); its grouped matmuls, lowered for the TPU from any
  backend, request DEFAULT precision for bf16 and HIGHEST for float32 operands.
  TPU-marked: the grouped matmul and the routed MLP compile in bf16 at the tiny and
  the real models' widths and match transformers within a few bf16 ulps where both
  select the same experts; tiny Qwen3-MoE and Mixtral checkpoints run in bf16; and
  (also `slow`) real models in bf16 against the float32 CPU reference.
* **Generation**: greedy against `generate` with `repetition_penalty = 1.3`,
  `pad_token_id` in `eos_token_id`, one batch with mixed prompt lengths whose longest
  prompt is not a bucket size; each row's tokens and response length match exactly.
  Sampling processors against transformers' warpers on fixed logits. A `stop_check`
  that marks a row finished yields `pad_id` afterwards and `finish` equal to the
  number of tokens generated when it was marked.
* **ArchConfig**: built for unsloth/gemma-3-4b-it (an ungated mirror of
  google/gemma-3-4b-it), unsloth/gemma-3-1b-it, Qwen/Qwen2.5-0.5B-Instruct,
  Qwen/Qwen3-0.6B, mistralai/Mistral-7B-Instruct-v0.3,
  mistralai/Mistral-Small-3.1-24B-Instruct-2503, microsoft/Phi-3-mini-4k-instruct,
  microsoft/Phi-3.5-mini-instruct, microsoft/Phi-4-mini-instruct and
  unsloth/Llama-3.2-1B-Instruct, plus a copy of each round-tripped through
  `save_pretrained` (rope_parameters-only format), plus the synthetic yarn (qwen3) and
  dynamic (llama) configs above; fields compared with
  transformers' own modules (attention `scaling` and `sliding_window`,
  `ROPE_INIT_FUNCTIONS` inv_freq and attention scaling per layer type, compared with
  `rope.build_tables`; head counts, `head_dim`), instantiating only rotary and
  attention modules. `hash(arch)` works and an `ArchConfig` built twice from the same
  checkpoint compares equal; Qwen3-0.6B (ships `lm_head.weight`) and a re-saved copy
  without it differ in `tied_head` and so compare unequal; a config.json with
  `"vision_config": null` gives `multimodal` true. Accepted: Qwen3-MoE and Mixtral
  configs and Ministral-3-3B-Instruct-2512-BF16. Rejections: gemma
  (unsloth/gemma-2b-it), gemma2 (unsloth/gemma-2-2b-it), the FP8
  Ministral-3-3B-Instruct-2512 and Qwen3-0.6B-FP8 (pre-quantised), a tiny checkpoint
  with `quantization_config` (rejected before any shard is downloaded or the prefix
  is detected; the quantisation check precedes every other), one with F8_E4M3 linear
  weights, Qwen3-MoE with dense layers, a supported text type inside an unsupported
  wrapper (llava), an unsupported RoPE type, `longrope` on some layer types only,
  bidirectional attention, unknown layer types, an unsupported activation, partial
  rotary outside Phi-3 and per-layer overrides. The tensor index is cached on disk
  and reused offline.
* **Components**: `get_abliterable_components()` and `get_module_count()` equal what
  upstream's `get_layer_modules` finds on the same tiny transformers model (MoE:
  `attn.o_proj` only).
* **Precision**: (1) every float32 entry point (adapter forward, abliteration kernel,
  `svd_lowrank`, `knn_mean`, ARA objective and gradient, `lbfgs.step`) is lowered with
  `jax.jit(f).lower(...).as_text()` under `jax.default_matmul_precision("default")`,
  and every `dot_general` with float32 operands must have
  `precision = [HIGHEST, HIGHEST]` (stricter than "not `[DEFAULT, DEFAULT]`", which
  `backend/__init__.py`'s global setting would satisfy even where an explicit
  precision is missing); (2) a float32 `jnp.dot` of random 512×512 matrices matches
  float64 NumPy within relative error 1e-5; the ARA objective changes by `grad·δ`
  (within 5%) for a perturbation of B whose effect on W is `1e-6·‖W‖`, measured with
  the preserve term only, exact float32 outputs and `B @ A` at 1% of W (the k-NN
  distances come from the Gram matrix, whose float32 cancellation near coincident
  points exceeds such a change); and the ARA loss and gradient (at a point with
  well-separated neighbours), `svd_lowrank` and the abliteration adapters match
  float64 NumPy references. These assertions run on the default backend and are
  collected twice, unmarked (CPU) and marked `tpu`; only the TPU run discriminates,
  because they fail on TPU at default precision; (3) TPU-marked and `slow`: ARA with
  `steer_bad_behavior_weight = 1e-3` on module I/O captured from
  HuggingFaceTB/SmolLM2-135M-Instruct in bf16, a value at which default precision
  makes far less progress (measured with upstream's `W_eff` form of the objective: 10
  of 60 modules move, median `‖B@A‖/‖W‖` 8e-7, against 56–58 of 60 at HIGHEST).
  With so small a weight the first
  L-BFGS step decides per module whether optimisation takes off, depending on
  rounding (in PyTorch too), so the TPU run is compared with the CPU float32 run in
  aggregate over all modules: on the CPU at least L modules reach
  `‖B@A‖/‖W‖ > 1e-3` and the summed float64 loss falls by more than its initial
  value, and the TPU's summed loss and number of such modules are within 25% of the
  CPU's; (4) the `tpu`-marked variant of `test_matches_torch_in_float32` in
  `test_lbfgs.py`: `lbfgs.py` against `torch.optim.LBFGS` executed on the TPU.
* **lbfgs.py** against `torch.optim.LBFGS` on CPU (same objective, several outer
  steps, history wrap-around, line search with `max_ls = max_eval − current_evals`),
  in float64 (state cast under `jax.enable_x64`) and float32; the tiled history rows
  occupy whole tiles, their padding stays zero, and the layout does not change
  results; a state initialised for another size raises; `svd_lowrank`
  reconstruction quality against `torch.svd_lowrank`; `knn_mean` value and gradient
  against `cdist(...).topk(...)`. ARA: `ara_optimise` against torch in float64 and
  `modify_model` in float32; the low-rank objective equals upstream's `W_eff` form in
  float64 (loss to 1e-12, gradient to 1e-10, including a zero row with a finite
  gradient), and neither it nor its gradient defines a `[d_out, d_in]` array; ARA
  compiles once for k = 1…15 (executable-cache size), later trials compile nothing,
  and its lowered HLO has no large constants.
* **Memory**: the abliteration function lowered with `jax.ShapeDtypeStruct` inputs of
  shape `[L, 1, 3584, 18944]` bf16 has `temp_size_in_bytes <= 4 × d_out × d_in × 4`
  (one module in flight) at L = 28, and grows by at most half a module from L = 4 to
  L = 28 (on TPU it grows by about 0.37 module, from the tile padding of the stacked
  rank-r adapters; a per-layer weight copy would add at least half a module per
  layer); its jaxpr contains a `cond` inside the map's `scan` and no `select_n` over
  `[d_out, d_in]` operands (so skipped modules do no work); and its adapters equal a
  per-layer reference loop. The decode function on a synthetic 16-layer
  configuration whose cache dwarfs its activations (deep enough that the one layer's
  cache slice a CPU decode step materialises is small), compiled at two `T_cache`
  values: with the cache allocated inside the function, Δ(`temp + output − alias`) ≤
  1.25 × Δcache bytes; for the chunk function with a donated cache, Δ`temp` ≤ 0.25 ×
  Δcache bytes and `alias` equals the cache size; the long-RoPE re-prefill rebuilds
  the cache in place; single-token generation builds no cache; scoring and capturing
  never hold logits for every column; `memory_need` matches the programs.
  `sum(a.nbytes for a in jax.live_arrays())` never exceeds about one parameter set
  during a slow-path `reset_model` or a forced smoke-test failure in the dtype loop.
  Loading on a forced 4-device CPU mesh with `parallelism = "tensor"` gives correct
  per-device shards and never creates per-layer device arrays.
* **Weights**: loaded parameters equal the transformers state dict per family
  (including fused Phi-3 and MoE layouts), also when the storage and session dtypes
  differ (float16 or float32 stored, bfloat16 loaded, and the reverse); local shard
  sets never glob, and stray files are never downloaded; a cached model loads with
  `HF_HUB_OFFLINE=1` and during a simulated Hub outage (connection error, timeout,
  5xx, 429, 408), other Hub errors are raised, and offline errors name what is not
  cached; the generation config without generation_config.json comes from
  config.json; a folder re-saved in the other layout loads the latest save, a stale
  index on the Hub is ignored, and an index with missing shards and no
  `model.safetensors` raises; missing or misshapen tensors raise before any data is
  read; the pre-flight rejects an oversized model (also on a real TPU); placement
  creates only full parameter arrays; module paths equal `named_modules()`.
* **Facade**: `batch_size == 0` compiles at exactly `B = n` and never calls
  `batch_limit` or `lower_limit`; a candidate whose `fits` check fails, and one that
  raises a simulated RESOURCE_EXHAUSTED error at run time, both reach the tuning loop
  as failures without any halving, and the loop propagates any other error; with
  `batch_size > 0`, a simulated run-time RESOURCE_EXHAUSTED re-runs the failed batch
  in batches of half the size with identical results (asserted as the `(B, T)` of
  every call, which differ between generation and captures); generation packs rows
  across buckets, longest first, only while a row's cache at most doubles, with
  responses equal to upstream's; adapters are reserved by `fits` only until they
  exist (and again after a slow-path reload); `get_residuals_mean` equals a float64
  NumPy reference and ignores filler rows; response strings with
  `skip_special_tokens=False` match upstream's trailing pads; a seed of `2**32`
  raises `ValueError`; Qwen3-MoE and Mixtral checkpoints load with
  `dtypes = ["bfloat16"]` and keep their router and expert parameters in bf16 (on the
  TPU this guards the ragged-dot precision rule, without which the smoke test failed
  and the model silently fell back to float32).
* **JaxLM** against `HFLM(pretrained=hf_model, tokenizer=tok)` on tiny Gemma 3,
  Llama 3 and Qwen2.5 checkpoints with synthetic `Instance`s: identical
  `(context_enc, continuation_enc)` and `inp` lists (empty contexts, contexts ending
  in spaces, over-length inputs); `loglikelihood` and `loglikelihood_rolling` values
  within tolerance and identical greedy flags; `generate_until` strings with `until`
  and `max_gen_toks`. State resolution: (a) a JaxLM built before `apply_lora` returns,
  after `set_lora` with non-zero B, the values of a JaxLM built afterwards; (b) inside
  `lora_disabled()` it returns the pre-LoRA values; (c) after `settings.model` is set
  to a perturbed local copy and `reset_model()` runs, it returns the values of a fresh
  facade on that copy (Engine reused); (d) the same after switching to a checkpoint
  with a different `ArchConfig` (another layer count and `max_position_embeddings`)
  and a different generation_config.json (another `eos_token_id` and
  `repetition_penalty`): a new Engine is built, `max_length` and `generate_until`
  follow the new checkpoint, and `save_adapter` records the new revision. Batches
  follow the shape policy, and sampling takes one key per batch. `slow`, needs
  network: `simple_evaluate` of gsm8k and piqa on HuggingFaceTB/SmolLM2-135M-Instruct
  with `limit=2` and `log_samples=True`, comparing per-sample responses with HFLM.
* **Export**: merged output loads with upstream's class choice
  (`AutoModelForImageTextToText` when the config has `vision_config`, else
  `AutoModelForCausalLM`) and `output_loading_info=True` with empty missing,
  unexpected and mismatched lists; every tensor outside the abliterated set
  (including vision tensors and the head) is byte-identical to the source; each
  abliterated tensor equals `(W.astype(f32) + B @ A).astype(dtype)`; logits match the
  JAX forward with adapters; untouched shards are byte copies; `__metadata__` is kept;
  stale shards are removed, and a single-file export into a directory holding an
  earlier sharded export leaves no index; the source directory is refused. Adapter output loads
  with `PeftModel.from_pretrained` on the same class and with `base.load_adapter`, with
  no missing or unexpected LoRA keys, loaded `lora_A`/`lora_B` equal to the exported
  tensors and logits matching the JAX forward (gemma3 and mistral3 legacy-layout
  fixtures included). The static `module_path` table equals `named_modules()` of the
  model instantiated under `torch.device("meta")`. A fixture snapshot containing
  `consolidated.safetensors`, `params.json` and `original/x.pth`: none is downloaded
  or exported, and the exported `.safetensors` set equals the `weight_map` values.
* **Reproducibility**: `get_readme_intro`, `generate_reproduce_readme` and
  `generate_reproduce_json` run on CPU with a synthetic trial, raise nothing and
  contain no `heretic-llm`, `torch==` or bare `heretic --reproduce`; with
  `importlib.metadata.distribution` patched to give jax a `libtpu; extra == "tpu"`
  requirement and an installed libtpu, libtpu appears in `get_requirements_dict()`;
  an upstream reproduce.json passes through the port's `check_environment` and
  settings validation, and upstream's reader accepts the port's; a port
  reproduce.json contains every key upstream's reader uses and upstream-form plugin
  names; the reproduce folder records the loaded (or pinned) commit; an installation
  of unknown origin is accepted; the dependency bounds exclude untested versions and
  the default benchmarks' dependencies are installed.
* `tests/e2e/` (`slow`): five non-interactive CLI runs on tiny Hub checkpoints at
  pinned commits, adapted from upstream's `tests/*/config.toml`, covering both
  modifiers, both export strategies and the qwen2.5, qwen3, llama, gemma3 and mistral3
  families. Each checks that the saved model loads in transformers (or PEFT) with only
  abliterable weights changed and that a reproduction from a reproduce.json made from
  the run's study saves byte-identical files, which `--reproduce` verifies against
  the recorded hashes; an adapter merged with PEFT must equal the merged export of the
  same trial. Adaptations: `tiny-random/llama-3.3-dim64` replaces
  `tiny-random/llama-3` (rejected by transformers 5.17's config validation) and
  `optimum-intel-internal-testing/tiny-random-mistral3` (a mistral3 wrapper around a
  mistral text model) replaces `tiny-random/mistral-3`; the ARA cases use 16 good and
  bad prompts instead of 5 (neighbour counts go up to 15, and both programs raise when
  k exceeds the prompt count) and seed 1, so that their 2-layer models get a non-empty
  layer range; the reproduction sets `HERETIC_IGNORE_MISMATCHES=true` (local installs
  always report a version mismatch, upstream's random suffix) and writes
  `save_directory` to config.toml (stored settings exclude paths). The CLI runs in
  the pytest process (`heretic_tpu.main.main()` with `sys.argv` and the working
  directory patched), because only one process can hold the TPU and the test process
  already does. The runs work on CPU and TPU.
* `scripts/tpu.sh` (configured by `TPU_NAME`, `TPU_ZONE`, `TPU_PROJECT` and
  `TPU_REMOTE_DIR`, the remote checkout relative to `$HOME`, one per concurrent user)
  runs everything on the TPU VM over IAP. `sync` copies the working tree (tracked and
  untracked non-ignored files, the git-ignored `heretic/` checkout without its `.git`
  directory when present, and the git-ignored `.scratch/` directory for throwaway
  scripts),
  tolerating tracked files deleted without staging; `run` syncs and runs a command,
  `exec` runs without syncing, `setup` installs uv and the shared virtual environment
  (`uv sync --extra tpu --group dev --group parity`). Remote commands run with that
  environment activated, `PYTHONPATH` at the synced `src/`, and
  `HERETIC_TPU_REQUIRE_UPSTREAM=1` when `heretic/` was synced. Only one process can use the TPU, so commands take
  an exclusive lock: `run` and `exec` give up with exit status 75 after
  `TPU_LOCK_WAIT` seconds (default 300) when another job holds it, and `TPU_LOCK=0`
  skips the lock for commands that do not touch the TPU (for example with
  `JAX_PLATFORMS=cpu`). `submit NAME CMD` syncs and starts a detached job, which waits
  for the lock and logs to `NAME.log` in the remote checkout (that checkout must not be
  synced again while it runs), and `logs NAME [LINES]` shows the log's tail and
  whether the job finished, with its exit status.

## Divergences from upstream

1. **Execution and support.** Models run on the JAX engine; only the architectures
   listed above are supported. Gemma (v1) and Gemma 2 are rejected (upstream fails on
   them too, because their chat templates reject the system role). Pre-quantised
   checkpoints are rejected (upstream loads them through transformers' quantisers).
   There is no bitsandbytes quantisation, no `device_map`/`max_memory` and no
   `trust_remote_code`. Qwen3-MoE checkpoints with dense layers are rejected, and so
   are configurations the scanned decoder cannot represent exactly, which transformers
   would run: per-layer overrides, `partial_rotary_factor != 1` outside Phi-3 (which
   transformers ignores for default RoPE), and `longrope` with a different
   `original_max_position_embeddings` per layer type. Dynamic RoPE beyond
   `max_position_embeddings` raises. A checkpoint with no head and
   `tie_word_embeddings` false raises (transformers would initialise a random head).
2. **Forward numerics.** The port always uses the eager recipe; upstream uses
   transformers' default SDPA attention and grouped-matmul experts, so results differ
   at rounding level (the port follows the eager per-expert loop). Gemma 3
   `attn_logit_softcapping` (null in released checkpoints) is applied when set, whereas
   transformers 5.17 ignores it on every attention path, eager included. For MoE
   router probabilities that tie exactly at the top-k boundary, the port selects the
   lower expert index; `torch.topk` on the CPU was observed to select the higher. On
   the TPU, XLA's default excess precision may skip some intermediate bf16 roundings
   of fused elementwise code. Phi-3 generation that crosses
   `original_max_position_embeddings` follows transformers' no-cache semantics;
   transformers 5.17's cached Phi-3 generate drops all context after the switch.
3. **dtypes.** float16 is not used: `"auto"` maps float16 checkpoints to bfloat16,
   float16 is removed from the default `dtypes`, and explicit `"float16"` entries
   (for example from restored upstream settings) become `"bfloat16"` with a warning.
   Any entry other than `auto`, `float16`, `bfloat16` and `float32` (for example
   `float64`, which upstream would load) counts as a failed attempt.
4. **Plugin-facing Model API.** `Context.get_logits`/`get_residuals` and the facade
   return NumPy or JAX arrays instead of PyTorch tensors. `get_layers()` returns
   layer indices instead of modules. `get_layer_modules`, `generate`, `model`,
   `processor`, `peft_config`, `get_merged_model` and `needs_reload` are removed
   (replaced by `get_module_count`, `get_base_weights`, `get_lora`/`set_lora`,
   `lora_rank` and the export methods). `ModuleIO` is
   `{component: (inputs [L, M, N, d_in], outputs [L, M, N, d_out])}` instead of a
   per-layer list of per-module tensor pairs. `get_lora`, `set_lora` and the export
   methods raise `RuntimeError` before `apply_lora`. When every dtype fails, the
   facade raises `RuntimeError` instead of a bare `Exception` (same message). In auto
   mode (`batch_size == 0`), the `*_batched` methods and `get_residuals_mean` treat
   the whole call as one batch, where upstream's `batchify(prompts, 0)` raises.
5. **Adapter reset.** LoRA `A` is re-initialised deterministically on every reset
   (upstream keeps the previous trial's `A`, which seeds ARA's next optimisation), so
   a trial restored after the study yields exactly the model that was evaluated.
6. **Randomised linear algebra.** `svd_lowrank`, L-BFGS, chat sampling and lm-eval
   sampling are re-implemented in JAX with the same algorithms; random draws differ
   from PyTorch, and sampling keys are derived per call from the seed (see PRNG keys)
   instead of from a global generator. `main.py` seeds Python's and NumPy's generators
   but not PyTorch's (`transformers.set_seed` would import it).
7. **Batching.** Captures group prompts by length bucket; generation packs shorter
   prompts into the batches of longer buckets, left-padded to the batch's bucket as
   long as a row's KV-cache length at most doubles; batches may be smaller than
   `batch_size` (`B_eff`). Per-row results depend on this only through the rounding
   of the bucket a row runs at, because the repetition-penalty pad rule, the Phi-3
   long-factor choice and response lengths are computed from upstream's batches
   (`ref_len`). Such rounding differences are common in long greedy generations: on
   ARA-modified Qwen3-4B, packing changed some token in about 60% of responses and
   flipped about 2 in 100 refusal classifications, against 43–50% and 0–1 for the
   per-bucket grouping at batch size 64 that was already accepted. Batch-size tuning
   treats only out-of-memory errors (and a failed headroom check) as "too large";
   upstream treats any exception above B = 1 that way.
8. **Export.** The merged export reads `W` from disk, merges in float32 and preserves
   the original shard layout, file names, storage dtypes, `__metadata__`,
   `config.json` and `generation_config.json` (upstream re-serialises the in-memory
   model and configs in the session dtype); `max_shard_size` is ignored; untouched
   shards are byte copies. `tokenizer.model`, `special_tokens_map.json`,
   `added_tokens.json` and the other listed tokenizer and processor files are copied
   from the source (upstream's `tokenizer.save_pretrained` drops some of them).
   Multimodal exports persist the tokenizer's `pad_token` (and `padding_side`, when
   the source tokenizer config sets one), which upstream's final
   `processor.save_pretrained` overwrites. `adapter_config.json` records the base
   `revision`, and no PEFT-generated README is written. `reset_trial_model()` is not
   called after an export. The merged export refuses, with `ValueError`, a target
   that is, or lies inside, the source checkpoint directory (upstream merges in memory
   and can save there); it deletes a stale single-file `model.safetensors` and a stale
   `model.safetensors.index.json` in the target (upstream's `save_pretrained` removes
   only stale numbered shards), and writes every file through `<name>.tmp` and
   `os.replace`.
9. **Benchmarks.** JaxLM replaces `HFLM`: log-softmax in float32 (HFLM uses the
   logits dtype) and log-probability sums in float64 (HFLM: float32); batch sizes
   come from the `B_eff` policy capped at 64 instead of HFLM's out-of-memory probing,
   so batch composition, and with it the repetition-penalty pad rule in
   `generate_until` and the Phi-3 long-factor choice, can differ from upstream's
   hardware-dependent batches; the BOS-prefix test in `generate_until` is applied per
   context (HFLM applies it to the first context of each batch, which only matters for
   tasks that mix BOS-prefixed and plain contexts). A `max_length` generation kwarg is
   ignored with a warning, and HFLM's assertions are `ValueError`s and `TypeError`s
   (see [JaxLM](#lm-eval-adapter-jaxlm)).
10. **Reproducibility.** Weight hashes from GPU runs cannot be matched on TPU.
    reproduce.json gains `backend`, `jax_version`, `jaxlib_version` and
    `libtpu_version`, writes `pytorch_version = null`, and records heretic-tpu's
    version under `environment.heretic`; requirements include jax, jaxlib and libtpu
    instead of torch; the model card and reproduce README name heretic-tpu. System
    information comes from JAX devices, which expose no driver version, so the
    accelerator report has no driver line. The reproduce folder records the commit
    that was loaded, where upstream records the current head of the default branch
    and overwrites the live settings with it (so upstream's later adapter exports in
    the session record that revision). An installation of unknown origin is written
    without an origin type, which both readers accept (upstream writes
    `"type": "unknown"`, which its own reader rejects, and which the port's
    `--reproduce` accepts).
11. **Checkpoint resolution.** Config, tokenizer and weights all come from the one
    commit `model_commit` resolves to, including the multimodal test (upstream's
    `get_model_class` reads config.json from the default branch whatever
    `model_commit` says). When the Hub cannot be reached, a cached model loads from
    the commit the cache records, as with upstream's transformers loading, but with a
    warning; HTTP 429 and 408 count as unreachable; and, because the port records the
    resolved branch in the cache, an unpinned offline run also works after the model
    was loaded only by heretic-tpu. A checkpoint with both a complete
    `model.safetensors.index.json` and a `model.safetensors` loads the indexed shards,
    where transformers loads `model.safetensors`; when the index names missing (or,
    offline, uncached) shards, both load `model.safetensors`, and the port warns (see
    Checkpoint resolution under [Weights](#weights)).
12. **ARA objective.** The objective is evaluated through the adapter's rank-r
    structure instead of forming `W_eff = W_base + B @ A` (see [ARA](#ara)). The
    mathematics is upstream's, but float32 rounding changes, and because ARA's float32
    L-BFGS trajectory is chaotic, trial outcomes change by as much as under any other
    float32 reassociation: on five real Qwen3-4B trials the low-rank form's score
    changes (mean KL 1.92, summed refusal rate 1.27) were the size of those of two
    mathematically identical reassociations of the dense form (1.86 and 1.48; 1.30 and
    1.25) against the dense form itself (1.98; 1.35), with no systematic shift.
