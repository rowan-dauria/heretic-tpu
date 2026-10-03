# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
The model facade: the Model API that plugins and `main.py` use, implemented on the
JAX engine (see "Model facade" and "Device memory lifetime" in docs/DESIGN.md).

The facade owns the parameter pytree and the LoRA adapters, formats prompts as
upstream does, splits calls into engine batches following the engine shape policy
and assembles the results in the original prompt order. Upstream's batches (the whole
call, or the chunks of `batchify(prompts, batch_size)` for the `*_batched` methods)
are only reproduced where results depend on them: through each row's `ref_len` (the
repetition-penalty pad rule and long-RoPE switching) and through response lengths.
"""

import gc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any, NamedTuple, TypeAlias, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedTokenizerBase,
    TextStreamer,
)

# Importing the backend first sets the float32 matmul precision before anything is
# traced (see "Matmul precision" in docs/DESIGN.md).
from .backend import export, weights
from .backend.arch import ArchConfig, check_config
from .backend.engine import (
    GENERATE_CHUNK,
    STREAM_CHUNK,
    DecodeSpec,
    Engine,
    ShapeKey,
    TokenBatch,
    bucket_length,
    decode_spec,
    is_out_of_memory,
    next_power_of_two,
    token_batch,
)
from .backend.sharding import ShardingPlan, choose_plan
from .backend.transformer import Lora
from .backend.weights import Checkpoint, Params, TensorIndex
from .config import ExportStrategy, Settings
from .utils import Prompt, batchify, format_exception, print

Array: TypeAlias = np.ndarray | jax.Array

# Component name -> (inputs [L, M, N, d_in], outputs [L, M, N, d_out]), model dtype.
ModuleIO: TypeAlias = dict[str, tuple[Array, Array]]

R = TypeVar("R")

# Maximum number of tokens generated for a chat response, as upstream.
CHAT_MAX_NEW_TOKENS = 4096

# PRNG key tags (see "PRNG keys" in docs/DESIGN.md). Tag 2 (abliteration's
# svd_lowrank) is derived by the modifier.
ADAPTER_KEY_TAG = 1
SAMPLING_KEY_TAG = 3

# Session dtypes that can be requested by name. float16 is not used on TPUs.
DTYPES = ("bfloat16", "float32")

# Safetensors storage dtypes that "auto" can resolve to, mapped to session dtypes.
STORAGE_DTYPES = {"BF16": "bfloat16", "F16": "bfloat16", "F32": "float32"}


class LMState(NamedTuple):
    """The current model state, as the lm-eval adapter (JaxLM) fetches it per request."""

    engine: Engine
    params: Params
    lora: Lora | None

    # Top-level resolved config of the current model.
    hf_config: PretrainedConfig

    # Resolved from the current model's checkpoint.
    generation_config: GenerationConfig

    # tokenizer.pad_token_id
    pad_id: int

    # The facade's sampling-key factory.
    next_key: Callable[[], jax.Array]


class _Rows(NamedTuple):
    """The tokenised prompts of a call, with what results depend on of upstream's batches."""

    tokens: list[list[int]]

    # Per row: the real token count of the longest prompt of the row's upstream batch.
    ref_len: list[int]

    # Upstream's batches, as lists of row indices.
    upstream_batches: list[list[int]]


@partial(jax.jit, static_argnames=("winsorise",))
def _float32_residuals(
    hidden: jax.Array,
    quantile: jax.Array,
    *,
    winsorise: bool,
) -> jax.Array:
    """
    Residuals [B, L + 1, D] in float32, with symmetric winsorisation of each prompt's
    layer as upstream applies it.
    """

    # Upcast the data type to avoid precision (bfloat16) problems during
    # calculations involving residual vectors.
    residuals = hidden.astype(jnp.float32)

    if winsorise:
        # The (prompt, layer, 1) quantiles of the (prompt, layer, component) residuals.
        thresholds = jnp.quantile(
            jnp.abs(residuals),
            quantile,
            axis=-1,
            keepdims=True,
            method="linear",
        )
        residuals = jnp.clip(residuals, -thresholds, thresholds)

    return residuals


def _place_float32(array: ArrayLike, sharding: jax.sharding.Sharding) -> jax.Array:
    if isinstance(array, jax.Array):
        array = array.astype(jnp.float32)
    else:
        array = np.asarray(array, dtype=np.float32)
    return jax.device_put(array, sharding)


class Model:
    settings: Settings
    tokenizer: PreTrainedTokenizerBase
    dtype: jnp.dtype
    revision_kwargs: dict[str, str]
    trusted_models: set[str]
    lora_rank: int | None

    # The loaded checkpoint (whose commit the export records), its tensor index and
    # architecture, and how its parameters are placed.
    checkpoint: Checkpoint
    tensors: TensorIndex
    arch: ArchConfig
    plan: ShardingPlan

    # The facade is the only long-lived owner of the device arrays
    # (see "Device memory lifetime" in docs/DESIGN.md).
    engine: Engine | None
    params: Params | None
    adapters: Lora | None

    def __init__(self, settings: Settings):
        self.settings = settings

        self.revision_kwargs = {}
        if settings.model_commit is not None:
            self.revision_kwargs["revision"] = settings.model_commit

        # Remote code is not supported.
        self.trusted_models = set()

        self.lora_rank = None
        self.engine = None
        self.params = None
        self.adapters = None

        # Nesting depth of lora_disabled().
        self._lora_disabled_depth = 0

        # Number of sampling keys handed out (see _next_key).
        self._sampling_calls = 0

        print()
        print(f"Loading model [bold]{settings.model}[/]...")

        checkpoint = weights.resolve_checkpoint(settings.model, settings.model_commit)

        # The tokenizer comes from the commit the weights are resolved to.
        if checkpoint.sha is None:
            tokenizer_kwargs = self.revision_kwargs
        else:
            tokenizer_kwargs = {"revision": checkpoint.sha}
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model,
            **tokenizer_kwargs,
        )

        # Fallback for tokenizers that don't declare a special pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CRITICAL: Always use left-padding for decoder-only models during generation.
        #           Right-padding causes empty outputs because the model sees PAD tokens
        #           after the prompt and thinks the sequence is complete.
        self.tokenizer.padding_side = "left"

        check_config(checkpoint.config)
        weights.fetch_shards(checkpoint)
        tensors = weights.build_tensor_index(checkpoint)
        arch = ArchConfig.from_hf(checkpoint.config, checkpoint.raw_config, tensors)

        self.checkpoint = checkpoint
        self.tensors = tensors
        self.arch = arch

        tried = set()

        for entry in settings.dtypes:
            print(f"* Trying dtype [bold]{entry}[/]...")

            loaded = None
            try:
                dtype = self._resolve_dtype(entry)
                if dtype in tried:
                    print(f"* Skipped ([bold]{dtype.name}[/] was tried already)")
                    continue
                tried.add(dtype)

                loaded = self._load_and_test(dtype)
            # Any failure (for example running out of memory, or numerical problems
            # in the smoke test) moves on to the next dtype, as upstream.
            except Exception as error:  # noqa: BLE001
                formatted = format_exception(error)
                if "\n" in formatted:
                    print(f"* [red]Failed:\n{formatted}[/]")
                else:
                    print(f"* [red]Failed ({formatted})[/]")

            if loaded is None:
                # The failed attempt built everything in local variables, which are
                # gone together with the exception at this point.
                self._release_device_state()
                continue

            self.dtype = dtype
            self.plan, self.params, self.engine = loaded
            break

        if self.params is None:
            raise RuntimeError("Failed to load model with all configured dtypes.")

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")
        print("* Abliterable components:")
        for component in self.get_abliterable_components():
            count = len(self.get_layers()) * self.get_module_count(component)
            print(f"  * [bold]{component}[/]: [bold]{count}[/] modules total")

    def _resolve_dtype(self, entry: str) -> np.dtype:
        """The session dtype of a `dtypes` entry."""

        if entry == "auto":
            # As transformers resolves "auto": the config's dtype (which it applies to
            # every sub-config), or else the first floating-point storage dtype
            # (excluding float8 and float4) of the first shard.
            config_dtype = getattr(self.checkpoint.config, "dtype", None)
            if config_dtype is not None:
                name = str(config_dtype).removeprefix("torch.")
            else:
                first_shard = self.checkpoint.shard_files[0]
                storage_dtype = next(
                    (
                        info.dtype
                        for info in self.tensors.tensors.values()
                        if info.shard == first_shard
                        and info.dtype.startswith(("F", "BF"))
                        and not info.dtype.startswith(("F8", "F4"))
                    ),
                    None,
                )
                if storage_dtype not in STORAGE_DTYPES:
                    raise ValueError(
                        f"Cannot load weights stored as {storage_dtype} in a "
                        "supported dtype."
                    )
                name = STORAGE_DTYPES[storage_dtype]
            # float16 is not used on TPUs.
            if name == "float16":
                name = "bfloat16"
        elif entry == "float16":
            print(
                "[yellow]Warning: float16 is not used on TPUs; "
                "loading in [bold]bfloat16[/] instead.[/]"
            )
            name = "bfloat16"
        else:
            name = entry

        if name not in DTYPES:
            raise ValueError(
                f"Unsupported dtype '{name}'. Supported dtypes are: "
                f"auto, float16 (loaded as bfloat16), {', '.join(DTYPES)}."
            )
        return jnp.dtype(name)

    def _load_and_test(self, dtype: np.dtype) -> tuple[ShardingPlan, Params, Engine]:
        """
        Loads the parameters in a dtype and runs a smoke test, building everything in
        local variables, so that nothing is left behind if either fails.
        """

        plan = choose_plan(self.arch, dtype, self.settings.parallelism)
        params = weights.load_params(
            self.checkpoint,
            self.tensors,
            self.arch,
            dtype,
            plan,
        )
        engine = self._new_engine(plan, dtype)

        # A test run can reveal dtype-related problems. As upstream, a single token is
        # generated greedily, which needs only the prefill.
        (tokens,) = self._encode(
            [Prompt(system=self.settings.system_prompt, user="What is 1+1?")]
        )
        spec = self._greedy_spec(max_new_tokens=1)
        engine.generate(
            params,
            None,
            token_batch([tokens], [len(tokens)], pad_id=spec.pad_id),
            spec,
        )

        return plan, params, engine

    def _new_engine(self, plan: ShardingPlan, dtype: np.dtype) -> Engine:
        return Engine(
            self.arch,
            plan,
            dtype,
            offload_outputs_to_cpu=self.settings.offload_outputs_to_cpu,
        )

    def _release_device_state(self, keep_engine: bool = False) -> None:
        """
        Frees the parameters and adapters on the device, and the engine's executables
        unless `keep_engine` is true.
        """

        leaves = [
            x
            for x in jax.tree.leaves((self.params, self.adapters))
            if isinstance(x, jax.Array)
        ]
        self.params = None
        self.adapters = None

        if self.engine is not None:
            # A kept engine reserves room for adapters again until they are applied.
            self.engine.adapters_allocated = False
            if not keep_engine:
                self.engine.release()
                self.engine = None

        # Tied leaves appear twice. Deleting the arrays makes stale references
        # elsewhere fail loudly instead of keeping the memory in use.
        for x in leaves:
            if not x.is_deleted():
                x.delete()

        del leaves
        gc.collect()

    # Structure

    def get_layers(self) -> list[int]:
        return list(range(self.arch.num_hidden_layers))

    def get_abliterable_components(self) -> list[str]:
        return weights.abliterable_components(self.arch)

    def get_module_count(self, component: str) -> int:
        self._check_component(component)
        return 1

    def get_base_weights(self, component: str) -> jax.Array:
        """The stored weights [L, M, d_out, d_in] of a component, in the model dtype."""

        self._check_component(component)
        return self.params["layers"][weights.COMPONENTS[component]]

    def _check_component(self, component: str) -> None:
        if component not in self.get_abliterable_components():
            raise ValueError(
                f"Unknown component '{component}'. The abliterable components of this "
                f"model are: {', '.join(self.get_abliterable_components())}."
            )

    # Adapters

    def apply_lora(self, lora_rank: int) -> None:
        """
        Allocates float32 LoRA adapters of the given rank for every component:
        A initialised as PEFT does, B zero, so the model is unchanged.
        """

        if lora_rank < 1:
            raise ValueError(f"The LoRA rank must be positive, but is {lora_rank}.")

        self.adapters = self._initial_adapters(lora_rank)
        self.lora_rank = lora_rank
        # The engine no longer needs to reserve room for them (see Engine.fits).
        self.engine.adapters_allocated = True

    def _initial_adapters(self, lora_rank: int) -> Lora:
        root = self._root_key()
        layer_count = self.arch.num_hidden_layers

        adapters = {}
        for index, component in enumerate(self.get_abliterable_components()):
            d_out, d_in = weights.component_shape(self.arch, component)
            A_sharding, B_sharding = self.plan.lora_shardings(component)

            # PEFT initialises A with kaiming_uniform_(a=sqrt(5)),
            # i.e. uniformly in ±1/sqrt(d_in).
            bound = 1 / np.sqrt(d_in)
            key = jax.random.fold_in(jax.random.fold_in(root, ADAPTER_KEY_TAG), index)
            A = jax.random.uniform(
                key,
                (layer_count, 1, lora_rank, d_in),
                jnp.float32,
                -bound,
                bound,
            )
            B = jnp.zeros(
                (layer_count, 1, d_out, lora_rank),
                jnp.float32,
                device=B_sharding,
            )
            adapters[component] = (jax.device_put(A, A_sharding), B)

        return adapters

    def get_lora(self, component: str) -> tuple[jax.Array, jax.Array]:
        """The adapters (A [L, M, r, d_in], B [L, M, d_out, r]) of a component."""

        self._check_component(component)
        return self._require_adapters()[component]

    def set_lora(self, component: str, A: ArrayLike, B: ArrayLike) -> None:
        old_A, old_B = self.get_lora(component)
        if np.shape(A) != old_A.shape or np.shape(B) != old_B.shape:
            raise ValueError(
                f"The adapters of {component} must have the shapes {old_A.shape} and "
                f"{old_B.shape}, but have the shapes {np.shape(A)} and {np.shape(B)}."
            )

        A_sharding, B_sharding = self.plan.lora_shardings(component)
        # A new dict, so that a state handed out earlier (lm_eval_state) is unchanged.
        self.adapters = {
            **self._require_adapters(),
            component: (_place_float32(A, A_sharding), _place_float32(B, B_sharding)),
        }

    def _require_adapters(self) -> Lora:
        if self.adapters is None:
            raise RuntimeError("The model has no adapters. Call apply_lora() first.")
        return self.adapters

    def reset_model(self) -> bool:
        """
        Resets the model to a clean state for the next trial or evaluation.

        Behaviour:
        - Fast path: If the same model is loaded, resets the LoRA adapters
          (B = 0, A re-initialised).
        - Slow path: If the model to use has changed, loads it in the session dtype.
          The adapters are dropped and have to be applied again.

        Returns True if the fast path was taken.
        """

        if self.settings.model == self.checkpoint.model:
            if self.lora_rank is not None:
                self.adapters = self._initial_adapters(self.lora_rank)
            return True

        checkpoint = weights.resolve_checkpoint(
            self.settings.model,
            self.revision_kwargs.get("revision"),
        )
        check_config(checkpoint.config)
        weights.fetch_shards(checkpoint)
        tensors = weights.build_tensor_index(checkpoint)
        arch = ArchConfig.from_hf(checkpoint.config, checkpoint.raw_config, tensors)
        plan = choose_plan(arch, self.dtype, self.settings.parallelism)

        # Executables depend only on the architecture, the plan and the dtype.
        self._release_device_state(keep_engine=(arch, plan) == (self.arch, self.plan))

        params = weights.load_params(checkpoint, tensors, arch, self.dtype, plan)

        self.checkpoint = checkpoint
        self.tensors = tensors
        self.arch = arch
        self.plan = plan
        if self.engine is None:
            self.engine = self._new_engine(plan, self.dtype)
        self.params = params
        self.lora_rank = None
        self.adapters = None

        return False

    @contextmanager
    def lora_disabled(self) -> Iterator[None]:
        """Runs every forward pass inside the block without adapters (nestable)."""

        self._lora_disabled_depth += 1
        try:
            yield
        finally:
            self._lora_disabled_depth -= 1

    def _active_lora(self) -> Lora | None:
        if self._lora_disabled_depth > 0:
            return None
        return self.adapters

    # PRNG keys

    def _root_key(self) -> jax.Array:
        seed = self.settings.seed

        # Without x64, jax.random.key silently truncates seeds outside the uint32
        # range, and upstream's transformers.set_seed rejects them.
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int | np.integer)
            or not 0 <= seed < 2**32
        ):
            raise ValueError(
                f"The seed must be an integer between 0 and {2**32 - 1}, "
                f"but is {seed!r}."
            )

        return jax.random.key(int(seed))

    def _next_key(self) -> jax.Array:
        """The key of the next sampling call (chat response or lm-eval batch)."""

        key = jax.random.fold_in(
            jax.random.fold_in(self._root_key(), SAMPLING_KEY_TAG),
            self._sampling_calls,
        )
        self._sampling_calls += 1
        return key

    # Prompts and batches

    def _encode(self, prompts: list[Prompt]) -> list[list[int]]:
        """The token ids of the templated prompts, as upstream builds them."""

        if not prompts:
            raise ValueError("prompts must not be empty")

        chats = [
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
            for prompt in prompts
        ]

        # This cast is valid because list[str] is the return type
        # for batched operation with tokenize=False.
        chat_prompts = cast(
            list[str],
            self.tokenizer.apply_chat_template(
                chats,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

        if self.settings.response_prefix:
            # Append the common response prefix to the prompts so that evaluation happens
            # at the point where responses start to differ for different prompts.
            chat_prompts = [
                prompt + self.settings.response_prefix for prompt in chat_prompts
            ]

        return self.tokenizer(chat_prompts, return_token_type_ids=False)["input_ids"]

    def _rows(self, prompts: list[Prompt], batched: bool) -> _Rows:
        """
        Tokenises the prompts of a call. Upstream's batch is the whole call, or for the
        `*_batched` methods each chunk of `batchify(prompts, batch_size)`. In auto mode
        (batch size 0, before tuning has fixed it) the whole call is one batch.
        """

        tokens = self._encode(prompts)

        indices = list(range(len(tokens)))
        batch_size = self.settings.batch_size
        if batched and batch_size > 0:
            upstream_batches = batchify(indices, batch_size)
        else:
            upstream_batches = [indices]

        ref_len = [0] * len(tokens)
        for upstream_batch in upstream_batches:
            longest = max(len(tokens[index]) for index in upstream_batch)
            for index in upstream_batch:
                ref_len[index] = longest

        return _Rows(tokens, ref_len, upstream_batches)

    def _run(
        self,
        rows: _Rows,
        key: Callable[[int], ShapeKey],
        run: Callable[[TokenBatch], R],
    ) -> list[tuple[np.ndarray, R]]:
        """
        Runs `run` on engine batches covering every row, following the engine shape
        policy, and returns each batch's result with the indices of its real rows.
        `key(T)` is the shape key of the batches of prompt bucket T.

        In auto mode the call is one batch, and out-of-memory errors propagate.
        Otherwise rows are grouped by bucket (in their original order) and split into
        batches of at most B_eff rows, padded with filler rows; after an out-of-memory
        error, the rows of the failed batch are run again from the start in batches
        of the lowered B_eff.
        """

        pad_id = self.tokenizer.pad_token_id

        if self.settings.batch_size == 0:
            batch = token_batch(rows.tokens, rows.ref_len, pad_id=pad_id)
            return [(np.arange(len(rows.tokens)), run(batch))]

        groups: dict[int, list[int]] = {}
        for index, tokens in enumerate(rows.tokens):
            groups.setdefault(bucket_length(len(tokens)), []).append(index)

        results = []

        for length, indices in sorted(groups.items()):
            shape_key = key(length)
            limit = self.engine.batch_limit(
                shape_key,
                min(self.settings.batch_size, next_power_of_two(len(indices))),
            )

            start = 0
            while start < len(indices):
                chunk = indices[start : start + limit]
                # Smaller batches are padded only to the next power of two,
                # so that few batch sizes are compiled.
                batch_size = min(limit, next_power_of_two(len(chunk)))
                batch = token_batch(
                    [rows.tokens[index] for index in chunk],
                    [rows.ref_len[index] for index in chunk],
                    pad_id=pad_id,
                    batch_size=batch_size,
                    length=length,
                )

                try:
                    result = run(batch)
                except Exception as error:
                    if not is_out_of_memory(error):
                        raise
                    limit = self.engine.lower_limit(shape_key, batch_size)
                    continue

                results.append((np.array(chunk), result))
                start += len(chunk)

        return results

    def _in_order(self, results: list[tuple[np.ndarray, Any]], on_host: bool) -> Any:
        """
        Concatenates per-batch results (pytrees of arrays over each batch's real rows)
        into one pytree over all rows in their original order, in NumPy on the host
        or on the device.
        """

        order = np.concatenate([indices for indices, _ in results])
        parts = [result for _, result in results]

        if on_host:
            combined = jax.tree.map(lambda *xs: np.concatenate(xs), *parts)
        elif len(parts) == 1:
            combined = parts[0]
        else:
            combined = jax.tree.map(lambda *xs: jnp.concatenate(xs), *parts)

        if np.array_equal(order, np.arange(len(order))):
            return combined

        inverse = np.argsort(order)
        return jax.tree.map(lambda x: x[inverse], combined)

    def _real_rows(self, result: Any, batch: TokenBatch, on_host: bool) -> Any:
        """
        The real rows of a batch's device results, fetched to the host (and the device
        buffers freed at once) or kept on the device.
        """

        if batch.n_real < len(batch.tokens):
            real = jax.tree.map(lambda x: x[: batch.n_real], result)
        else:
            real = result

        if not on_host:
            # Out-of-memory errors at run time surface here, inside the batch.
            return jax.block_until_ready(real)

        host = jax.device_get(real)
        for x in jax.tree.leaves((result, real)):
            if not x.is_deleted():
                x.delete()
        return host

    # Inference

    def _greedy_spec(self, max_new_tokens: int) -> DecodeSpec:
        return decode_spec(
            self.checkpoint.generation_config,
            # Use greedy decoding to ensure deterministic outputs.
            {"do_sample": False},
            max_new_tokens=max_new_tokens,
            chunk=GENERATE_CHUNK,
            pad_id=self.tokenizer.pad_token_id,
            key=None,
        )

    def _responses(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool,
        batched: bool,
    ) -> list[str]:
        rows = self._rows(prompts, batched)
        spec = self._greedy_spec(self.settings.max_response_length)
        lora = self._active_lora()

        def key(length: int) -> ShapeKey:
            return ShapeKey(
                "generate",
                length,
                self.lora_rank if lora is not None else None,
                max_new_tokens=spec.max_new_tokens,
                C_chunk=spec.C_chunk,
                sampling=spec.sampling,
            )

        def run(batch: TokenBatch) -> tuple[np.ndarray, np.ndarray]:
            generated = self.engine.generate(self.params, lora, batch, spec)
            return generated.tokens, generated.finish

        tokens, finish = self._in_order(self._run(rows, key, run), on_host=True)

        # Upstream decodes each batch up to the step at which its last row finished,
        # so rows that finished earlier end with padding.
        sequences = []
        for upstream_batch in rows.upstream_batches:
            length = max(finish[index] for index in upstream_batch)
            for index in upstream_batch:
                sequences.append(tokens[index, :length].tolist())

        return self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=skip_special_tokens,
        )

    def get_responses(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        return self._responses(prompts, skip_special_tokens, batched=False)

    def get_responses_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        return self._responses(prompts, skip_special_tokens, batched=True)

    def _capture(
        self,
        prompts: list[Prompt],
        want: str,
        batched: bool,
        on_host: bool,
        winsorization_quantile: float = 1.0,
    ) -> Any:
        """
        One quantity captured at the last prompt position ("logits", "hidden" as
        float32 residuals or "module_io"), with a leading axis over the prompts.
        """

        rows = self._rows(prompts, batched)
        lora = self._active_lora()
        rank = self.lora_rank if lora is not None else None
        winsorise = 0 <= winsorization_quantile < 1

        def run(batch: TokenBatch) -> Any:
            captures = self.engine.capture(self.params, lora, batch, frozenset({want}))
            result = getattr(captures, want)
            if want == "hidden":
                result = _float32_residuals(
                    result,
                    np.float32(winsorization_quantile),
                    winsorise=winsorise,
                )
            return self._real_rows(result, batch, on_host)

        results = self._run(
            rows,
            lambda length: ShapeKey("capture", length, rank, want=frozenset({want})),
            run,
        )
        return self._in_order(results, on_host)

    def get_logits(self, prompts: list[Prompt]) -> Array:
        """Raw logits [N, V] float32 at the last prompt position."""

        return self._capture(
            prompts,
            "logits",
            batched=False,
            on_host=self.settings.offload_outputs_to_cpu,
        )

    def get_logits_batched(self, prompts: list[Prompt]) -> Array:
        return self._capture(
            prompts,
            "logits",
            batched=True,
            on_host=self.settings.offload_outputs_to_cpu,
        )

    def get_residuals(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> Array:
        """
        Residuals [N, L + 1, D] float32 at the last prompt position (the hidden states
        in the transformers convention), winsorised if `winsorization_quantile` is in
        [0, 1).
        """

        return self._capture(
            prompts,
            "hidden",
            batched=False,
            on_host=self.settings.offload_outputs_to_cpu,
            winsorization_quantile=winsorization_quantile,
        )

    def get_residuals_batched(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> Array:
        return self._capture(
            prompts,
            "hidden",
            batched=True,
            on_host=self.settings.offload_outputs_to_cpu,
            winsorization_quantile=winsorization_quantile,
        )

    def get_residuals_mean(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> np.ndarray:
        """The mean [L + 1, D] float32 of the residuals of the prompts."""

        if not prompts:
            raise ValueError("prompts must not be empty")

        batch_size = self.settings.batch_size
        batches = batchify(prompts, batch_size) if batch_size > 0 else [prompts]

        running_sum = None
        total_count = 0

        for batch in batches:
            residuals = self._capture(
                batch,
                "hidden",
                batched=False,
                on_host=True,
                winsorization_quantile=winsorization_quantile,
            )

            # Accumulate in high precision on the host.
            batch_sum = np.sum(residuals, axis=0, dtype=np.float64)
            if running_sum is None:
                running_sum = batch_sum
            else:
                running_sum += batch_sum

            total_count += len(residuals)

        return (running_sum / total_count).astype(np.float32)

    def _module_io(self, prompts: list[Prompt], batched: bool) -> ModuleIO:
        on_host = self.settings.offload_outputs_to_cpu
        module_io = self._capture(prompts, "module_io", batched, on_host)

        # [N, L, M, d] -> [L, M, N, d]
        if on_host:
            return jax.tree.map(
                lambda x: np.ascontiguousarray(np.transpose(x, (1, 2, 0, 3))),
                module_io,
            )
        return jax.tree.map(lambda x: jnp.transpose(x, (1, 2, 0, 3)), module_io)

    def get_module_io(self, prompts: list[Prompt]) -> ModuleIO:
        """
        The inputs [L, M, N, d_in] and outputs [L, M, N, d_out] of every abliterable
        module at the last prompt position, in the model dtype.
        """

        return self._module_io(prompts, batched=False)

    def get_module_io_batched(self, prompts: list[Prompt]) -> ModuleIO:
        return self._module_io(prompts, batched=True)

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        # This cast is valid because str is the return type
        # for single-chat operation with tokenize=False.
        chat_prompt = cast(
            str,
            self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

        tokens = self.tokenizer(chat_prompt, return_token_type_ids=False)["input_ids"]
        pad_id = self.tokenizer.pad_token_id

        # The checkpoint's generation parameters, sampling if they say so.
        spec = decode_spec(
            self.checkpoint.generation_config,
            {},
            max_new_tokens=CHAT_MAX_NEW_TOKENS,
            chunk=STREAM_CHUNK,
            pad_id=pad_id,
            key=self._next_key(),
        )

        streamer = TextStreamer(
            # The TextStreamer constructor annotates this parameter with the AutoTokenizer
            # type, which makes no sense because AutoTokenizer is a factory class,
            # not a base class that tokenizers inherit from.
            self.tokenizer,  # ty:ignore[invalid-argument-type]
            # Only the generated tokens are passed to the streamer.
            skip_prompt=False,
            skip_special_tokens=True,
        )

        generated = []
        for token in self.engine.stream(
            self.params,
            self._active_lora(),
            token_batch([tokens], [len(tokens)], pad_id=pad_id),
            spec,
        ):
            streamer.put(token)
            generated.append(int(token[0]))
        streamer.end()

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    # Benchmarks

    def lm_eval_state(self) -> LMState:
        """The current state for the lm-eval adapter, to be fetched per request."""

        return LMState(
            engine=self.engine,
            params=self.params,
            lora=self._active_lora(),
            hf_config=self.checkpoint.config,
            generation_config=self.checkpoint.generation_config,
            pad_id=self.tokenizer.pad_token_id,
            next_key=self._next_key,
        )

    # Export

    def save_merged(self, directory: str) -> None:
        """Saves the model with the adapters merged in, with its tokenizer."""

        export.save_merged(
            directory,
            self.checkpoint,
            self.tensors,
            self.arch,
            self._require_adapters(),
            self.tokenizer,
        )

    def save_adapter(self, directory: str) -> None:
        """Saves the adapters as a PEFT LoRA adapter."""

        export.save_adapter(
            directory,
            self.checkpoint,
            self.arch,
            self._require_adapters(),
            base_model=self.settings.model,
            model_commit=self.settings.model_commit,
        )

    def push_to_hub(
        self,
        repo_id: str,
        *,
        private: bool,
        token: str,
        strategy: ExportStrategy,
    ) -> None:
        """Uploads the merged model or the adapter to a Hugging Face Hub repository."""

        # Fail before anything is created.
        self._require_adapters()

        if strategy == ExportStrategy.ADAPTER:
            export_function = self.save_adapter
        else:
            export_function = self.save_merged

        export.push_to_hub(repo_id, export_function, private=private, token=token)
