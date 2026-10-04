# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
The model facade on tiny random-weight checkpoints with the Qwen2.5 tokenizer and chat
template, compared with upstream heretic's Model methods running on transformers
(float32 on CPU, eager attention).

`Upstream` is upstream's Model code (generate, responses, logits, residuals, module
I/O hooks and chat) bound to a transformers model, so every facade result is compared
with what upstream computes for the same prompts and the same batches. The prompts
fall into two length buckets, and the batch sizes used make the facade regroup rows
across upstream batches and add filler rows.

The main checkpoint's generation config stops at a token that the longest prompt's
response produces early, which is also the tokenizer's padding token (as for Qwen2.5,
whose padding token is one of its EOS tokens): responses of rows that finish early end
with padding, and the repetition penalty applies to it exactly for rows that upstream
left-pads.
"""

import json
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from lm_eval.api.instance import Instance
from transformers import AutoTokenizer, TextStreamer

from heretic_tpu import model as model_module
from heretic_tpu.backend import engine as engine_module
from heretic_tpu.backend import export, weights
from heretic_tpu.backend.errors import DeviceMemoryError
from heretic_tpu.backend.lm_eval_adapter import JaxLM
from heretic_tpu.config import ExportStrategy, Settings
from heretic_tpu.model import Model, is_out_of_memory
from heretic_tpu.utils import Prompt, batchify
from tests.backend.tiny import (
    REAL_MODELS,
    FakeHub,
    build_checkpoint,
    load_reference,
    tiny_config,
)

torch = pytest.importorskip("torch")

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = "You are a helpful assistant."

MAX_RESPONSE_LENGTH = 10

PENALTY = 1.3

# Qwen2.5's <|im_end|>, the first EOS token of every checkpoint here.
IM_END = 151645

# With the chat template, short prompts fall into the bucket of 32 tokens
# and long ones into the bucket of 64. The first long prompt is the longest.
SHORT = ["What is 1+1?", "Hi", "Name three colours.", "Is water wet?"]
LONG = [
    (
        "Write a long and detailed story about a dragon who lives in a cave near the "
        "sea and loves to read old books about sailors and their ships in the stormy "
        "northern seas."
    ),
    (
        "Explain, in great detail and with many examples, how the immune system of "
        "the human body defends itself against viruses and bacteria."
    ),
    (
        "Describe the history of the Roman Empire from its founding to its fall, "
        "including the most important emperors, wars and reforms of that time."
    ),
]

# Short and long prompts alternate.
PROMPTS = [
    Prompt(system=SYSTEM_PROMPT, user=user)
    for user in [SHORT[0], LONG[0], SHORT[1], LONG[1], SHORT[2], LONG[2], SHORT[3]]
]

# With a batch size of 4, the upstream batches are rows 0-3 and 4-6, and the facade
# runs the four short rows in one batch and the three long rows in a batch of four
# with a filler row.
BATCH_SIZE = 4

# Every checkpoint here has the vocabulary of the tokenizer.
VOCAB_SIZE = 151936

# The Phi-3 case switches to the long RoPE factors above this sequence length,
# which lies between the lengths of the short and the long prompts.
PHI3_SWITCH = 40


class Upstream:
    """
    Upstream heretic's Model methods (heretic/src/heretic/model.py at the pinned
    commit) on a transformers model, with the tokenizer prepared as upstream does.
    """

    def __init__(self, model: Any, tokenizer: Any, settings: Settings):
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings

    def get_layers(self) -> Any:
        from peft import PeftModel

        model = self.model
        if isinstance(model, PeftModel):
            model = model.base_model.model
        with suppress(Exception):
            return model.model.language_model.layers
        return model.model.layers

    def get_layer_modules(self, layer_index: int) -> dict[str, list[Any]]:
        layer = self.get_layers()[layer_index]
        return {
            "attn.o_proj": [layer.self_attn.o_proj],
            "mlp.down_proj": [layer.mlp.down_proj],
        }

    def generate(self, prompts: list[Prompt], **kwargs: Any) -> tuple[Any, Any]:
        chats = [
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
            for prompt in prompts
        ]
        chat_prompts = cast(
            list[str],
            self.tokenizer.apply_chat_template(
                chats,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )
        if self.settings.response_prefix:
            chat_prompts = [
                prompt + self.settings.response_prefix for prompt in chat_prompts
            ]
        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **kwargs,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
            )
        return inputs, outputs

    def get_responses(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=self.settings.max_response_length,
        )
        return self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=skip_special_tokens,
        )

    def get_responses_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        return [
            response
            for batch in batchify(prompts, self.settings.batch_size)
            for response in self.get_responses(batch, skip_special_tokens)
        ]

    def get_logits(self, prompts: list[Prompt]) -> np.ndarray:
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_logits=True,
            return_dict_in_generate=True,
            use_cache=False,
        )
        return outputs.logits[0].numpy()

    def get_logits_batched(self, prompts: list[Prompt]) -> np.ndarray:
        return np.concatenate(
            [
                self.get_logits(batch)
                for batch in batchify(prompts, self.settings.batch_size)
            ]
        )

    def get_residuals(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> np.ndarray:
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[0]
        residuals = torch.stack(
            [layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states],
            dim=1,
        ).to(torch.float32)
        if 0 <= winsorization_quantile < 1:
            abs_residuals = torch.abs(residuals)
            thresholds = torch.quantile(
                abs_residuals,
                winsorization_quantile,
                dim=2,
                keepdim=True,
            )
            residuals = torch.clamp(residuals, -thresholds, thresholds)
        return residuals.numpy()

    def get_residuals_batched(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> np.ndarray:
        return np.concatenate(
            [
                self.get_residuals(batch, winsorization_quantile)
                for batch in batchify(prompts, self.settings.batch_size)
            ]
        )

    def get_module_io(self, prompts: list[Prompt]) -> dict[str, tuple[Any, Any]]:
        """Upstream's hooks, converted to the facade's [L, M, N, d] layout."""

        module_io: list[dict[str, dict[int, tuple[Any, Any]]]] = []

        def get_hook(layer_index: int, component: str, module_index: int) -> Any:
            def hook(module: Any, inputs: Any, outputs: Any) -> None:
                if len(module_io) == layer_index:
                    module_io.append({})
                module_io[layer_index].setdefault(component, {})[module_index] = (
                    inputs[0][:, -1, :].detach().clone(),
                    outputs[:, -1, :].detach().clone(),
                )

            return hook

        handles = []
        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                for module_index, module in enumerate(modules):
                    handles.append(
                        module.register_forward_hook(
                            get_hook(layer_index, component, module_index)
                        )
                    )
        self.generate(prompts, max_new_tokens=1)
        for handle in handles:
            handle.remove()

        return {
            component: tuple(
                np.stack(
                    [
                        np.stack([layer[component][0][side].numpy()])
                        for layer in module_io
                    ]
                )
                for side in (0, 1)
            )
            for component in module_io[0]
        }

    def stream_chat_response(
        self,
        chat: list[dict[str, str]],
        max_new_tokens: int,
    ) -> str:
        chat_prompt = cast(
            str,
            self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )
        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
            )
        return self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )


def make_settings(model: str | Path, **fields: Any) -> Settings:
    # model_construct skips the settings sources (command line, environment and
    # config.toml), which would otherwise pick up pytest's arguments.
    return Settings.model_construct(
        **{
            "model": str(model),
            "seed": 0,
            "dtypes": ["auto"],
            "batch_size": 0,
            "max_response_length": MAX_RESPONSE_LENGTH,
            "system_prompt": SYSTEM_PROMPT,
            **fields,
        }
    )


def upstream_tokenizer(directory: str | Path) -> Any:
    """The tokenizer as upstream prepares it."""

    tokenizer = AutoTokenizer.from_pretrained(directory)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def write_generation_config(directory: Path, **fields: Any) -> None:
    (directory / "generation_config.json").write_text(json.dumps(fields))


def build(
    directory: Path,
    family: str,
    *,
    seed: int = 0,
    generation_config: dict[str, Any],
    pad_token: str | None = None,
    **text_overrides: Any,
) -> Path:
    """A tiny checkpoint with the vocabulary, tokenizer and chat template of Qwen2.5."""

    config = tiny_config(REAL_MODELS[family], vocab_size=VOCAB_SIZE, **text_overrides)
    build_checkpoint(config, directory, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    if pad_token is not None:
        tokenizer.pad_token = pad_token
    tokenizer.save_pretrained(directory)

    write_generation_config(directory, **generation_config)
    return directory


def stop_token_candidates(directory: Path) -> list[int]:
    """
    Tokens that the response to the longest prompt produces early, and that do not
    occur in the response to some other prompt, best first (those that occur in the
    response to a shorter prompt too, then by position). Making such a token the
    padding token, and an EOS token, changes nothing before it is generated in a row
    that upstream does not pad.
    """

    tokenizer = upstream_tokenizer(directory)
    upstream = Upstream(load_reference(directory), tokenizer, make_settings(directory))
    inputs, outputs = upstream.generate(
        PROMPTS,
        max_new_tokens=MAX_RESPONSE_LENGTH,
        repetition_penalty=PENALTY,
        eos_token_id=[IM_END],
    )
    generated = outputs[:, inputs["input_ids"].shape[1] :].tolist()
    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    longest = int(np.argmax(prompt_lengths))
    assert prompt_lengths.count(prompt_lengths[longest]) == 1

    candidates = []
    for position, token in enumerate(generated[longest]):
        if (
            position == 0
            or token in generated[longest][:position]
            # Ids beyond the tokenizer's vocabulary have no token.
            or token >= len(tokenizer)
            or token in tokenizer.all_special_ids
        ):
            continue
        absent = any(token not in row for row in generated)
        shared = any(token in row for i, row in enumerate(generated) if i != longest)
        if absent:
            candidates.append((not shared, position, token))
    return [token for _, _, token in sorted(candidates)]


def set_pad_token(directory: Path, candidates: list[int]) -> int:
    """
    Saves the tokenizer with the first candidate as its padding token that leaves the
    tokenisation of the prompts unchanged (a padding token becomes an added token,
    which is matched in the raw text).
    """

    texts = [
        AutoTokenizer.from_pretrained(TOKENIZER).apply_chat_template(
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in PROMPTS
    ]
    expected = AutoTokenizer.from_pretrained(TOKENIZER)(texts)["input_ids"]

    for token in candidates:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(token)
        if tokenizer(texts)["input_ids"] == expected:
            tokenizer.save_pretrained(directory)
            return token

    raise AssertionError("no suitable stop token")


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory) -> dict[str, Path]:
    """
    "qwen2": the main checkpoint; "qwen2-copy": the same architecture with other
    weights; "llama": another architecture (two layers, other generation parameters);
    "phi3": Phi-4-mini's architecture with long RoPE factors above PHI3_SWITCH tokens.
    """

    root = tmp_path_factory.mktemp("facade")

    main = build(
        root / "qwen2",
        "qwen2",
        generation_config={"eos_token_id": IM_END, "repetition_penalty": PENALTY},
    )
    stop_token = set_pad_token(main, stop_token_candidates(main))
    generation_config = {
        "eos_token_id": [IM_END, stop_token],
        "pad_token_id": stop_token,
        "repetition_penalty": PENALTY,
    }
    write_generation_config(main, **generation_config)

    copy = build(
        root / "qwen2-copy",
        "qwen2",
        seed=1,
        generation_config=generation_config,
        pad_token=AutoTokenizer.from_pretrained(main).pad_token,
    )

    llama = build(
        root / "llama",
        "llama",
        generation_config={"eos_token_id": IM_END, "repetition_penalty": 1.1},
        num_hidden_layers=2,
    )

    phi3_rope = dict(
        tiny_config(REAL_MODELS["phi3"]).get_text_config().rope_parameters,
        original_max_position_embeddings=PHI3_SWITCH,
    )
    phi3 = build(
        root / "phi3",
        "phi3",
        generation_config={"eos_token_id": IM_END},
        rope_parameters=phi3_rope,
        max_position_embeddings=4 * PHI3_SWITCH,
        original_max_position_embeddings=PHI3_SWITCH,
    )

    return {"qwen2": main, "qwen2-copy": copy, "llama": llama, "phi3": phi3}


@pytest.fixture(scope="module")
def model(checkpoints) -> Model:
    """A facade on the main checkpoint, without adapters, shared by read-only tests."""

    return Model(make_settings(checkpoints["qwen2"]))


@pytest.fixture
def settings(model, monkeypatch) -> Settings:
    """The shared facade's settings, restored after each test."""

    for name in ("batch_size", "offload_outputs_to_cpu"):
        monkeypatch.setattr(model.settings, name, getattr(model.settings, name))
    return model.settings


@pytest.fixture(scope="module")
def references(checkpoints) -> Callable[[str], Any]:
    """The transformers model of a checkpoint, loaded on first use."""

    cache = {}

    def get(name: str) -> Any:
        if name not in cache:
            cache[name] = load_reference(checkpoints[name])
        return cache[name]

    return get


def upstream_for(model: Model, references, name: str = "qwen2") -> Upstream:
    return Upstream(references(name), model.tokenizer, model.settings)


def lengths(model: Model, prompts: list[Prompt]) -> list[int]:
    return [len(tokens) for tokens in model._encode(prompts)]


def live_bytes() -> int:
    return sum(array.nbytes for array in jax.live_arrays())


def params_bytes(model: Model) -> int:
    return sum(leaf.nbytes for leaf in jax.tree.leaves(model.params))


def random_adapters(model: Model, seed: int = 1) -> dict[str, tuple[Any, Any]]:
    """Adapters of the current rank whose B is not zero."""

    rng = np.random.default_rng(seed)
    adapters = {}
    for component in model.get_abliterable_components():
        A, B = model.get_lora(component)
        adapters[component] = (
            rng.standard_normal(A.shape).astype(np.float32) / A.shape[-1] ** 0.5,
            rng.standard_normal(B.shape).astype(np.float32) / B.shape[-1] ** 0.5,
        )
    return adapters


def peft_reference(model: Model, reference: Any) -> Any:
    """The reference model wrapped in PEFT with the facade's current adapters."""

    peft = pytest.importorskip("peft")

    arch = model.arch
    paths = {
        (component, layer_index): weights.module_path(arch, component, layer_index, 0)
        for component in model.get_abliterable_components()
        for layer_index in model.get_layers()
    }
    peft_model = peft.get_peft_model(
        reference,
        peft.LoraConfig(
            r=model.lora_rank,
            lora_alpha=model.lora_rank,
            lora_dropout=0.0,
            bias="none",
            target_modules=sorted(paths.values()),
            task_type="CAUSAL_LM",
        ),
    )
    base = peft_model.base_model.model
    with torch.no_grad():
        for (component, layer_index), path in paths.items():
            A, B = (np.array(x) for x in model.get_lora(component))
            module = base.get_submodule(path)
            module.lora_A["default"].weight.copy_(torch.from_numpy(A[layer_index, 0]))
            module.lora_B["default"].weight.copy_(torch.from_numpy(B[layer_index, 0]))
    return peft_model


@contextmanager
def recording(target: Any, name: str, calls: list) -> Iterator[None]:
    """Records the arguments of every call of a method, which still runs."""

    original = getattr(target, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return original(*args, **kwargs)

    setattr(target, name, wrapper)
    try:
        yield
    finally:
        delattr(target, name)


def assert_module_io_close(actual: dict, expected: dict, **kwargs: Any) -> None:
    assert sorted(actual) == sorted(expected)
    for component, pair in expected.items():
        for side, expected_array in enumerate(pair):
            np.testing.assert_allclose(
                np.asarray(actual[component][side]),
                expected_array,
                **{"rtol": 1e-4, "atol": 1e-4, **kwargs},
                err_msg=f"{component} {('inputs', 'outputs')[side]}",
            )


# The prompts and checkpoints.


def test_prompts_cover_two_buckets(model) -> None:
    prompt_lengths = lengths(model, PROMPTS)
    assert all(length <= 32 for length in prompt_lengths[0::2])
    assert all(32 < length <= 64 for length in prompt_lengths[1::2])


def test_structure(model) -> None:
    assert model.get_layers() == [0, 1, 2]
    assert model.get_abliterable_components() == ["attn.o_proj", "mlp.down_proj"]
    assert model.get_module_count("attn.o_proj") == 1
    assert model.get_base_weights("attn.o_proj") is model.params["layers"]["o_proj"]
    assert model.get_base_weights("mlp.down_proj").shape == (3, 1, 64, 128)
    assert model.trusted_models == set()
    assert model.revision_kwargs == {}
    assert model.tokenizer.padding_side == "left"
    assert model.dtype == jnp.float32

    with pytest.raises(ValueError, match="Unknown component"):
        model.get_base_weights("mlp.experts")
    with pytest.raises(RuntimeError, match="apply_lora"):
        model.get_lora("attn.o_proj")


# Inference against upstream.


@pytest.mark.parametrize("batch_size", [0, BATCH_SIZE])
def test_responses_match_upstream(model, settings, references, batch_size) -> None:
    settings.batch_size = batch_size
    upstream = upstream_for(model, references)

    for skip_special_tokens in (False, True):
        expected = upstream.get_responses(PROMPTS, skip_special_tokens)
        assert model.get_responses(PROMPTS, skip_special_tokens) == expected

    # Some response stops early, so it ends with padding.
    inputs, outputs = upstream.generate(PROMPTS, max_new_tokens=MAX_RESPONSE_LENGTH)
    generated = outputs[:, inputs["input_ids"].shape[1] :]
    pad_id = model.tokenizer.pad_token_id
    assert torch.any(torch.all(generated[:, -2:] == pad_id, dim=1))


@pytest.mark.parametrize("batch_size", [2, BATCH_SIZE])
def test_batched_responses_match_upstream(
    model,
    settings,
    references,
    batch_size,
) -> None:
    settings.batch_size = batch_size
    upstream = upstream_for(model, references)

    for skip_special_tokens in (False, True):
        expected = upstream.get_responses_batched(PROMPTS, skip_special_tokens)
        assert model.get_responses_batched(PROMPTS, skip_special_tokens) == expected


def test_logits_match_upstream(model, settings, references) -> None:
    upstream = upstream_for(model, references)

    logits = model.get_logits(PROMPTS)
    assert isinstance(logits, np.ndarray)
    assert logits.dtype == np.float32
    np.testing.assert_allclose(
        logits, upstream.get_logits(PROMPTS), rtol=1e-4, atol=1e-4
    )

    settings.batch_size = BATCH_SIZE
    np.testing.assert_allclose(
        model.get_logits_batched(PROMPTS),
        upstream.get_logits_batched(PROMPTS),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("winsorization_quantile", [1.0, 0.9])
def test_residuals_match_upstream(
    model,
    settings,
    references,
    winsorization_quantile,
) -> None:
    upstream = upstream_for(model, references)

    residuals = model.get_residuals(PROMPTS, winsorization_quantile)
    assert residuals.shape == (len(PROMPTS), 4, 64)
    np.testing.assert_allclose(
        residuals,
        upstream.get_residuals(PROMPTS, winsorization_quantile),
        rtol=1e-4,
        atol=1e-5,
    )

    settings.batch_size = BATCH_SIZE
    np.testing.assert_allclose(
        model.get_residuals_batched(PROMPTS, winsorization_quantile),
        upstream.get_residuals_batched(PROMPTS, winsorization_quantile),
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize("batch_size", [0, 3, 8])
def test_residuals_mean_matches_float64_reference(
    model,
    settings,
    references,
    batch_size,
) -> None:
    settings.batch_size = batch_size
    upstream = upstream_for(model, references)
    quantile = 0.9

    expected = np.sum(
        upstream.get_residuals(PROMPTS, quantile).astype(np.float64), axis=0
    ) / len(PROMPTS)

    batches = []
    with recording(model.engine, "capture", batches):
        mean = model.get_residuals_mean(PROMPTS, quantile)

    assert mean.dtype == np.float32
    assert mean.shape == (4, 64)
    np.testing.assert_allclose(mean, expected, rtol=1e-4, atol=1e-5)

    # With one upstream batch of all prompts, the three long ones run in a batch of
    # four, whose filler row is ignored.
    if batch_size == 8:
        assert any(batch.n_real < len(batch.tokens) for _, _, batch, _ in batches)


def test_module_io_matches_hooks(model, settings, references) -> None:
    upstream = upstream_for(model, references)

    module_io = model.get_module_io(PROMPTS)
    for inputs, outputs in module_io.values():
        assert isinstance(inputs, np.ndarray)
        assert inputs.flags.c_contiguous
        assert inputs.shape[:3] == outputs.shape[:3] == (3, 1, len(PROMPTS))
    assert_module_io_close(module_io, upstream.get_module_io(PROMPTS))

    # Upstream's batches don't change module I/O (there is no RoPE switching).
    settings.batch_size = BATCH_SIZE
    assert_module_io_close(
        model.get_module_io_batched(PROMPTS),
        upstream.get_module_io(PROMPTS),
    )


def test_outputs_can_stay_on_the_device(model, settings) -> None:
    expected_logits = model.get_logits(PROMPTS)
    expected_residuals = model.get_residuals(PROMPTS, 0.9)
    expected_module_io = model.get_module_io(PROMPTS)

    settings.offload_outputs_to_cpu = False
    for batch_size in (0, BATCH_SIZE):
        settings.batch_size = batch_size

        logits = model.get_logits_batched(PROMPTS)
        residuals = model.get_residuals_batched(PROMPTS, 0.9)
        module_io = model.get_module_io_batched(PROMPTS)
        for array in [logits, residuals, *jax.tree.leaves(module_io)]:
            assert isinstance(array, jax.Array)

        np.testing.assert_allclose(logits, expected_logits, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(residuals, expected_residuals, rtol=1e-6, atol=1e-6)
        assert_module_io_close(module_io, expected_module_io, rtol=1e-6, atol=1e-6)

        # The mean is always accumulated on the host.
        assert isinstance(model.get_residuals_mean(PROMPTS), np.ndarray)


def test_batched_rope_factor_follows_upstream_batches(checkpoints, references) -> None:
    """
    Phi-3 uses the long RoPE factors when its upstream batch is longer than
    PHI3_SWITCH tokens. Short rows of different upstream batches share engine batches
    but use the factors of their own upstream batch.
    """

    model = Model(make_settings(checkpoints["phi3"], batch_size=2))
    upstream = upstream_for(model, references, "phi3")

    prompt_lengths = lengths(model, PROMPTS)
    assert prompt_lengths[0] < PHI3_SWITCH < prompt_lengths[1]
    assert prompt_lengths[-1] < PHI3_SWITCH

    np.testing.assert_allclose(
        model.get_logits_batched(PROMPTS),
        upstream.get_logits_batched(PROMPTS),
        rtol=1e-4,
        atol=1e-4,
    )

    # The factors differ, so a whole-call reference with the long factors for every
    # row differs from the batched result for the last row (alone in its batch).
    whole_call = upstream.get_logits(PROMPTS)
    assert not np.allclose(
        model.get_logits_batched(PROMPTS)[-1], whole_call[-1], atol=1e-4
    )


def test_stream_chat_response(model, references, monkeypatch, capsys) -> None:
    monkeypatch.setattr(model_module, "CHAT_MAX_NEW_TOKENS", 8)
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Tell me a joke."},
    ]

    capsys.readouterr()
    expected = upstream_for(model, references).stream_chat_response(chat, 8)
    expected_output = capsys.readouterr().out

    calls = model._sampling_calls
    assert model.stream_chat_response(chat) == expected
    assert capsys.readouterr().out == expected_output

    # Every chat response takes a sampling key.
    assert model._sampling_calls == calls + 1


# Adapters.


def test_adapters_are_initialised_like_peft(checkpoints) -> None:
    model = Model(make_settings(checkpoints["qwen2"], seed=7))
    model.apply_lora(3)
    assert model.lora_rank == 3

    root = jax.random.key(7)
    for index, component in enumerate(model.get_abliterable_components()):
        A, B = model.get_lora(component)
        d_out, d_in = weights.component_shape(model.arch, component)
        assert A.shape == (3, 1, 3, d_in)
        assert B.shape == (3, 1, d_out, 3)
        assert A.dtype == B.dtype == jnp.float32
        assert (A.sharding, B.sharding) == model.plan.lora_shardings(component)

        expected_A = jax.random.uniform(
            jax.random.fold_in(jax.random.fold_in(root, 1), index),
            A.shape,
            jnp.float32,
            -1 / np.sqrt(d_in),
            1 / np.sqrt(d_in),
        )
        np.testing.assert_array_equal(A, expected_A)
        assert np.abs(np.asarray(A)).max() <= 1 / np.sqrt(d_in)
        assert not np.any(np.asarray(B))

    # Zero B leaves the model unchanged.
    with model.lora_disabled():
        base = model.get_logits(PROMPTS)
    np.testing.assert_allclose(model.get_logits(PROMPTS), base, rtol=1e-6, atol=1e-6)


def test_adapters_match_peft(checkpoints, references) -> None:
    model = Model(make_settings(checkpoints["qwen2"]))
    base_logits = model.get_logits(PROMPTS)
    base_module_io = model.get_module_io(PROMPTS)

    model.apply_lora(2)
    for component, (A, B) in random_adapters(model).items():
        model.set_lora(component, A, B)

    upstream = Upstream(
        peft_reference(model, load_reference(checkpoints["qwen2"])),
        model.tokenizer,
        model.settings,
    )
    logits = model.get_logits(PROMPTS)
    np.testing.assert_allclose(
        logits, upstream.get_logits(PROMPTS), rtol=1e-4, atol=1e-4
    )
    assert not np.allclose(logits, base_logits, atol=1e-3)

    # Module outputs include the adapters' deltas, as upstream's hooks on the PEFT
    # modules see them.
    assert_module_io_close(
        model.get_module_io(PROMPTS),
        upstream.get_module_io(PROMPTS),
    )
    assert model.get_responses(PROMPTS) == upstream.get_responses(PROMPTS)

    # lora_disabled() nests, and the adapters are back after it.
    with model.lora_disabled():
        with model.lora_disabled():
            np.testing.assert_array_equal(model.get_logits(PROMPTS), base_logits)
            assert model.lm_eval_state().lora is None
        np.testing.assert_array_equal(model.get_logits(PROMPTS), base_logits)
        assert_module_io_close(model.get_module_io(PROMPTS), base_module_io)
    np.testing.assert_array_equal(model.get_logits(PROMPTS), logits)
    assert model.lm_eval_state().lora is model.adapters


def test_set_lora(checkpoints) -> None:
    model = Model(make_settings(checkpoints["qwen2"]))
    model.apply_lora(2)
    A, B = model.get_lora("attn.o_proj")

    # Any array type, cast to float32 and placed with the plan's shardings.
    new_A = np.ones(A.shape, dtype=np.float64)
    new_B = jnp.full(B.shape, 2, dtype=jnp.bfloat16)
    state = model.lm_eval_state()
    model.set_lora("attn.o_proj", new_A, new_B)

    A, B = model.get_lora("attn.o_proj")
    assert A.dtype == B.dtype == jnp.float32
    assert (A.sharding, B.sharding) == model.plan.lora_shardings("attn.o_proj")
    np.testing.assert_array_equal(A, 1)
    np.testing.assert_array_equal(B, 2)

    # A state fetched earlier is unchanged.
    assert not np.any(np.asarray(state.lora["attn.o_proj"][1]))

    with pytest.raises(ValueError, match="shapes"):
        model.set_lora("attn.o_proj", new_A[:, :, :1], new_B)
    with pytest.raises(ValueError, match="Unknown component"):
        model.set_lora("mlp.experts", new_A, new_B)


def test_reset_model_fast_path(checkpoints) -> None:
    model = Model(make_settings(checkpoints["qwen2"]))

    # Without adapters there is nothing to reset.
    assert model.reset_model()
    assert model.adapters is None

    model.apply_lora(2)
    initial = {
        component: model.get_lora(component)
        for component in model.get_abliterable_components()
    }
    base_logits = model.get_logits(PROMPTS)
    for component, (A, B) in random_adapters(model).items():
        model.set_lora(component, A, B)

    engine = model.engine
    assert model.reset_model()
    assert model.engine is engine
    assert model.lora_rank == 2

    for component, (initial_A, initial_B) in initial.items():
        A, B = model.get_lora(component)
        np.testing.assert_array_equal(A, initial_A)
        np.testing.assert_array_equal(B, initial_B)
    np.testing.assert_array_equal(model.get_logits(PROMPTS), base_logits)


def test_reset_model_slow_path_keeps_the_engine(
    checkpoints,
    references,
    monkeypatch,
) -> None:
    model = Model(make_settings(checkpoints["qwen2"]))
    model.apply_lora(2)
    engine = model.engine

    # Before reading the new checkpoint, the old parameters and adapters are freed.
    model_bytes = params_bytes(model) + sum(
        leaf.nbytes for leaf in jax.tree.leaves(model.adapters)
    )
    before = live_bytes()
    loads = []
    load_params = weights.load_params

    def recording_load(*args: Any) -> Any:
        loads.append(live_bytes())
        return load_params(*args)

    monkeypatch.setattr(weights, "load_params", recording_load)

    model.settings.model = str(checkpoints["qwen2-copy"])
    assert not model.reset_model()

    assert loads[0] <= before - model_bytes + 2**20
    assert model.engine is engine
    assert model.lora_rank is None
    assert model.adapters is None
    assert model.checkpoint.snapshot_dir == str(checkpoints["qwen2-copy"])

    upstream = upstream_for(model, references, "qwen2-copy")
    np.testing.assert_allclose(
        model.get_logits(PROMPTS), upstream.get_logits(PROMPTS), rtol=1e-4, atol=1e-4
    )
    assert model.get_responses(PROMPTS) == upstream.get_responses(PROMPTS)

    # As upstream, adapters are applied again after the slow path.
    model.apply_lora(2)
    assert model.reset_model()


def test_reset_model_slow_path_to_another_architecture(
    checkpoints,
    references,
    tmp_path,
) -> None:
    model = Model(make_settings(checkpoints["qwen2"]))
    model.apply_lora(2)
    engine = model.engine

    model.settings.model = str(checkpoints["llama"])
    assert not model.reset_model()

    assert model.engine is not engine
    assert model.get_layers() == [0, 1]
    assert model.arch.model_type == "llama"
    assert model.lora_rank is None

    # The evaluated checkpoint's generation parameters apply from now on.
    state = model.lm_eval_state()
    assert state.generation_config.repetition_penalty == 1.1
    assert state.hf_config.model_type == "llama"
    assert state.engine is model.engine
    assert state.params is model.params

    upstream = upstream_for(model, references, "llama")
    assert model.get_responses(PROMPTS) == upstream.get_responses(PROMPTS)
    np.testing.assert_allclose(
        model.get_residuals(PROMPTS),
        upstream.get_residuals(PROMPTS),
        rtol=1e-4,
        atol=1e-5,
    )

    # Exports describe the new checkpoint.
    model.apply_lora(2)
    model.save_adapter(str(tmp_path / "adapter"))
    adapter_config = json.loads((tmp_path / "adapter/adapter_config.json").read_text())
    assert adapter_config["base_model_name_or_path"] == str(checkpoints["llama"])
    assert len(adapter_config["target_modules"]) == 4


# Loading.


def test_dtype_entries(checkpoints, tmp_path, capsys) -> None:
    # float16 is not used: an explicit entry loads bfloat16, with a warning.
    model = Model(make_settings(checkpoints["qwen2"], dtypes=["float16"]))
    assert model.dtype == jnp.bfloat16
    assert model.get_base_weights("attn.o_proj").dtype == jnp.bfloat16
    assert "float16 is not used" in capsys.readouterr().out

    # "auto" follows the checkpoint's dtype, mapping float16 to bfloat16.
    directory = tmp_path / "float16"
    shutil.copytree(checkpoints["qwen2"], directory)
    config = json.loads((directory / "config.json").read_text())
    (directory / "config.json").write_text(json.dumps(config | {"dtype": "float16"}))
    assert Model(make_settings(directory)).dtype == jnp.bfloat16

    # Without a configured dtype, the storage dtype decides.
    del config["dtype"]
    (directory / "config.json").write_text(json.dumps(config))
    assert Model(make_settings(directory)).dtype == jnp.float32

    # Unknown entries fail like any other dtype.
    model = Model(make_settings(checkpoints["qwen2"], dtypes=["float64", "auto"]))
    assert model.dtype == jnp.float32
    assert "Unsupported dtype 'float64'" in capsys.readouterr().out


def test_failed_smoke_test_releases_device_memory(checkpoints, monkeypatch) -> None:
    before = live_bytes()
    loads = []
    load_params = weights.load_params

    def recording_load(*args: Any) -> Any:
        loads.append(live_bytes())
        return load_params(*args)

    generate = engine_module.Engine.generate
    calls = []

    def failing_generate(self, *args: Any, **kwargs: Any) -> Any:
        calls.append(args[2].tokens.shape)
        if len(calls) == 1:
            raise RuntimeError("probability tensor contains either inf, nan or < 0")
        return generate(self, *args, **kwargs)

    monkeypatch.setattr(weights, "load_params", recording_load)
    monkeypatch.setattr(engine_module.Engine, "generate", failing_generate)

    model = Model(make_settings(checkpoints["qwen2"], dtypes=["float32", "bfloat16"]))
    assert model.dtype == jnp.bfloat16

    # The smoke test generates one token for one prompt.
    assert calls[0][0] == 1

    # The float32 parameters were freed before the bfloat16 ones were loaded.
    assert len(loads) == 2
    assert loads[1] <= before + 2**20


def test_all_dtypes_failing(checkpoints, monkeypatch, capsys) -> None:
    loads = []

    def failing_load(*args: Any) -> None:
        loads.append(args[3])
        raise MemoryError

    monkeypatch.setattr(weights, "load_params", failing_load)

    # "auto" resolves to float32, which was tried already.
    with pytest.raises(RuntimeError, match="all configured dtypes"):
        Model(make_settings(checkpoints["qwen2"], dtypes=["float32", "auto"]))
    assert loads == [np.dtype(np.float32)]
    assert "float32 was tried already" in capsys.readouterr().out


@pytest.mark.parametrize("family", ["qwen3_moe", "mixtral"])
def test_moe_checkpoints_load_in_bfloat16(tmp_path, family: str) -> None:
    """
    Mixture-of-experts checkpoints pass the smoke test in bfloat16 and generate in it.
    On the TPU, their expert matmuls used to fail to compile in bfloat16 (see
    layers._grouped_linear), so that these models silently fell back to float32, at
    twice the device memory, or failed to load with dtypes=["bfloat16"].
    """

    directory = build(
        tmp_path / family, family, generation_config={"eos_token_id": IM_END}
    )
    model = Model(make_settings(directory, dtypes=["bfloat16"]))
    assert model.dtype == jnp.bfloat16
    for name in ("router", "expert_gate", "expert_up", "expert_down"):
        assert model.params["layers"][name].dtype == jnp.bfloat16

    responses = model.get_responses(PROMPTS)
    assert len(responses) == len(PROMPTS)
    assert all(isinstance(response, str) for response in responses)


# PRNG keys.


@pytest.mark.parametrize("seed", [2**32, -1, None, 1.5])
def test_invalid_seeds_are_rejected(model, monkeypatch, seed) -> None:
    monkeypatch.setattr(model.settings, "seed", seed)
    with pytest.raises(ValueError, match="seed"):
        model.apply_lora(1)
    with pytest.raises(ValueError, match="seed"):
        model.lm_eval_state().next_key()
    assert model.adapters is None


def test_sampling_keys(model, monkeypatch) -> None:
    monkeypatch.setattr(model.settings, "seed", 2**32 - 1)
    monkeypatch.setattr(model, "_sampling_calls", 0)
    root = jax.random.key(2**32 - 1)

    next_key = model.lm_eval_state().next_key
    for count in range(2):
        expected = jax.random.fold_in(jax.random.fold_in(root, 3), count)
        np.testing.assert_array_equal(
            jax.random.key_data(next_key()), jax.random.key_data(expected)
        )


# Batch sizes.


def test_auto_mode_runs_the_whole_call_once(checkpoints, monkeypatch) -> None:
    model = Model(make_settings(checkpoints["qwen2"]))

    def forbidden(*args: Any) -> None:
        raise AssertionError("auto mode must not limit batch sizes")

    monkeypatch.setattr(model.engine, "batch_limit", forbidden)
    monkeypatch.setattr(model.engine, "lower_limit", forbidden)

    calls = []
    with recording(model.engine, "generate", calls):
        model.get_responses(PROMPTS)
    assert [batch.tokens.shape for _, _, batch, _ in calls] == [(len(PROMPTS), 64)]
    assert calls[0][2].n_real == len(PROMPTS)

    with recording(model.engine, "capture", calls):
        model.get_residuals(PROMPTS)
    assert calls[-1][2].tokens.shape == (len(PROMPTS), 64)

    # Executables exist at exactly B = n (besides the smoke test's).
    sizes = {
        (key.entry, size)
        for key, size in model.engine._executables
        if key.max_new_tokens != 1
    }
    assert sizes == {("generate", len(PROMPTS)), ("capture", len(PROMPTS))}


def run_tuning_candidate(model: Model, prompts: list[Prompt]) -> bool:
    """main.py's classification of a batch-size candidate: True if it fits."""

    try:
        model.get_responses(prompts)
    except Exception as error:
        if not is_out_of_memory(error):
            raise
        return False
    return True


def test_auto_mode_errors_reach_the_tuning_loop(checkpoints, monkeypatch) -> None:
    model = Model(make_settings(checkpoints["qwen2"]))

    def forbidden(*args: Any) -> None:
        raise AssertionError("auto mode must not limit batch sizes")

    monkeypatch.setattr(model.engine, "batch_limit", forbidden)
    monkeypatch.setattr(model.engine, "lower_limit", forbidden)

    # A failed memory check, without halving.
    with monkeypatch.context() as patch:
        patch.setattr(model.engine, "fits", lambda key, size: size < 4)
        assert run_tuning_candidate(model, PROMPTS[:2])
        with pytest.raises(DeviceMemoryError):
            model.get_responses(PROMPTS[:4])
        assert not run_tuning_candidate(model, PROMPTS[:4])

    # An out-of-memory error at run time, without a retry.
    calls = []

    def exhausted(*args: Any) -> None:
        calls.append(args)
        raise jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: simulated")

    with monkeypatch.context() as patch:
        patch.setattr(model.engine, "generate", exhausted)
        assert not run_tuning_candidate(model, PROMPTS[:8])
    assert len(calls) == 1

    # Any other error is a genuine failure.
    def broken(*args: Any) -> None:
        raise ValueError("broken")

    with monkeypatch.context() as patch:
        patch.setattr(model.engine, "generate", broken)
        with pytest.raises(ValueError, match="broken"):
            run_tuning_candidate(model, PROMPTS[:2])


def test_out_of_memory_halves_the_batch(checkpoints, monkeypatch) -> None:
    model = Model(make_settings(checkpoints["qwen2"], batch_size=BATCH_SIZE))
    expected_responses = model.get_responses_batched(PROMPTS)
    expected_logits = model.get_logits_batched(PROMPTS)

    # The batches as (B, T): generation packs the first short prompt into the long
    # prompts' batch, captures run each bucket separately. The first batch fails
    # and runs again in two batches of two; the other batch keeps its size.
    expected_shapes = {
        "generate": [(4, 64), (2, 64), (2, 64), (4, 32)],
        "capture": [(4, 32), (2, 32), (2, 32), (4, 64)],
    }
    for name in ("generate", "capture"):
        shapes = []
        lowered = []
        original = getattr(model.engine, name)

        def failing(*args: Any, original=original, shapes=shapes) -> Any:
            shapes.append(args[2].tokens.shape)
            if len(shapes) == 1:
                raise jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: simulated")
            return original(*args)

        with (
            monkeypatch.context() as patch,
            recording(model.engine, "lower_limit", lowered),
        ):
            patch.setattr(model.engine, name, failing)
            if name == "generate":
                assert model.get_responses_batched(PROMPTS) == expected_responses
            else:
                np.testing.assert_allclose(
                    model.get_logits_batched(PROMPTS),
                    expected_logits,
                    rtol=1e-6,
                    atol=1e-6,
                )

        assert shapes == expected_shapes[name]
        assert [size for _, size in lowered] == [BATCH_SIZE]


@pytest.mark.parametrize("batch_size", [BATCH_SIZE, 8])
def test_generation_packs_rows_across_buckets(
    model,
    settings,
    references,
    batch_size,
) -> None:
    """
    Generation fills the long prompts' batches with short prompts, so that a call
    needs as few decodes as B_eff allows, and the responses are still upstream's.
    Captures keep one bucket per batch.
    """

    settings.batch_size = batch_size
    upstream = upstream_for(model, references)
    prompt_lengths = lengths(model, PROMPTS)

    calls = []
    with recording(model.engine, "generate", calls):
        responses = model.get_responses_batched(PROMPTS)
    assert responses == upstream.get_responses_batched(PROMPTS)

    # The long prompts first, then the short ones, each in their original order.
    order = [1, 3, 5, 0, 2, 4, 6]
    expected = {
        BATCH_SIZE: [(4, 64, order[:4]), (4, 32, order[4:])],
        8: [(8, 64, order)],
    }
    batches = [
        (*batch.tokens.shape, np.sum(batch.mask[: batch.n_real], axis=1).tolist())
        for _, _, batch, _ in calls
    ]
    assert batches == [
        (size, length, [prompt_lengths[index] for index in rows])
        for size, length, rows in expected[batch_size]
    ]

    calls = []
    with recording(model.engine, "capture", calls):
        model.get_logits_batched(PROMPTS)
    assert [batch.tokens.shape[1] for _, _, batch, _ in calls] == [32, 64]


def test_packing_at_most_doubles_the_cache(model, settings) -> None:
    """
    A row joins a batch of a longer bucket only if that at most doubles its KV-cache
    length (bucket + max_new_tokens).
    """

    settings.batch_size = 8
    very_long = Prompt(system=SYSTEM_PROMPT, user=f"{LONG[0]} {LONG[1]}")
    prompts = [*PROMPTS, very_long]
    assert 64 < lengths(model, [very_long])[0] <= 128

    calls = []
    with recording(model.engine, "generate", calls):
        model.get_responses_batched(prompts)

    # The very long prompt's batch takes the long prompts, whose cache grows from
    # 64 + 10 to 128 + 10 tokens, but not the short ones, whose cache would grow
    # from 32 + 10 tokens.
    assert MAX_RESPONSE_LENGTH == 10
    assert [(batch.tokens.shape, batch.n_real) for _, _, batch, _ in calls] == [
        ((4, 128), 4),
        ((4, 32), 4),
    ]


def test_adapters_are_reserved_only_until_they_exist(checkpoints, monkeypatch) -> None:
    """
    Calls without adapters keep room for rank-50 adapters free only while there are
    none: inside lora_disabled(), the allocated adapters are in use already.
    """

    model = Model(make_settings(checkpoints["qwen2"]))
    short = PROMPTS[::2]
    key = engine_module.ShapeKey("capture", 32, None, want=frozenset({"logits"}))

    # Device memory that holds the call (auto mode: B = 4) and the 5 % reserve, but
    # not room for adapters as well.
    limit = 2**30
    need = model.engine.memory_need(key, len(short))
    in_use = int(limit - engine_module.MEMORY_RESERVE * limit - need)
    monkeypatch.setattr(
        engine_module,
        "_memory_stats",
        lambda devices: [{"bytes_limit": limit, "bytes_in_use": in_use}],
    )

    checks = []
    with recording(model.engine, "fits", checks):
        with pytest.raises(DeviceMemoryError):
            model.get_logits(short)

        model.apply_lora(2)
        with model.lora_disabled():
            model.get_logits(short)
    assert checks == [(key, len(short))] * 2

    # A fast reset keeps the adapters allocated; after the slow path, the kept engine
    # keeps room for them free again until they are applied anew.
    assert model.reset_model()
    assert model.engine.fits(key, len(short))
    model.settings.model = str(checkpoints["qwen2-copy"])
    assert not model.reset_model()
    assert not model.engine.fits(key, len(short))


# Benchmarks and export.


def test_lm_eval_state(model) -> None:
    state = model.lm_eval_state()
    assert state.engine is model.engine
    assert state.params is model.params
    assert state.lora is None
    assert state.hf_config is model.checkpoint.config
    assert state.generation_config is model.checkpoint.generation_config
    assert state.pad_id == model.tokenizer.pad_token_id
    assert state.next_key.__self__ is model


# JaxLM's state resolution against the facade, (a) to (d) of "Testing" in
# docs/DESIGN.md. JaxLM fetches the facade's state per call, so a JaxLM built
# earlier gives the results of one built for the current state.

LOGLIKELIHOOD_ARGUMENTS = [
    ("The capital of France is", " Paris"),
    ("Once upon a time there", " was a dragon who lived in a cave"),
]

# One request per group of generation kwargs, so each runs alone: the padding token
# (which the switch to another checkpoint keeps, as upstream) is never penalised.
GENERATION_ARGUMENTS = [
    ("Once upon a time", {"until": ["\n\n"], "max_gen_toks": 8, "do_sample": False}),
    ("List three colours:", {"until": ["."], "max_gen_toks": 6, "do_sample": False}),
]


def jax_lm(model: Model) -> JaxLM:
    # As main.py and the benchmark scorer build it.
    return JaxLM(model.tokenizer, model.lm_eval_state)


def lm_eval_results(lm: JaxLM) -> tuple[list[tuple[float, bool]], list[str]]:
    def instances(request_type: str, arguments: list[tuple]) -> list[Instance]:
        return [
            Instance(request_type=request_type, doc={}, arguments=args, idx=index)
            for index, args in enumerate(arguments)
        ]

    return (
        lm.loglikelihood(instances("loglikelihood", LOGLIKELIHOOD_ARGUMENTS)),
        lm.generate_until(instances("generate_until", GENERATION_ARGUMENTS)),
    )


def test_jax_lm_follows_adapters(checkpoints) -> None:
    model = Model(make_settings(checkpoints["qwen2"]))
    lm = jax_lm(model)
    base = lm_eval_results(lm)

    # (a) Built before apply_lora, it scores the adapted model.
    model.apply_lora(2)
    for component, (A, B) in random_adapters(model).items():
        model.set_lora(component, A, B)
    adapted = lm_eval_results(lm)
    assert adapted == lm_eval_results(jax_lm(model))
    assert [logprob for logprob, _ in adapted[0]] != [logprob for logprob, _ in base[0]]

    # (b) Inside lora_disabled(), it scores the model without adapters.
    with model.lora_disabled():
        assert lm_eval_results(lm) == base
    assert lm_eval_results(lm) == adapted


def test_jax_lm_follows_a_reloaded_model(checkpoints) -> None:
    # (c) After the slow path of reset_model to another checkpoint of the same
    # architecture, which keeps the engine.
    model = Model(make_settings(checkpoints["qwen2"]))
    model.apply_lora(2)
    lm = jax_lm(model)
    before = lm_eval_results(lm)
    engine = model.engine

    model.settings.model = str(checkpoints["qwen2-copy"])
    assert not model.reset_model()
    assert model.engine is engine

    expected = lm_eval_results(jax_lm(Model(make_settings(checkpoints["qwen2-copy"]))))
    assert lm_eval_results(lm) == expected
    assert expected[0] != before[0]


def test_jax_lm_follows_another_architecture(
    checkpoints, tmp_path, monkeypatch
) -> None:
    # (d) After the slow path of reset_model to a checkpoint with another depth,
    # max_position_embeddings and generation config, served as Hub repositories, so
    # that each has its own commit.
    repos = {"org/qwen2": checkpoints["qwen2"], "org/llama": checkpoints["llama"]}
    shas = {"org/qwen2": "1" * 40, "org/llama": "2" * 40}
    FakeHub(tmp_path / "hub", repos, shas).install(monkeypatch)

    def tokenizer_from_hub(repo_id: str, revision: str) -> Any:
        assert revision == shas[repo_id]
        return AutoTokenizer.from_pretrained(repos[repo_id])

    monkeypatch.setattr(
        model_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=tokenizer_from_hub),
    )

    model = Model(make_settings("org/qwen2"))
    model.apply_lora(2)
    lm = jax_lm(model)
    max_length = lm.max_length
    before = lm_eval_results(lm)
    engine = model.engine

    model.settings.model = "org/llama"
    assert not model.reset_model()
    assert model.engine is not engine
    assert model.get_layers() == [0, 1]

    fresh = jax_lm(Model(make_settings("org/llama")))
    assert lm.max_length == fresh.max_length != max_length
    expected = lm_eval_results(fresh)
    assert lm_eval_results(lm) == expected
    assert expected != before

    # The adapter of the evaluated model records its commit.
    model.apply_lora(2)
    model.save_adapter(str(tmp_path / "adapter"))
    adapter_config = json.loads((tmp_path / "adapter/adapter_config.json").read_text())
    assert adapter_config["base_model_name_or_path"] == "org/llama"
    assert adapter_config["revision"] == shas["org/llama"]


def test_export_round_trips(checkpoints, references, tmp_path, monkeypatch) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = Model(make_settings(checkpoints["qwen2"]))
    for save in (model.save_merged, model.save_adapter):
        with pytest.raises(RuntimeError, match="apply_lora"):
            save(str(tmp_path / "early"))

    model.apply_lora(2)
    for component, (A, B) in random_adapters(model).items():
        model.set_lora(component, A, B)
    logits = model.get_logits(PROMPTS)

    # The merged model, with the facade's tokenizer.
    merged = tmp_path / "merged"
    model.save_merged(str(merged))
    tokenizer = AutoTokenizer.from_pretrained(merged)
    assert tokenizer.pad_token == model.tokenizer.pad_token
    reference = AutoModelForCausalLM.from_pretrained(
        merged, dtype=torch.float32, attn_implementation="eager"
    )
    np.testing.assert_allclose(
        Upstream(reference, model.tokenizer, model.settings).get_logits(PROMPTS),
        logits,
        rtol=1e-4,
        atol=1e-4,
    )

    # The adapter, loaded with PEFT.
    adapter = tmp_path / "adapter"
    model.save_adapter(str(adapter))
    assert sorted(path.name for path in adapter.iterdir()) == [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    peft_model = PeftModel.from_pretrained(
        load_reference(checkpoints["qwen2"]), adapter
    )
    np.testing.assert_allclose(
        Upstream(peft_model, model.tokenizer, model.settings).get_logits(PROMPTS),
        logits,
        rtol=1e-4,
        atol=1e-4,
    )

    # Uploads contain exactly the export.
    uploads = []

    class FakeApi:
        def __init__(self, token: str):
            self.token = token

        def create_repo(self, repo_id: str, private: bool, exist_ok: bool) -> Any:
            return SimpleNamespace(repo_id=repo_id)

        def upload_folder(self, repo_id: str, folder_path: str, **kwargs: Any) -> None:
            uploads.append(sorted(path.name for path in Path(folder_path).iterdir()))

    monkeypatch.setattr(export, "HfApi", FakeApi)
    model.push_to_hub(
        "org/model", private=True, token="t", strategy=ExportStrategy.ADAPTER
    )
    model.push_to_hub(
        "org/model", private=True, token="t", strategy=ExportStrategy.MERGE
    )
    assert uploads[0] == ["adapter_config.json", "adapter_model.safetensors"]
    assert {"model.safetensors", "config.json", "tokenizer.json"} <= set(uploads[1])


# A real model on the TPU.


@pytest.mark.tpu
@pytest.mark.slow
def test_real_model_responses_on_tpu() -> None:
    model = Model(make_settings("Qwen/Qwen2.5-0.5B-Instruct", max_response_length=16))
    assert model.dtype == jnp.bfloat16

    prompts = [
        Prompt(system=SYSTEM_PROMPT, user="What is 1+1? Answer with a single digit."),
        Prompt(system=SYSTEM_PROMPT, user="What is the capital of France?"),
    ]
    responses = model.get_responses(prompts, skip_special_tokens=True)
    assert "2" in responses[0]
    assert "Paris" in responses[1]

    model.settings.batch_size = 4
    responses = model.get_responses_batched(prompts * 3, skip_special_tokens=True)
    assert all("2" in response for response in responses[0::2])
    assert all("Paris" in response for response in responses[1::2])
