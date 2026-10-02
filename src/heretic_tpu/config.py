# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import os
from enum import Enum
from typing import Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

# !!!IMPORTANT!!!
#
# Any settings added to the classes defined in this module
# must be evaluated for privacy implications and have
# exclude=True set in their field definitions if appropriate.


class QuantizationMethod(str, Enum):
    NONE = "none"


# Settings of upstream Heretic that configure Accelerate's model placement,
# which heretic-tpu doesn't use. They are still found in settings restored from
# upstream runs (config.toml, reproduce.json, study checkpoints), which must
# remain loadable, so they are ignored with a warning instead of being rejected.
UNSUPPORTED_UPSTREAM_SETTINGS = ["device_map", "max_memory"]


class ExportStrategy(str, Enum):
    MERGE = "merge"
    ADAPTER = "adapter"


class SingleDatasetSpecification(BaseModel):
    dataset: str = Field(
        description="Hugging Face dataset ID, or path to dataset on disk."
    )

    commit: str | None = Field(
        default=None,
        description="Hugging Face commit hash of the dataset.",
    )

    config: str | None = Field(
        default=None,
        description=(
            "Dataset config/subset name. Each config can have its own split. "
            "Used to load a specific config of a dataset that has multiple configurations."
        ),
    )

    split: str | None = Field(
        default=None,
        description="Portion of the dataset to use. Required for datasets, optional for plain text files.",
    )

    column: str | None = Field(
        default=None,
        description="Column in the dataset that contains the prompts. Required for datasets, ignored for plain text files.",
    )

    prefix: str = Field(
        default="",
        description="Text to prepend to each prompt.",
    )

    suffix: str = Field(
        default="",
        description="Text to append to each prompt.",
    )

    system_prompt: str | None = Field(
        default=None,
        description="System prompt to use with the prompts (overrides global system prompt if set).",
    )


DatasetSpecification: TypeAlias = (
    SingleDatasetSpecification | list[SingleDatasetSpecification]
)


class ScorerConfig(BaseModel):
    """
    Configuration for a scorer plugin.

    TOML format:
    - { plugin = "<plugin>", optimization = "<optimization>", instance_name = "<optional>" }
    """

    plugin: str = Field(
        description=(
            "Plugin to load. Either a file path with class name "
            "(`path/to/plugin.py:ClassName`) or a fully-qualified import path "
            "(`module.submodule.ClassName`)."
        ),
    )

    optimization: Literal["minimize", "maximize", "none"] = Field(
        description=(
            "Optimization direction for this scorer. "
            '"minimize" / "maximize" to include the scorer as an objective, '
            '"none" to compute the score without optimizing for it.'
        ),
    )

    instance_name: str | None = Field(
        default=None,
        description=(
            "Optional name to distinguish multiple instances of the same plugin class. "
            "Instance-specific settings live under `[scorer.<ClassName>_<instance_name>]`."
        ),
    )

    @field_validator("instance_name")
    @classmethod
    def validate_instance_name(cls, value: str | None) -> str | None:
        if value is None:
            return value

        if not value.strip():
            raise ValueError("cannot be empty or whitespace")

        if "." in value:
            raise ValueError("'.' is not allowed")

        if any(char.isspace() for char in value):
            raise ValueError("whitespace is not allowed")

        return value


class ModifierConfig(BaseModel):
    """
    Configuration for a modifier plugin.

    TOML format:
    - { plugin = "<plugin>", instance_name = "<optional>" }
    """

    plugin: str = Field(
        description=(
            "Plugin to load. Either a file path with class name "
            "(`path/to/plugin.py:ClassName`) or a fully-qualified import path "
            "(`module.submodule.ClassName`)."
        ),
    )

    instance_name: str | None = Field(
        default=None,
        description=(
            "Optional name to distinguish multiple instances of the same plugin class. "
            "Instance-specific settings live under `[modifier.<ClassName>_<instance_name>]`."
        ),
    )

    @field_validator("instance_name")
    @classmethod
    def validate_instance_name(cls, value: str | None) -> str | None:
        if value is None:
            return value

        if not value.strip():
            raise ValueError("cannot be empty or whitespace")

        if "." in value:
            raise ValueError("'.' is not allowed")

        if any(char.isspace() for char in value):
            raise ValueError("whitespace is not allowed")

        return value


class BenchmarkSpecification(BaseModel):
    task: str = Field(
        description="Task ID of the benchmark in the Language Model Evaluation Harness."
    )

    name: str = Field(description="Name of the benchmark for presentation purposes.")

    description: str = Field(
        description="Description of the benchmark for presentation purposes."
    )


class Settings(BaseSettings):
    model: str = Field(description="Hugging Face model ID, or path to model on disk.")

    model_commit: str | None = Field(
        default=None,
        description="Hugging Face commit hash of the model.",
    )

    evaluate_model: str | None = Field(
        default=None,
        description=(
            "If this model ID or path is set, then instead of abliterating the main model, "
            "evaluate this model relative to the main model."
        ),
        exclude=True,
    )

    collect_reproducibles: str | None = Field(
        default=None,
        description=(
            "If this directory path is set, then instead of abliterating a model, "
            "download all reproduce.json files from public Heretic model repositories "
            "on Hugging Face, and store them in that directory for archival purposes."
        ),
        exclude=True,
    )

    reproduce: str | None = Field(
        default=None,
        description=(
            "If this path or URL to a reproduce.json file is set, load reproduction information "
            "from that file, and attempt to reproduce the abliterated model it originated from."
        ),
        exclude=True,
    )

    dtypes: list[str] = Field(
        default=[
            # In practice, "auto" almost always means bfloat16.
            "auto",
            # If "auto" resolves to float32, and that fails because it is too large,
            # try bfloat16.
            "bfloat16",
            # If neither of those work, fall back to float32 (which will of course fail
            # if that was the dtype "auto" resolved to).
            "float32",
        ],
        description=(
            "List of dtypes to try when loading model tensors. "
            "If loading with a dtype fails, the next dtype in the list will be tried. "
            'float16 is not used: "float16" entries, and "auto" for float16 checkpoints, '
            "load the model in bfloat16."
        ),
    )

    quantization: QuantizationMethod = Field(
        default=QuantizationMethod.NONE,
        description=(
            "Quantisation method to use when loading the model. "
            'Only "none" (no quantisation) is supported, because bitsandbytes '
            "quantisation is unavailable on TPU."
        ),
    )

    parallelism: Literal["auto", "single", "tensor"] = Field(
        default="auto",
        description=(
            "How to place the model on the local TPU devices. Options: "
            '"auto" (a single device if the model fits on one, otherwise "tensor"), '
            '"single" (the first local device only), '
            '"tensor" (tensor parallelism across all local devices).'
        ),
    )

    offload_outputs_to_cpu: bool = Field(
        default=True,
        description=(
            "Whether to move intermediate analysis tensors (such as residuals and logprobs) "
            "to CPU memory as soon as possible to reduce peak device memory usage. "
            "This lowers peak device memory usage during residual analysis and evaluation, "
            "but may slightly reduce performance due to host/device transfers."
        ),
    )

    batch_size: NonNegativeInt = Field(
        default=0,  # auto
        description="Number of input sequences to process in parallel (0 = auto).",
    )

    max_batch_size: PositiveInt = Field(
        default=128,
        description="Maximum batch size to try when automatically determining the optimal batch size.",
        # When storing a settings object, the batch size is already fixed,
        # either determined by the automatic mechanism or by explicit user choice.
        exclude=True,
    )

    batch_size_test_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="train[:256]",
            column="text",
        ),
        description="Dataset of prompts to use for automatically determining the optimal batch size.",
        # When storing a settings object, the batch size is already fixed,
        # either determined by the automatic mechanism or by explicit user choice.
        exclude=True,
    )

    max_response_length: PositiveInt = Field(
        default=100,
        description="Maximum number of tokens to generate for each response.",
    )

    response_prefix: str | None = Field(
        default=None,
        description=(
            "Common prefix to assume for all responses, so that evaluation happens "
            "at the point where responses start to differ for different prompts. "
            "If not set, the prefix is determined automatically by comparing multiple responses."
        ),
    )

    response_prefix_test_prompts: DatasetSpecification = Field(
        default=[
            SingleDatasetSpecification(
                dataset="mlabonne/harmless_alpaca",
                split="train[:100]",
                column="text",
            ),
            SingleDatasetSpecification(
                dataset="mlabonne/harmful_behaviors",
                split="train[:100]",
                column="text",
            ),
        ],
        description="Dataset of prompts to use for automatically determining the response prefix.",
        # When storing a settings object, the response prefix is already fixed,
        # either determined by the automatic mechanism or by explicit user choice.
        exclude=True,
    )

    chain_of_thought_skips: list[tuple[str, str]] = Field(
        default=[
            # Most thinking models.
            (
                "<think>",
                "<think></think>",
            ),
            # gpt-oss.
            (
                "<|channel|>analysis<|message|>",
                "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>",
            ),
            # Unknown, suggested by user.
            (
                "<thought>",
                "<thought></thought>",
            ),
            # Unknown, suggested by user.
            (
                "[THINK]",
                "[THINK][/THINK]",
            ),
        ],
        description=(
            "List of pairs of the form (cot_initializer, closed_cot_block) used to skip "
            "the Chain-of-Thought block in responses, so that evaluation happens "
            "at the start of the actual response."
        ),
        # When storing a settings object, the response prefix is already fixed,
        # either determined by the automatic mechanism or by explicit user choice.
        exclude=True,
    )

    print_debug_information: bool = Field(
        default=False,
        description="Whether to print additional information that can help with debugging.",
        exclude=True,
    )

    scorers: list[ScorerConfig] = Field(
        default=[
            ScorerConfig(
                plugin="heretic_tpu.scorers.keyword_rate.KeywordRate",
                optimization="minimize",
            ),
            ScorerConfig(
                plugin="heretic_tpu.scorers.kl_divergence.KLDivergence",
                optimization="minimize",
            ),
        ],
        description=(
            "List of scorer plugin configs. Each entry is an object "
            "{ plugin = <plugin>, optimization = <optimization>, instance_name = <optional> }. "
            '<optimization> is one of "minimize", "maximize", or "none" (do not optimize).'
        ),
    )

    modifiers: list[ModifierConfig] = Field(
        default=[
            ModifierConfig(
                plugin="heretic_tpu.modifiers.ara.ARA",
            ),
        ],
        description=(
            "List of modifier plugin configs. Each entry is an object "
            "{ plugin = <plugin>, instance_name = <optional> }. "
            "Note that only a single modifier can currently be applied, "
            "and this list must contain exactly one entry."
        ),
    )

    n_trials: PositiveInt = Field(
        default=100,
        description="Number of abliteration trials to run during optimization.",
    )

    n_startup_trials: NonNegativeInt = Field(
        default=30,
        description="Number of trials that use random sampling for the purpose of exploration.",
    )

    seed: int | None = Field(
        default=None,
        description=(
            "Random seed for reproducible optimization. "
            "Applies to Python's random module, NumPy, JAX, and Optuna."
        ),
    )

    study_checkpoint_dir: str = Field(
        default="checkpoints",
        description="Directory to save and load study progress to/from.",
        exclude=True,
    )

    compilation_cache_dir: str = Field(
        default="~/.cache/heretic-tpu/xla",
        description=(
            "Directory of the persistent XLA compilation cache, which lets later runs "
            "reuse compiled programs instead of compiling them again. "
            "An empty string disables the cache."
        ),
        # The default contains "~", which must be expanded too.
        validate_default=True,
        exclude=True,
    )

    benchmarks: list[BenchmarkSpecification] = Field(
        default=[
            BenchmarkSpecification(
                task="agieval",
                name="AGIEval",
                description="A Human-Centric Benchmark for Evaluating Foundation Models",
            ),
            BenchmarkSpecification(
                task="bbh",
                name="BIG-Bench Hard (BBH)",
                description="Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them",
            ),
            BenchmarkSpecification(
                task="commonsense_qa",
                name="CommonsenseQA",
                description="A Question Answering Challenge Targeting Commonsense Knowledge",
            ),
            BenchmarkSpecification(
                task="eq_bench",
                name="EQ-Bench",
                description="An Emotional Intelligence Benchmark for Large Language Models",
            ),
            BenchmarkSpecification(
                task="gsm8k",
                name="GSM8K",
                description="Training Verifiers to Solve Math Word Problems",
            ),
            BenchmarkSpecification(
                task="hellaswag",
                name="HellaSwag",
                description="Can a Machine Really Finish Your Sentence?",
            ),
            BenchmarkSpecification(
                task="ifeval",
                name="IFEval",
                description="Instruction-Following Evaluation for Large Language Models",
            ),
            BenchmarkSpecification(
                task="mmlu",
                name="MMLU",
                description="Measuring Massive Multitask Language Understanding",
            ),
            BenchmarkSpecification(
                task="mmlu_pro",
                name="MMLU-Pro",
                description="A More Robust and Challenging Multi-Task Language Understanding Benchmark",
            ),
            BenchmarkSpecification(
                task="piqa",
                name="PIQA",
                description="Reasoning about Physical Commonsense in Natural Language",
            ),
            BenchmarkSpecification(
                task="winogrande",
                name="WinoGrande",
                description="An Adversarial Winograd Schema Challenge at Scale",
            ),
        ],
        description="Benchmarks to offer to the user for evaluating abliterated models.",
        exclude=True,
    )

    max_shard_size: PositiveInt | str = Field(
        default="5GB",
        description=(
            "Accepted for compatibility with Heretic, but ignored: "
            "exported models keep the shard layout of the original model."
        ),
    )

    export_strategy: ExportStrategy | None = Field(
        default=None,
        description='How to export the model: "merge", "adapter", or unset to prompt the user.',
    )

    checkpoint_action: str | None = Field(
        default=None,
        description='Action to take in case a checkpoint exists: "continue", "restart", or unset to prompt the user.',
    )

    trial_index: NonNegativeInt | None = Field(
        default=None,
        description="Index (in the sorted Pareto front) of the trial to use, or unset to prompt the user.",
    )

    n_additional_trials: PositiveInt | None = Field(
        default=None,
        description="Number of additional trials to run, or unset to prompt the user.",
    )

    model_action: str | None = Field(
        default=None,
        description='Action to take with the decensored model: "save", "upload", or unset to prompt the user.',
    )

    save_directory: str | None = Field(
        default=None,
        description="Directory to save the model to, or unset to prompt the user.",
        exclude=True,
    )

    upload_repo_id: str | None = Field(
        default=None,
        description="Name of the Hugging Face repository to upload the model to, or unset to prompt the user.",
        exclude=True,
    )

    upload_repo_private: bool | None = Field(
        default=None,
        description="Whether the Hugging Face repository to upload the model to should be private, or unset to prompt the user.",
    )

    upload_reproducibility_information: str | None = Field(
        default=None,
        description='Which reproducibility information to add to the Hugging Face repository: "full", "basic", "none", or unset to prompt the user.',
    )

    ignore_mismatches: bool | None = Field(
        default=None,
        description="Whether to attempt to reproduce the model even if there are environment mismatches, or unset to prompt the user.",
    )

    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="System prompt to use when prompting the model.",
    )

    @field_validator("quantization", mode="before")
    @classmethod
    def validate_quantization(cls, value: Any) -> Any:
        # Upstream Heretic also offers 4-bit quantisation with bitsandbytes,
        # so explain why that option is missing instead of only listing the valid ones.
        if value == "bnb_4bit":
            raise ValueError(
                '"bnb_4bit" is not supported: bitsandbytes quantisation requires CUDA '
                'and is unavailable on TPU. Use "none".'
            )

        return value

    @field_validator("compilation_cache_dir")
    @classmethod
    def expand_compilation_cache_dir(cls, value: str) -> str:
        # Values from configuration files and the environment don't pass through
        # a shell that would expand "~".
        return os.path.expanduser(value)

    @model_validator(mode="before")
    @classmethod
    def ignore_unsupported_upstream_settings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        unsupported = [key for key in UNSUPPORTED_UPSTREAM_SETTINGS if key in data]

        if unsupported:
            # Imported here because utils imports this module.
            from .utils import print

            print(
                "[yellow]Warning: Ignoring "
                + " and ".join(f"[bold]{key}[/]" for key in unsupported)
                + ", which heretic-tpu does not support. "
                "Model placement is controlled by the [bold]parallelism[/] setting.[/]"
            )

            # Removing the keys also keeps them out of model_extra,
            # so they are not written back when the settings are stored.
            data = {key: value for key, value in data.items() if key not in unsupported}

        return data

    # We intentionally allow extra keys so users can provide plugin-specific
    # configuration in TOML tables like `[scorer.KeywordRate]` which are later
    # consumed via `settings.model_extra` (see `plugin.get_plugin_namespace`).
    model_config = SettingsConfigDict(extra="allow")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,  # Used during resume - should override *all* other sources.
            CliSettingsSource(
                settings_cls,
                cli_parse_args=True,
                cli_implicit_flags=True,
                cli_kebab_case=True,
            ),
            EnvSettingsSource(settings_cls, env_prefix="HERETIC_"),
            dotenv_settings,
            file_secret_settings,
            TomlConfigSettingsSource(settings_cls, toml_file="config.toml"),
        )
