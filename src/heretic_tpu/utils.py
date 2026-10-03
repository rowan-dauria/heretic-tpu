# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import huggingface_hub
import jax
import tomli_w
from datasets import DatasetDict, ReadInstruction, load_dataset, load_from_disk
from datasets.config import DATASET_STATE_JSON_FILENAME
from datasets.download.download_manager import DownloadMode
from datasets.utils.info_utils import VerificationMode
from huggingface_hub.utils import validate_repo_id
from optuna import Trial
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from psutil import Process
from questionary import Question
from rich.console import Console

from .config import DatasetSpecification, Settings, SingleDatasetSpecification
from .system import (
    get_accelerator_info_dict,
    get_cpu_info_dict,
    get_heretic_version_info,
    get_libtpu_version,
    get_package_version,
    get_python_env_info_dict,
    get_requirements_dict,
)

if TYPE_CHECKING:
    from .modifier import Modifier


T = TypeVar("T")


print = Console(highlight=False).print


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two dicts.

    Values from `override` take precedence. Nested dicts are merged recursively.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def parse_study_direction(optimization: str) -> StudyDirection:
    """
    Converts the optimization value stored as a `str` to the
    `StudyDirection` object required by Optuna.
    """
    if optimization == "none":
        return StudyDirection.NOT_SET
    return StudyDirection[optimization.upper()]


def print_memory_usage():
    def p(label: str, size_in_bytes: int):
        print(f"[grey50]{label}: [bold]{size_in_bytes / (1024**3):.2f} GB[/][/]")

    p("Resident system RAM", Process().memory_info().rss)

    # The CPU backend reports no memory statistics.
    memory_stats = [
        stats
        for stats in (device.memory_stats() for device in jax.local_devices())
        if stats is not None
    ]

    if memory_stats:
        p(
            "Allocated device memory",
            sum(stats.get("bytes_in_use", 0) for stats in memory_stats),
        )
        p(
            "Peak allocated device memory",
            sum(stats.get("peak_bytes_in_use", 0) for stats in memory_stats),
        )


def format_duration(seconds: float) -> str:
    seconds = round(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def format_exception(error: Exception) -> str:
    # Walk causal chain to find a non-empty message.
    current = error
    while current is not None:
        message = str(current).strip()
        if message:
            return message
        current = current.__cause__ or current.__context__

    # If there is no message in the entire causal chain, fall back to the complete traceback.
    return traceback.format_exc().strip()


def ask_if_unset(value: T, question: Question, unsafe: bool = False) -> T:
    if value is None:
        if unsafe:
            return question.unsafe_ask()
        else:
            return question.ask()
    else:
        return value


def is_hf_path(path: str) -> bool:
    """Checks whether a path likely refers to a Hugging Face repository."""

    # Match Transformers: Existing local paths take precedence over Hub lookup,
    # even if the path string is also a valid repository ID.
    if Path(path).exists():
        return False

    validate_repo_id(path)
    return True


@dataclass
class Prompt:
    system: str
    user: str


def get_split_slice(split_str: str, length: int) -> tuple[int, int]:
    """Resolves a split specification into absolute (start, end) indices."""

    # The split name is the part before the slice, e.g. "train" in "train[:400]".
    split_name = split_str.split("[")[0]

    # Associate the split with its number of examples (lines).
    name_to_length = {split_name: length}

    # Convert the instructions to absolute indices and select the first one.
    absolute_instruction = ReadInstruction.from_spec(split_str).to_absolute(
        name_to_length
    )[0]

    return absolute_instruction.from_, absolute_instruction.to


def _load_prompts_single(
    settings: Settings,
    specification: SingleDatasetSpecification,
) -> list[Prompt]:
    path = specification.dataset
    split_str = specification.split

    if os.path.isfile(path):
        # Plain text file with one prompt per line. Empty lines are ignored.
        with open(path, encoding="utf-8") as file:
            prompts = [line.strip() for line in file if line.strip()]

        # The split is optional for text files. When given, it selects a subset
        # of the lines using slice notation (e.g. "[:400]"). A synthetic split
        # name is prepended because ReadInstruction expects a named split.
        if split_str is not None:
            start, end = get_split_slice(f"_{split_str}", len(prompts))
            prompts = prompts[start:end]
    else:
        # All dataset sources require an explicit split and column.
        if split_str is None:
            raise ValueError(f'The "split" field is required for datasets: {path}')

        if specification.column is None:
            raise ValueError(f'The "column" field is required for datasets: {path}')

        if is_hf_path(path):
            # Pin to the latest commit if not already set, so the exact dataset
            # version is recorded for reproducibility.
            if specification.commit is None:
                try:
                    specification.commit = huggingface_hub.dataset_info(path).sha
                except Exception as error:  # noqa: BLE001
                    # Fetching the commit hash requires internet access, but the
                    # dataset itself may be fully cached locally. Proceed without
                    # pinning; an unpinned dataset disables the reproducibility
                    # offer during upload.
                    print(
                        f"[yellow]Warning: Could not fetch the latest commit hash for dataset [bold]{path}[/] ({error}). "
                        "The dataset version will not be pinned.[/]"
                    )
            dataset = load_dataset(
                path,
                name=specification.config,
                revision=specification.commit,
                split=split_str,
            )
        elif Path(path, DATASET_STATE_JSON_FILENAME).exists():
            # Dataset saved with datasets.save_to_disk; needs special handling.
            # Path should be the subdirectory for a particular split.
            dataset = load_from_disk(path)
            assert not isinstance(dataset, DatasetDict), (
                "Loading dataset dicts is not supported"
            )
            # Parse the split instructions and apply them.
            start, end = get_split_slice(split_str, len(dataset))
            dataset = dataset[start:end]
        else:
            # Path should be a local directory.
            dataset = load_dataset(
                path,
                name=specification.config,
                split=split_str,
                # Don't require the number of examples (lines) per split to be pre-defined.
                verification_mode=VerificationMode.NO_CHECKS,
                # But also don't use cached data, as the dataset may have changed on disk.
                download_mode=DownloadMode.FORCE_REDOWNLOAD,
            )

        prompts = list(dataset[specification.column])

    if specification.prefix:
        prompts = [f"{specification.prefix} {prompt}" for prompt in prompts]

    if specification.suffix:
        prompts = [f"{prompt} {specification.suffix}" for prompt in prompts]

    system_prompt = (
        settings.system_prompt
        if specification.system_prompt is None
        else specification.system_prompt
    )

    return [
        Prompt(
            system=system_prompt,
            user=prompt,
        )
        for prompt in prompts
    ]


def load_prompts(
    settings: Settings,
    specification: DatasetSpecification,
) -> list[Prompt]:
    if isinstance(specification, SingleDatasetSpecification):
        return _load_prompts_single(settings, specification)
    else:
        return [
            prompt
            for single_specification in specification
            for prompt in _load_prompts_single(settings, single_specification)
        ]


def format_dataset_specification(specification: DatasetSpecification) -> str:
    if isinstance(specification, SingleDatasetSpecification):
        return specification.dataset
    else:
        return (
            "\\["
            + ", ".join(
                single_specification.dataset for single_specification in specification
            )
            + "]"
        )


def is_dataset_specification_reproducible(specification: DatasetSpecification) -> bool:
    if isinstance(specification, SingleDatasetSpecification):
        return is_hf_path(specification.dataset) and specification.commit is not None
    else:
        return all(
            is_hf_path(single_specification.dataset)
            and single_specification.commit is not None
            for single_specification in specification
        )


def batchify(items: list[T], batch_size: int) -> list[list[T]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def get_upstream_plugin_name(name: str) -> str:
    """
    Maps the import path of a plugin that ships with heretic-tpu to the path
    of the same plugin in upstream Heretic, which heretic-tpu also accepts.
    """
    if name.startswith("heretic_tpu."):
        return "heretic." + name.removeprefix("heretic_tpu.")
    return name


def dump_settings(settings: Settings, exclude_none: bool = False) -> dict[str, Any]:
    """
    Serialises the settings with built-in plugins named as in upstream Heretic,
    so that both programs can read the stored settings.
    """
    data = settings.model_dump(exclude_none=exclude_none)

    for plugin_config in [*data["scorers"], *data["modifiers"]]:
        plugin_config["plugin"] = get_upstream_plugin_name(plugin_config["plugin"])

    return data


def get_readme_intro(
    settings: Settings,
    modifier: Modifier[Any],
    trial: Trial | FrozenTrial,
    contains_reproducibility_information: bool,
) -> str:
    if is_hf_path(settings.model):
        model_link = f"[{settings.model}](https://huggingface.co/{settings.model})"
    else:
        # Hide the path, which may contain private information.
        model_link = "a model"

    scores_raw = trial.user_attrs["scores"]
    scores_by_name: dict[str, dict[str, Any]] = {}
    score_names: list[str] = []
    for score in scores_raw:
        name = score["name"]
        scores_by_name[name] = score
        score_names.append(name)

    score_rows = "\n".join(
        [
            (
                f"| **{name}** | "
                f"{scores_by_name[name]['score']['md_display']} | "
                f"{scores_by_name[name]['baseline']['md_display']} |"
            )
            for name in score_names
        ]
    )

    if contains_reproducibility_information:
        reproducibility_instructions = """
> [!TIP]
> **This model is reproducible!**
>
> See the [README](reproduce/README.md) in the `reproduce` directory for more information.
"""
    else:
        reproducibility_instructions = ""

    return f"""# This is a decensored version of {model_link}, made using heretic-tpu v{
        version("heretic-tpu")
    }, a TPU port of [Heretic](https://heretic-project.org)
{reproducibility_instructions}
## {modifier.modifier_name} parameters

| Parameter | Value |
| :-------- | :---: |
{
        chr(10).join(
            [
                f"| **{name}** | {value} |"
                for name, value in modifier.render_trial_parameters(trial).items()
            ]
        )
    }

## Performance

| Metric | This model | Original model ({model_link}) |
| :----- | :--------: | :---------------------------: |
{score_rows}

-----

"""


def generate_config_toml(settings: Settings) -> str:
    """Serializes the full Settings object to TOML."""

    return tomli_w.dumps(dump_settings(settings, exclude_none=True))


def generate_requirements_txt() -> str:
    """Collects direct project dependencies as a formatted string."""

    requirements = [
        f"{package}=={version}" for package, version in get_requirements_dict().items()
    ]
    return "\n".join(requirements) + "\n"


def format_hf_link(
    path: str,
    commit: str | None = None,
    is_dataset: bool = False,
) -> str:
    prefix = "datasets/" if is_dataset else ""
    base_url = f"https://huggingface.co/{prefix}{path}"
    link = f"[{path}]({base_url})"

    if commit:
        commit_url = f"{base_url}/commit/{commit}"
        link += f" (Commit: [`{commit[:7]}`]({commit_url}))"

    return link


def generate_reproduce_readme(
    settings: Settings,
    dataset_specifications: list[DatasetSpecification],
    checkpoint_filename: str,
    trial: Trial | FrozenTrial,
    include_system_information: bool,
) -> str:
    """Generates the contents of a README.md for the reproduce/ folder."""

    heterogeneous_warning = ""

    if include_system_information:
        cpu = get_cpu_info_dict()
        python_env = get_python_env_info_dict()

        accelerators = get_accelerator_info_dict()
        if accelerators["type"] is None:
            accelerator_report = "**No TPU or other accelerator detected.**"
        else:
            devices = accelerators["devices"]

            if len({device["name"] for device in devices}) > 1:
                heterogeneous_warning = """
> [!WARNING]
> **Heterogeneous accelerators**
>
> This model was generated using multiple non-identical accelerators. When operations are distributed across different devices
> (e.g. via tensor parallelism), non-deterministic behaviour can occur.
>
> Reproducibility *cannot* be guaranteed in this environment.
"""

            total_memory = sum(device.get("vram_gb", 0) for device in devices)
            memory_suffix = (
                f" ({total_memory:.2f} GB total HBM)" if total_memory > 0 else ""
            )
            accelerator_lines = [
                f"- **{accelerators['type']}:** Detected {len(devices)} device(s){memory_suffix}"
            ]

            if accelerators.get("api_name") and accelerators.get("api_version"):
                accelerator_lines.append(
                    f"  - **{accelerators['api_name']}:** {accelerators['api_version']}"
                )

            if accelerators.get("driver_version"):
                accelerator_lines.append(
                    f"  - **Driver Version:** {accelerators['driver_version']}"
                )

            accelerator_lines.append("- **Devices:**")
            for i, device in enumerate(devices):
                memory = (
                    f" ({device['vram_gb']:.2f} GB HBM)"
                    if device.get("vram_gb")
                    else ""
                )
                accelerator_lines.append(
                    f"  - **{accelerators['type']} {i}:** {device['name']}{memory}"
                )
            accelerator_report = "\n".join(accelerator_lines)

        system_report = f"""## System

- **Python:** {python_env["version"]} ({python_env["implementation"]}, {python_env["compiler"]}) [{python_env["environment"]}]
- **Operating system:** {platform.platform()} ({platform.machine()})
- **CPU:** {cpu["brand"] or "Unknown"}

### Accelerators

{accelerator_report}

"""
        system_instructions = (
            "1. Ensure your system matches the specifications in the **System** section above. "
            "Exact reproducibility is only guaranteed if all aspects of your system are identical to the one the model was originally generated on.\n"
        )
    else:
        system_report = ""
        system_instructions = ""

    version_info = get_heretic_version_info()
    origin_warning = ""
    if not version_info.is_standard_pypi:
        if version_info.origin and version_info.origin.startswith("Git"):
            repo_info = version_info.origin.split("Git (")[1].rstrip(")")
            origin_warning = f"""
> [!IMPORTANT]
> **Git installation**
>
> This system installed heretic-tpu from a Git repository: {repo_info}
>
> To reproduce the model, you must install heretic-tpu from this exact repository and commit.
"""
        elif version_info.origin == "Local":
            origin_warning = """
> [!WARNING]
> **Local code**
>
> This system installed heretic-tpu from a local directory or wheel. Uncommitted or experimental code may have been executed.
>
> Reproducibility *cannot* be guaranteed in this environment.
"""
        else:
            origin_warning = """
> [!WARNING]
> **Non-standard installation**
>
> This system installed heretic-tpu from an unknown non-standard source.
>
> Reproducibility *cannot* be guaranteed in this environment.
"""

    libtpu_version = get_libtpu_version()

    formatted_datasets = set()
    for specification in dataset_specifications:
        if isinstance(specification, SingleDatasetSpecification):
            formatted_datasets.add(
                format_hf_link(
                    specification.dataset,
                    specification.commit,
                    is_dataset=True,
                )
            )
        else:
            for single_specification in specification:
                formatted_datasets.add(
                    format_hf_link(
                        single_specification.dataset,
                        single_specification.commit,
                        is_dataset=True,
                    )
                )
    dataset_lines = "\n".join(
        f"- {formatted_dataset}" for formatted_dataset in sorted(formatted_datasets)
    )

    trial_scores = trial.user_attrs["scores"]
    score_lines = "\n".join(
        (
            f"- **{score['name']}:** {score['score']['md_display']}"
            f" (baseline: {score['baseline']['md_display']})"
        )
        for score in trial_scores
    )

    return f"""# Reproduction guide

This directory contains the necessary information and assets to reproduce the results obtained during this heretic-tpu run.{heterogeneous_warning}{origin_warning}

## Models

- **Base model:** {format_hf_link(settings.model, settings.model_commit)}

## Datasets

{dataset_lines}

## Selected trial

- **Trial number:** {trial.user_attrs["index"]}
{score_lines}

{system_report}## Environment

- **heretic-tpu:** v{version_info.version}{f" (Origin: {version_info.origin})" if version_info.origin else ""}
- **JAX:** {get_package_version("jax")}
- **jaxlib:** {get_package_version("jaxlib")}
- **libtpu:** {libtpu_version or "not installed"}
- **Other dependencies:** See [`requirements.txt`](requirements.txt).

## Contents of this directory

- [`requirements.txt`](requirements.txt): The exact versions of all Python packages.
- [`config.toml`](config.toml): The exact configuration used, including the RNG seed.
- [`{checkpoint_filename}`]({checkpoint_filename}): The Optuna study journal containing the history of all trials.
- [`SHA256SUMS`](SHA256SUMS): Cryptographic hashes for all weight files.
- [`reproduce.json`](reproduce.json): A machine-readable file containing all reproducibility information.

## How to reproduce

> [!TIP]
> You can automate this process, including all verification steps, by downloading the `reproduce.json` file and running
> `heretic-tpu --reproduce reproduce.json`.

{system_instructions}1. Install the exact version of heretic-tpu indicated in the **Environment** section above, from its original source.
1. Install the packages listed in `requirements.txt`, which pins the exact versions of JAX, jaxlib and (on TPU) libtpu: `pip install -r requirements.txt`
1. Place the provided `config.toml` in your working directory.
1. Run heretic-tpu without any additional arguments: `heretic-tpu`
1. Wait for the run to finish, then select trial **{trial.user_attrs["index"]}** and export the model.
1. Verify that the weight files have been exactly reproduced by comparing their SHA-256 hashes against those in `SHA256SUMS`:
   `sha256sum -c SHA256SUMS` (or look at the hashes online if you uploaded to Hugging Face)

> [!TIP]
> To use the included Optuna study journal `{checkpoint_filename}`, place it in the checkpoints directory (usually `checkpoints/`) before running heretic-tpu.
>
> This allows you to export other models from the Pareto front, or to run additional trials without having to re-run the stored trials.
"""


def generate_reproduce_json(
    settings: Settings,
    trial: Trial | FrozenTrial,
    timestamp: str,
    uploaded_model_hashes: dict[str, str],
    include_system_information: bool,
) -> str:
    """Generates the contents of a reproduce.json file for the reproduce/ folder."""

    version_info = get_heretic_version_info()

    data = {
        # Version 4: plugin-based schema with generic parameters and scores.
        "version": "4",
        "timestamp": timestamp,
        "system": None,  # Defined here to preserve insertion order.
        "environment": {
            # Upstream Heretic's reader requires the "heretic" and "pytorch_version" keys,
            # so heretic-tpu stores its own version information under "heretic" and
            # leaves "pytorch_version" empty. "backend" tells the two programs' files apart.
            "heretic": {
                "version": version_info.version,
                "is_standard_pypi": version_info.is_standard_pypi,
                "metadata": version_info.metadata,
            },
            "pytorch_version": None,
            "backend": "jax",
            "jax_version": get_package_version("jax"),
            "jaxlib_version": get_package_version("jaxlib"),
            "libtpu_version": get_libtpu_version(),
            "requirements": get_requirements_dict(),
        },
        "settings": dump_settings(settings),
        "parameters": trial.user_attrs["parameters"],
        "scores": trial.user_attrs["scores"],
        "hashes": uploaded_model_hashes,
    }

    if include_system_information:
        data["system"] = {
            "python": get_python_env_info_dict(),
            "os": {
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "cpu": get_cpu_info_dict(),
            "accelerators": get_accelerator_info_dict(),
        }
    else:
        del data["system"]

    return json.dumps(data, indent=4)


def generate_sha256sums(hashes: dict[str, str]) -> str:
    """Generates GNU Coreutils compatible SHA256SUMS file content."""

    lines = []

    for filename, sha256 in sorted(hashes.items()):
        # Use '*' to indicate binary mode for model weights.
        lines.append(f"{sha256} *{filename}")

    return "\n".join(lines) + "\n"


# TODO: Replace this with hashlib.file_digest when we drop support for Python 3.10.
def get_file_sha256(file_path: str | Path) -> str:
    hash = hashlib.sha256()

    with open(file_path, "rb") as file:
        # Read the file in 64 kB blocks.
        for block in iter(lambda: file.read(65536), b""):
            hash.update(block)

    return hash.hexdigest()


def create_reproduce_folder(
    path: Path,
    settings: Settings,
    dataset_specifications: list[DatasetSpecification],
    checkpoint_path: str | Path,
    trial: Trial | FrozenTrial,
    uploaded_model_hashes: dict[str, str],
    include_system_information: bool,
    model_commit: str | None,
):
    """
    Writes the reproducibility files to `path`/reproduce. `model_commit` is the full
    commit hash of the base model that was loaded; if it is None, the commit that
    settings.model_commit (or the default branch) currently resolves to is used.
    """

    reproduce_dir = path / "reproduce"
    reproduce_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_filename = Path(checkpoint_path).name

    # Upstream records the current head of the default branch here, which is not
    # the commit that was loaded if settings.model_commit pins another one or the
    # repository has received commits since. It also overwrites the live settings,
    # from which adapters saved later in the session take their base revision.
    # The files are therefore written from a copy that names the loaded commit.
    if model_commit is None:
        model_commit = huggingface_hub.model_info(
            settings.model,
            revision=settings.model_commit,
        ).sha
    settings = settings.model_copy(update={"model_commit": model_commit})

    # Strip microseconds and timezone for a clean format.
    timestamp = datetime.now(UTC).replace(microsecond=0, tzinfo=None).isoformat()

    (reproduce_dir / "requirements.txt").write_text(
        generate_requirements_txt(),
        encoding="utf-8",
    )

    (reproduce_dir / "config.toml").write_text(
        generate_config_toml(settings),
        encoding="utf-8",
    )

    if uploaded_model_hashes:
        (reproduce_dir / "SHA256SUMS").write_text(
            generate_sha256sums(uploaded_model_hashes),
            encoding="utf-8",
        )

    (reproduce_dir / "reproduce.json").write_text(
        generate_reproduce_json(
            settings,
            trial,
            timestamp=timestamp,
            uploaded_model_hashes=uploaded_model_hashes,
            include_system_information=include_system_information,
        ),
        encoding="utf-8",
    )

    (reproduce_dir / "README.md").write_text(
        generate_reproduce_readme(
            settings,
            dataset_specifications,
            checkpoint_filename,
            trial,
            include_system_information=include_system_information,
        ),
        encoding="utf-8",
    )

    # Copy Optuna study journal.
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.exists():
        (reproduce_dir / checkpoint_file.name).write_bytes(checkpoint_file.read_bytes())


def upload_reproduce_folder(
    repo_id: str,
    settings: Settings,
    dataset_specifications: list[DatasetSpecification],
    token: str,
    checkpoint_path: str | Path,
    trial: Trial | FrozenTrial,
    include_system_information: bool,
    model_commit: str | None,
):
    api = huggingface_hub.HfApi()
    info = api.model_info(repo_id=repo_id, files_metadata=True, token=token)

    if not info.siblings:
        raise RuntimeError("Could not fetch uploaded model hashes.")

    # For weights, we only care about safetensors.
    weight_extensions = (".safetensors",)

    uploaded_model_hashes = {}

    for file in info.siblings:
        if file.rfilename.endswith(weight_extensions):
            sha256 = getattr(file, "lfs", {}).get("sha256")
            if not sha256:
                raise RuntimeError("Could not fetch uploaded model hashes.")
            uploaded_model_hashes[file.rfilename] = sha256

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        create_reproduce_folder(
            tmp_path,
            settings,
            dataset_specifications,
            checkpoint_path=checkpoint_path,
            trial=trial,
            uploaded_model_hashes=uploaded_model_hashes,
            include_system_information=include_system_information,
            model_commit=model_commit,
        )

        reproduce_dir = tmp_path / "reproduce"
        for file_path in reproduce_dir.iterdir():
            if file_path.is_file():
                huggingface_hub.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=f"reproduce/{file_path.name}",
                    repo_id=repo_id,
                    token=token,
                )
