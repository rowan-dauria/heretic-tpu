# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import importlib
import importlib.metadata
import json
import tomllib
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from optuna.trial import FrozenTrial, create_trial

from heretic_tpu import reproduce, system, utils
from heretic_tpu.config import Settings, SingleDatasetSpecification
from heretic_tpu.reproduce import (
    MismatchSeverity,
    check_environment,
    format_version_information,
    get_package_mismatch_severity,
)
from heretic_tpu.system import (
    HereticVersionInfo,
    get_heretic_version_info,
    get_libtpu_version,
)
from heretic_tpu.utils import generate_reproduce_json, generate_reproduce_readme

UPSTREAM_SOURCE = Path(__file__).parents[1] / "heretic" / "src"

# Every key of a reproduce.json file that upstream Heretic's reader
# (main.py and check_environment) uses.
UPSTREAM_READER_KEYS = [
    "version",
    "settings",
    "parameters",
    "scores",
    "hashes",
    "environment.heretic.version",
    "environment.heretic.metadata",
    "environment.pytorch_version",
    "environment.requirements",
    "system.python.version",
    "system.os.platform",
    "system.cpu.brand",
    "system.accelerators.type",
]

# Every key that upstream's reader uses when the accelerator types match.
UPSTREAM_ACCELERATOR_KEYS = ["api_name", "api_version", "driver_version", "devices"]


pytestmark = pytest.mark.usefixtures("isolated_settings_sources")


@pytest.fixture
def pypi_installation(monkeypatch: pytest.MonkeyPatch):
    """
    Pretends that heretic-tpu was installed from PyPI. A local installation is
    given a random version suffix, so it never matches itself.
    """
    version_info = HereticVersionInfo(
        version="0.1.0",
        origin="PyPI",
        is_standard_pypi=True,
        metadata={"type": "pypi"},
    )

    for module in [system, utils, reproduce]:
        monkeypatch.setattr(module, "get_heretic_version_info", lambda: version_info)


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Upstream Heretic's own modules, imported from the submodule."""
    pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    if not (UPSTREAM_SOURCE / "heretic" / "reproduce.py").exists():
        pytest.skip("the upstream submodule is not checked out")

    monkeypatch.syspath_prepend(str(UPSTREAM_SOURCE))
    upstream_config = importlib.import_module("heretic.config")
    upstream_system = importlib.import_module("heretic.system")
    upstream_utils = importlib.import_module("heretic.utils")
    upstream_reproduce = importlib.import_module("heretic.reproduce")

    # heretic-llm is not installed, so its version information is made up.
    version_info = upstream_system.HereticVersionInfo(
        version="1.5.0",
        origin="PyPI",
        is_standard_pypi=True,
        metadata={"type": "pypi"},
    )

    for module in [upstream_system, upstream_utils, upstream_reproduce]:
        monkeypatch.setattr(module, "get_heretic_version_info", lambda: version_info)

    return SimpleNamespace(
        config=upstream_config,
        utils=upstream_utils,
        reproduce=upstream_reproduce,
    )


def make_trial() -> FrozenTrial:
    return create_trial(
        values=[0.03, 0.12],
        user_attrs={
            "index": 7,
            "parameters": {"start_layer_index": 3, "steer_bad_behavior_weight": 0.0123},
            "scores": [
                {
                    "name": "Refusals",
                    "score": {"value": 3, "rich_display": "3", "md_display": "3/100"},
                    "baseline": {
                        "value": 97,
                        "rich_display": "97",
                        "md_display": "97/100",
                    },
                },
            ],
        },
    )


def make_settings(**values: Any) -> Settings:
    return Settings.model_validate(
        {
            "model": "Qwen/Qwen3-0.6B",
            "model_commit": "c1899de289a04d12100db370d81485cdf75e47ca",
            "seed": 1234,
            **values,
        }
    )


def make_reproduce_json(include_system_information: bool = True) -> dict[str, Any]:
    return json.loads(
        generate_reproduce_json(
            make_settings(),
            make_trial(),
            timestamp="2026-10-01T12:00:00",
            uploaded_model_hashes={"model.safetensors": "0" * 64},
            include_system_information=include_system_information,
        )
    )


def make_reproduce_readme(include_system_information: bool = True) -> str:
    return generate_reproduce_readme(
        make_settings(),
        [
            SingleDatasetSpecification(
                dataset="mlabonne/harmful_behaviors",
                commit="d1c6a0c1c1b5fb4a7fbbc1fbbd51b7ee1de5bd1f",
                split="test[:100]",
                column="text",
            ),
        ],
        "Qwen--Qwen3-0.6B.jsonl",
        make_trial(),
        include_system_information=include_system_information,
    )


def get_key(data: dict[str, Any], path: str) -> Any:
    for key in path.split("."):
        data = data[key]
    return data


def assert_no_upstream_branding(text: str):
    assert "heretic-llm" not in text
    assert "torch==" not in text
    assert "heretic --reproduce" not in text


@pytest.mark.parametrize("include_system_information", [True, False])
def test_generate_reproduce_readme(include_system_information: bool) -> None:
    readme = make_reproduce_readme(include_system_information)

    assert_no_upstream_branding(readme)
    assert "PyTorch" not in readme
    assert "`heretic-tpu --reproduce reproduce.json`" in readme
    assert "`pip install -r requirements.txt`" in readme
    assert "without any additional arguments: `heretic-tpu`" in readme
    assert f"- **JAX:** {system.get_package_version('jax')}\n" in readme
    assert f"- **jaxlib:** {system.get_package_version('jaxlib')}\n" in readme
    assert f"- **libtpu:** {get_libtpu_version() or 'not installed'}\n" in readme
    assert ("## System" in readme) == include_system_information


def test_generate_reproduce_readme_on_tpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        utils,
        "get_accelerator_info_dict",
        lambda: {
            "type": "TPU",
            "api_name": "libtpu",
            "api_version": "0.0.48",
            "driver_version": None,
            "devices": [{"name": "TPU v6 lite", "vram_gb": 31.25}],
        },
    )
    monkeypatch.setattr(utils, "get_libtpu_version", lambda: "0.0.48")

    readme = make_reproduce_readme()

    assert_no_upstream_branding(readme)
    assert "- **libtpu:** 0.0.48\n" in readme
    assert (
        "- **TPU:** Detected 1 device(s) (31.25 GB total HBM)\n"
        "  - **libtpu:** 0.0.48\n"
        "- **Devices:**\n"
        "  - **TPU 0:** TPU v6 lite (31.25 GB HBM)\n"
    ) in readme
    assert "Heterogeneous" not in readme


def test_generate_reproduce_readme_warns_about_mixed_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        utils,
        "get_accelerator_info_dict",
        lambda: {
            "type": "TPU",
            "api_name": "libtpu",
            "api_version": None,
            "driver_version": None,
            "devices": [{"name": "TPU v5 lite"}, {"name": "TPU v6 lite"}],
        },
    )

    assert "**Heterogeneous accelerators**" in make_reproduce_readme()


LOADED_COMMIT = "699cc3f2b1e4c7d5a8f9e0b1c2d3e4f5a6b7c8d9"
HEAD_COMMIT = "84ad45b0f1e2d3c4b5a69788796a5b4c3d2e1f00"


def write_reproduce_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    model_commit: str | None,
) -> tuple[Path, list[dict[str, Any]]]:
    """
    Calls utils.create_reproduce_folder with a Hub whose default branch is at
    HEAD_COMMIT and on which every other revision resolves to LOADED_COMMIT.
    Returns the folder and the arguments of every model_info call.
    """
    calls: list[dict[str, Any]] = []

    def model_info(repo_id: str, revision: str | None = None) -> SimpleNamespace:
        calls.append({"repo_id": repo_id, "revision": revision})
        return SimpleNamespace(sha=HEAD_COMMIT if revision is None else LOADED_COMMIT)

    monkeypatch.setattr(utils.huggingface_hub, "model_info", model_info)

    journal = tmp_path / "Qwen--Qwen3-0.6B.jsonl"
    journal.write_text("{}\n", encoding="utf-8")

    utils.create_reproduce_folder(
        tmp_path,
        settings,
        [],
        checkpoint_path=journal,
        trial=make_trial(),
        uploaded_model_hashes={"model.safetensors": "0" * 64},
        include_system_information=False,
        model_commit=model_commit,
    )

    return tmp_path / "reproduce", calls


def assert_reproduce_folder_records(reproduce_dir: Path, commit: str) -> None:
    data = json.loads((reproduce_dir / "reproduce.json").read_text(encoding="utf-8"))
    assert data["settings"]["model_commit"] == commit

    config = tomllib.loads((reproduce_dir / "config.toml").read_text(encoding="utf-8"))
    assert config["model_commit"] == commit

    readme = (reproduce_dir / "README.md").read_text(encoding="utf-8")
    assert f"/commit/{commit})" in readme


def test_create_reproduce_folder_records_the_loaded_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The model was loaded from a branch that resolved to LOADED_COMMIT, while the
    # default branch has moved on to HEAD_COMMIT.
    settings = make_settings(model_commit="v1.0")

    reproduce_dir, calls = write_reproduce_folder(
        tmp_path, monkeypatch, settings, LOADED_COMMIT
    )

    assert_reproduce_folder_records(reproduce_dir, LOADED_COMMIT)
    assert calls == []
    # The live settings, from which adapters record their base model revision,
    # are left alone.
    assert settings.model_commit == "v1.0"


@pytest.mark.parametrize("model_commit", [None, "699cc3f"])
def test_create_reproduce_folder_resolves_the_pinned_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_commit: str | None,
) -> None:
    settings = make_settings(model_commit=model_commit)

    reproduce_dir, calls = write_reproduce_folder(tmp_path, monkeypatch, settings, None)

    commit = HEAD_COMMIT if model_commit is None else LOADED_COMMIT
    assert_reproduce_folder_records(reproduce_dir, commit)
    assert calls == [{"repo_id": "Qwen/Qwen3-0.6B", "revision": model_commit}]
    assert settings.model_commit == model_commit


@pytest.mark.parametrize("include_system_information", [True, False])
def test_generate_reproduce_json(include_system_information: bool) -> None:
    data = make_reproduce_json(include_system_information)

    assert_no_upstream_branding(json.dumps(data))
    assert data["version"] == "4"

    environment = data["environment"]
    assert environment["pytorch_version"] is None
    assert environment["backend"] == "jax"
    assert environment["jax_version"] == system.get_package_version("jax")
    assert environment["jaxlib_version"] == system.get_package_version("jaxlib")
    assert environment["libtpu_version"] == get_libtpu_version()
    assert environment["requirements"]["jax"] == environment["jax_version"]
    assert set(environment["heretic"]) == {"version", "is_standard_pypi", "metadata"}

    assert ("system" in data) == include_system_information


def test_reproduce_json_has_every_key_upstream_reads() -> None:
    data = make_reproduce_json()

    for path in UPSTREAM_READER_KEYS:
        get_key(data, path)

    # On CPU, the accelerator type is None, so the other accelerator keys are not read.
    accelerators = data["system"]["accelerators"]
    if accelerators["type"] is not None:
        assert set(UPSTREAM_ACCELERATOR_KEYS) <= accelerators.keys()

    settings = data["settings"]
    assert (
        settings["scorers"][0]["plugin"] == "heretic.scorers.keyword_rate.KeywordRate"
    )
    assert settings["scorers"][1]["plugin"] == (
        "heretic.scorers.kl_divergence.KLDivergence"
    )
    assert settings["modifiers"][0]["plugin"] == "heretic.modifiers.ara.ARA"
    assert settings["parallelism"] == "auto"
    for key in ["device_map", "max_memory", "compilation_cache_dir"]:
        assert key not in settings


@pytest.mark.usefixtures("pypi_installation")
def test_check_environment_accepts_own_reproduce_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = make_reproduce_json()

    # With nothing to confirm, no question is asked, whatever ignore_mismatches says.
    assert check_environment(make_settings(ignore_mismatches=False), data) is True
    assert "differ" not in capsys.readouterr().out

    # The stored settings validate, and their plugin names resolve to the port's plugins.
    settings = Settings.model_validate(data["settings"])
    assert settings.seed == 1234
    assert capsys.readouterr().out == ""


class ArchiveInstallation:
    """The installed heretic-tpu distribution, as if installed from an archive URL."""

    def __init__(self, distribution: importlib.metadata.Distribution):
        self.distribution = distribution

    def __getattr__(self, name: str) -> Any:
        return getattr(self.distribution, name)

    def read_text(self, filename: str) -> str | None:
        if filename == "direct_url.json":
            # As `pip install https://.../archive/main.zip` records it.
            return json.dumps(
                {"url": "https://example.com/main.zip", "archive_info": {}}
            )
        return self.distribution.read_text(filename)


@pytest.mark.parametrize("metadata", [None, {"type": "unknown"}])
def test_check_environment_accepts_unknown_installation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    metadata: dict[str, Any] | None,
) -> None:
    distribution = importlib.metadata.distribution

    def fake_distribution(name: str) -> Any:
        if name == "heretic-tpu":
            return ArchiveInstallation(distribution(name))
        return distribution(name)

    monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)

    data = make_reproduce_json()
    if metadata is not None:
        # Upstream Heretic and earlier versions of heretic-tpu record this.
        data["environment"]["heretic"]["metadata"] = metadata

    # Like a local installation, an installation of unknown origin cannot be told
    # apart from another one, even from itself, so the mismatch has to be confirmed.
    assert check_environment(make_settings(ignore_mismatches=True), data) is True
    mismatches = capsys.readouterr().out
    assert "heretic-tpu" in mismatches
    assert "critical chance" in mismatches

    assert check_environment(make_settings(ignore_mismatches=False), data) is False

    version = data["environment"]["heretic"]["version"]
    for version_information in [
        data["environment"]["heretic"],
        asdict(get_heretic_version_info()),
    ]:
        assert format_version_information(version_information).startswith(
            f"{version}-unknown-"
        )


def upstream_reproduce_json() -> dict[str, Any]:
    # A reproduce.json written by upstream Heretic 1.5 on a CUDA machine.
    return {
        "version": "4",
        "timestamp": "2026-09-01T12:00:00",
        "system": {
            "python": {
                "version": "3.12.3",
                "implementation": "CPython",
                "compiler": "GCC 13.2.0",
                "environment": "Virtualenv/Venv",
            },
            "os": {
                "platform": "Linux-6.8.0-x86_64-with-glibc2.39",
                "machine": "x86_64",
            },
            "cpu": {
                "brand": "AMD EPYC 7742 64-Core Processor",
                "vendor": "AuthenticAMD",
                "family": 23,
                "model": 49,
                "stepping": 0,
            },
            "accelerators": {
                "type": "CUDA",
                "api_name": "CUDA Version",
                "api_version": "12.8",
                "driver_version": "570.86.15",
                "devices": [{"name": "NVIDIA A100-SXM4-80GB", "vram_gb": 79.25}],
            },
        },
        "environment": {
            "heretic": {
                "version": "1.5.0",
                "is_standard_pypi": True,
                "metadata": {"type": "pypi"},
            },
            "pytorch_version": "2.9.1+cu128",
            "requirements": {
                "accelerate": "1.12.0",
                "heretic-llm": "1.5.0",
                "torch": "2.9.1",
                "transformers": "5.6.0",
            },
        },
        "settings": {
            "model": "Qwen/Qwen3-0.6B",
            "model_commit": "c1899de289a04d12100db370d81485cdf75e47ca",
            "dtypes": ["auto", "float16", "bfloat16", "float32"],
            "quantization": "none",
            "device_map": "auto",
            "max_memory": None,
            "batch_size": 64,
            "seed": 1234,
            "scorers": [
                {
                    "plugin": "heretic.scorers.keyword_rate.KeywordRate",
                    "optimization": "minimize",
                    "instance_name": None,
                },
            ],
            "modifiers": [
                {"plugin": "heretic.modifiers.ara.ARA", "instance_name": None},
            ],
        },
        "parameters": {"start_layer_index": 3},
        "scores": [],
        "hashes": {"model.safetensors": "0" * 64},
    }


def check_upstream_reproduce_json(
    data: dict[str, Any],
    output: pytest.CaptureFixture[str],
) -> str:
    """Checks that the port reads an upstream file, and returns the mismatch report."""
    assert check_environment(make_settings(ignore_mismatches=True), data) is True

    mismatches = output.readouterr().out
    for name in ["heretic-llm", "heretic-tpu", "torch"]:
        assert name in mismatches
    assert "critical chance" in mismatches

    settings = Settings.model_validate(data["settings"])

    assert "Warning: Ignoring device_map and max_memory" in output.readouterr().out
    assert settings.seed == 1234
    # "float16" entries are kept, and are mapped to bfloat16 when loading the model.
    assert "float16" in settings.dtypes
    assert settings.scorers[0].plugin == "heretic.scorers.keyword_rate.KeywordRate"
    assert not settings.model_extra

    return mismatches


def test_check_environment_accepts_upstream_reproduce_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mismatches = check_upstream_reproduce_json(upstream_reproduce_json(), capsys)

    assert "Accelerator type" in mismatches


def test_accepts_reproduce_json_written_by_upstream(
    upstream: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    upstream_settings = upstream.config.Settings.model_validate(
        {
            "model": "Qwen/Qwen3-0.6B",
            "model_commit": "c1899de289a04d12100db370d81485cdf75e47ca",
            "seed": 1234,
        }
    )
    data = json.loads(
        upstream.utils.generate_reproduce_json(
            upstream_settings,
            make_trial(),
            timestamp="2026-10-01T12:00:00",
            uploaded_model_hashes={"model.safetensors": "0" * 64},
            include_system_information=True,
        )
    )

    check_upstream_reproduce_json(data, capsys)


def test_upstream_accepts_reproduce_json(
    upstream: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = make_reproduce_json()

    upstream_settings = upstream.config.Settings.model_validate(
        {"model": "Qwen/Qwen3-0.6B", "ignore_mismatches": True}
    )
    assert upstream.reproduce.check_environment(upstream_settings, data) is True

    # This is how upstream's main.py restores the settings.
    restored = upstream.config.Settings.model_validate(data["settings"])
    assert restored.seed == 1234

    # Upstream can import the stored plugins.
    for plugin_config in [*restored.scorers, *restored.modifiers]:
        module_name = plugin_config.plugin.rsplit(".", 1)[0]
        assert (UPSTREAM_SOURCE / f"{module_name.replace('.', '/')}.py").exists()


@pytest.mark.parametrize(
    ("package_name", "severity"),
    [
        ("heretic-tpu", MismatchSeverity.CRITICAL),
        ("heretic-llm", MismatchSeverity.CRITICAL),
        ("jax", MismatchSeverity.HIGH),
        ("jaxlib", MismatchSeverity.HIGH),
        ("libtpu", MismatchSeverity.HIGH),
        ("torch", MismatchSeverity.HIGH),
        ("transformers", MismatchSeverity.HIGH),
        ("accelerate", MismatchSeverity.MEDIUM),
        ("optuna", MismatchSeverity.MEDIUM),
        ("peft", MismatchSeverity.MEDIUM),
        ("numpy", MismatchSeverity.LOW),
    ],
)
def test_get_package_mismatch_severity(
    package_name: str,
    severity: MismatchSeverity,
) -> None:
    assert get_package_mismatch_severity(package_name) == severity
