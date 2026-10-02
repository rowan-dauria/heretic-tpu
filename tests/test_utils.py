# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import tomllib
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import jax
import pytest
from optuna.trial import FrozenTrial, create_trial

from heretic_tpu.config import Settings
from heretic_tpu.utils import (
    dump_settings,
    generate_config_toml,
    get_readme_intro,
    get_upstream_plugin_name,
    print_memory_usage,
)

GiB = 1024**3


pytestmark = pytest.mark.usefixtures("isolated_settings_sources")


@dataclass
class FakeDevice:
    stats: dict[str, int] | None

    def memory_stats(self) -> dict[str, int] | None:
        return self.stats


class FakeModifier:
    modifier_name = "ARA"

    def render_trial_parameters(self, trial: FrozenTrial) -> dict[str, str]:
        return {"start_layer_index": "3", "steer_bad_behavior_weight": "0.0123"}


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
                {
                    "name": "KL divergence",
                    "score": {
                        "value": 0.12,
                        "rich_display": "0.12",
                        "md_display": "0.1200",
                    },
                    "baseline": {
                        "value": 0,
                        "rich_display": "0",
                        "md_display": "0 *(by definition)*",
                    },
                },
            ],
        },
    )


def test_print_memory_usage_on_cpu(capsys: pytest.CaptureFixture[str]) -> None:
    if jax.default_backend() != "cpu":
        pytest.skip("requires the cpu backend")

    print_memory_usage()

    output = capsys.readouterr().out
    assert output.startswith("Resident system RAM: ")
    assert "device memory" not in output


def test_print_memory_usage_sums_device_memory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    devices = [
        FakeDevice({"bytes_in_use": 1 * GiB, "peak_bytes_in_use": 2 * GiB}),
        FakeDevice({"bytes_in_use": 2 * GiB, "peak_bytes_in_use": 3 * GiB}),
    ]
    monkeypatch.setattr(jax, "local_devices", lambda: devices)

    print_memory_usage()

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("Resident system RAM: ")
    assert lines[1:] == [
        "Allocated device memory: 3.00 GB",
        "Peak allocated device memory: 5.00 GB",
    ]


@pytest.mark.parametrize(
    ("name", "upstream_name"),
    [
        (
            "heretic_tpu.scorers.keyword_rate.KeywordRate",
            "heretic.scorers.keyword_rate.KeywordRate",
        ),
        ("heretic_tpu.modifiers.ara.ARA", "heretic.modifiers.ara.ARA"),
        ("heretic.modifiers.ara.ARA", "heretic.modifiers.ara.ARA"),
        ("heretic_tpu_extras.Scorer", "heretic_tpu_extras.Scorer"),
        ("my_plugins.scorers.Scorer", "my_plugins.scorers.Scorer"),
        ("plugins/heretic_tpu.py:Scorer", "plugins/heretic_tpu.py:Scorer"),
    ],
)
def test_get_upstream_plugin_name(name: str, upstream_name: str) -> None:
    assert get_upstream_plugin_name(name) == upstream_name


def test_dump_settings_uses_upstream_plugin_names() -> None:
    settings = Settings.model_validate(
        {
            "model": "Qwen/Qwen3-0.6B",
            "scorers": [
                {
                    "plugin": "heretic_tpu.scorers.keyword_rate.KeywordRate",
                    "optimization": "minimize",
                },
                {"plugin": "plugins/scorer.py:Scorer", "optimization": "none"},
            ],
            "modifiers": [{"plugin": "heretic_tpu.modifiers.ara.ARA"}],
        }
    )

    data = dump_settings(settings)

    assert [scorer["plugin"] for scorer in data["scorers"]] == [
        "heretic.scorers.keyword_rate.KeywordRate",
        "plugins/scorer.py:Scorer",
    ]
    assert data["modifiers"][0]["plugin"] == "heretic.modifiers.ara.ARA"
    # The settings object itself is unchanged.
    assert settings.modifiers[0].plugin == "heretic_tpu.modifiers.ara.ARA"


def test_generate_config_toml(
    capsys: pytest.CaptureFixture[str],
    isolated_settings_sources: Path,
) -> None:
    settings = Settings.model_validate(
        {
            "model": "Qwen/Qwen3-0.6B",
            "model_commit": "c1899de289a04d12100db370d81485cdf75e47ca",
            "seed": 1234,
            "parallelism": "single",
            "device_map": "auto",
            "scorer": {"KeywordRate": {"score_name": "Refusals"}},
        }
    )

    config_toml = generate_config_toml(settings)
    config = tomllib.loads(config_toml)

    assert config["scorers"][0]["plugin"] == "heretic.scorers.keyword_rate.KeywordRate"
    assert config["modifiers"][0]["plugin"] == "heretic.modifiers.ara.ARA"
    assert config["parallelism"] == "single"
    assert config["scorer"] == {"KeywordRate": {"score_name": "Refusals"}}
    for key in ["device_map", "max_memory", "compilation_cache_dir"]:
        assert key not in config

    # The generated file reproduces the settings, without warnings.
    capsys.readouterr()
    (isolated_settings_sources / "config.toml").write_text(
        config_toml, encoding="utf-8"
    )
    restored = Settings()  # ty:ignore[missing-argument]

    assert capsys.readouterr().out == ""
    assert restored.seed == 1234
    assert restored.parallelism == "single"
    assert dump_settings(restored) == dump_settings(settings)


def get_readme_intro_for(model: str, contains_reproducibility_information: bool) -> str:
    return get_readme_intro(
        Settings(model=model),
        FakeModifier(),  # ty:ignore[invalid-argument-type]
        make_trial(),
        contains_reproducibility_information,
    )


def test_get_readme_intro() -> None:
    intro = get_readme_intro_for("Qwen/Qwen3-0.6B", True)

    assert intro.startswith(
        "# This is a decensored version of "
        "[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), "
        f"made using heretic-tpu v{version('heretic-tpu')}, "
        "a TPU port of [Heretic](https://heretic-project.org)\n"
    )
    assert "**This model is reproducible!**" in intro
    assert "## ARA parameters" in intro
    assert "| **steer_bad_behavior_weight** | 0.0123 |" in intro
    assert "| **Refusals** | 3/100 | 97/100 |" in intro
    assert "heretic-llm" not in intro


def test_get_readme_intro_hides_local_paths(tmp_path: Path) -> None:
    intro = get_readme_intro_for(str(tmp_path), False)

    assert intro.startswith("# This is a decensored version of a model, made using")
    assert str(tmp_path) not in intro
    assert "reproducible" not in intro


def test_get_readme_intro_accepts_restored_trial() -> None:
    # main.py rebuilds the trial from reproduce.json without values when reproducing.
    trial: Any = create_trial(values=[], user_attrs=make_trial().user_attrs)

    intro = get_readme_intro(
        Settings(model="Qwen/Qwen3-0.6B"),
        FakeModifier(),  # ty:ignore[invalid-argument-type]
        trial,
        False,
    )

    assert "| **KL divergence** | 0.1200 | 0 *(by definition)* |" in intro
