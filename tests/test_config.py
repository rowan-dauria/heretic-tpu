# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import os
import tomllib
import unittest
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from heretic_tpu.config import QuantizationMethod, ScorerConfig, Settings

REPOSITORY_ROOT = Path(__file__).parents[1]

UPSTREAM_DEFAULT_CONFIG = REPOSITORY_ROOT / "heretic" / "config.default.toml"


class ScorerConfigTests(unittest.TestCase):
    def test_accepts_slug_like_instance_name(self) -> None:
        config = ScorerConfig(
            plugin="heretic_tpu.scorers.keyword_rate.KeywordRate",
            optimization="minimize",
            instance_name="small-1",
        )

        self.assertEqual(config.instance_name, "small-1")

    def test_rejects_empty_instance_name(self) -> None:
        with self.assertRaises(ValidationError):
            ScorerConfig(
                plugin="heretic_tpu.scorers.keyword_rate.KeywordRate",
                optimization="minimize",
                instance_name=" \t",
            )

    def test_rejects_whitespace_in_instance_name(self) -> None:
        for instance_name in ["small name", "small\tname", "small\nname"]:
            with (
                self.subTest(instance_name=instance_name),
                self.assertRaisesRegex(ValidationError, "whitespace is not allowed"),
            ):
                ScorerConfig(
                    plugin="heretic_tpu.scorers.keyword_rate.KeywordRate",
                    optimization="minimize",
                    instance_name=instance_name,
                )

    def test_rejects_dot_in_instance_name(self) -> None:
        with self.assertRaisesRegex(ValidationError, "'\\.' is not allowed"):
            ScorerConfig(
                plugin="heretic_tpu.scorers.keyword_rate.KeywordRate",
                optimization="minimize",
                instance_name="small.name",
            )


pytestmark = pytest.mark.usefixtures("isolated_settings_sources")


def test_defaults() -> None:
    settings = Settings(model="Qwen/Qwen3-0.6B")

    assert settings.dtypes == ["auto", "bfloat16", "float32"]
    assert settings.quantization == QuantizationMethod.NONE
    assert settings.parallelism == "auto"
    assert settings.compilation_cache_dir == os.path.join(
        Path.home(), ".cache", "heretic-tpu", "xla"
    )
    assert "device_map" not in Settings.model_fields
    assert "max_memory" not in Settings.model_fields


def test_rejects_bnb_4bit_quantization() -> None:
    with pytest.raises(ValidationError) as error_info:
        Settings(model="Qwen/Qwen3-0.6B", quantization="bnb_4bit")

    [error] = error_info.value.errors()
    assert error["loc"] == ("quantization",)
    assert (
        "bitsandbytes quantisation requires CUDA and is unavailable on TPU"
        in error["msg"]
    )


def test_rejects_unknown_quantization() -> None:
    with pytest.raises(ValidationError, match="Input should be 'none'"):
        Settings(model="Qwen/Qwen3-0.6B", quantization="gptq")


def test_ignores_device_map_and_max_memory_in_restored_settings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The settings section of an upstream reproduce.json.
    settings = Settings.model_validate(
        {
            "model": "Qwen/Qwen3-0.6B",
            "dtypes": ["auto", "float16", "bfloat16", "float32"],
            "device_map": "auto",
            "max_memory": {"0": "20GB", "cpu": "64GB"},
            "n_trials": 42,
        }
    )

    output = capsys.readouterr().out
    assert "Warning: Ignoring device_map and max_memory" in output
    assert "parallelism" in output

    assert settings.n_trials == 42
    # "float16" entries are kept, and are mapped to bfloat16 when loading the model.
    assert settings.dtypes == ["auto", "float16", "bfloat16", "float32"]
    assert not settings.model_extra
    assert "device_map" not in settings.model_dump()
    assert "max_memory" not in settings.model_dump()


def test_ignores_device_map_in_study_checkpoint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Study checkpoints store the settings with model_dump_json().
    settings = Settings.model_validate_json(
        json.dumps(
            {
                "model": "Qwen/Qwen3-0.6B",
                "device_map": {"": 0},
                "batch_size": 64,
            }
        )
    )

    output = capsys.readouterr().out
    assert "Warning: Ignoring device_map," in output
    assert "max_memory" not in output

    assert settings.batch_size == 64
    assert not settings.model_extra


def test_ignores_device_map_and_max_memory_in_config_toml(
    capsys: pytest.CaptureFixture[str],
    isolated_settings_sources: Path,
) -> None:
    (isolated_settings_sources / "config.toml").write_text(
        'device_map = "auto"\nmax_memory = { "0" = "20GB" }\nmax_response_length = 50\n',
        encoding="utf-8",
    )

    settings = Settings(model="Qwen/Qwen3-0.6B")

    assert "Warning: Ignoring device_map and max_memory" in capsys.readouterr().out
    assert settings.max_response_length == 50
    assert not settings.model_extra


@pytest.mark.skipif(
    not UPSTREAM_DEFAULT_CONFIG.exists(),
    reason="upstream Heretic is not checked out in heretic/",
)
def test_accepts_upstream_default_config(
    capsys: pytest.CaptureFixture[str],
    isolated_settings_sources: Path,
) -> None:
    (isolated_settings_sources / "config.toml").write_text(
        UPSTREAM_DEFAULT_CONFIG.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    settings = Settings(model="Qwen/Qwen3-0.6B")

    assert "Warning: Ignoring device_map," in capsys.readouterr().out
    assert settings.dtypes == ["auto", "float16", "bfloat16", "float32"]
    assert settings.scorers[0].plugin == "heretic.scorers.keyword_rate.KeywordRate"
    assert set(settings.model_extra or {}) == {"scorer", "modifier"}


def test_no_warning_without_unsupported_settings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    Settings.model_validate({"model": "Qwen/Qwen3-0.6B", "batch_size": 8})

    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("parallelism", ["auto", "single", "tensor"])
def test_accepts_parallelism(parallelism: str) -> None:
    settings = Settings(model="Qwen/Qwen3-0.6B", parallelism=parallelism)

    assert settings.parallelism == parallelism
    assert settings.model_dump()["parallelism"] == parallelism


@pytest.mark.parametrize("parallelism", ["data", "pipeline", ""])
def test_rejects_unknown_parallelism(parallelism: str) -> None:
    with pytest.raises(ValidationError, match="parallelism"):
        Settings(model="Qwen/Qwen3-0.6B", parallelism=parallelism)


def test_reads_parallelism_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERETIC_PARALLELISM", "tensor")

    assert Settings(model="Qwen/Qwen3-0.6B").parallelism == "tensor"


def test_expands_compilation_cache_dir(tmp_path: Path) -> None:
    def cache_dir(value: str) -> str:
        return Settings(
            model="Qwen/Qwen3-0.6B",
            compilation_cache_dir=value,
        ).compilation_cache_dir

    assert cache_dir("~/xla") == os.path.join(Path.home(), "xla")
    assert cache_dir(str(tmp_path)) == str(tmp_path)
    # An empty string disables the cache.
    assert cache_dir("") == ""


def test_excludes_compilation_cache_dir_from_stored_settings() -> None:
    settings = Settings(model="Qwen/Qwen3-0.6B")

    assert "compilation_cache_dir" not in settings.model_dump()
    assert "compilation_cache_dir" not in json.loads(settings.model_dump_json())


def test_default_config_matches_settings() -> None:
    config = tomllib.loads(
        (REPOSITORY_ROOT / "config.default.toml").read_text(encoding="utf-8")
    )
    fields = Settings.model_fields

    # Everything else is a plugin table.
    assert {key for key in config if key not in fields} == {"scorer", "modifier"}

    assert {"parallelism", "compilation_cache_dir"} <= config.keys()
    assert not {"device_map", "max_memory"} & config.keys()

    for key, value in config.items():
        if key in fields:
            field = fields[key]
            assert TypeAdapter(field.annotation).validate_python(
                value
            ) == field.get_default(call_default_factory=True), key


@pytest.mark.parametrize(
    "file_name",
    [
        "config.default.toml",
        "config.noslop.toml",
        "config.nohumor.toml",
        "config.piqa.toml",
    ],
)
def test_example_configs_are_valid(
    file_name: str,
    capsys: pytest.CaptureFixture[str],
    isolated_settings_sources: Path,
) -> None:
    (isolated_settings_sources / "config.toml").write_text(
        (REPOSITORY_ROOT / file_name).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    Settings(model="Qwen/Qwen3-0.6B")

    assert capsys.readouterr().out == ""


def test_default_config_gives_default_settings(
    isolated_settings_sources: Path,
) -> None:
    (isolated_settings_sources / "config.toml").write_text(
        (REPOSITORY_ROOT / "config.default.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    from_file = Settings(model="Qwen/Qwen3-0.6B")

    (isolated_settings_sources / "config.toml").unlink()
    defaults = Settings(model="Qwen/Qwen3-0.6B")

    assert from_file.model_dump(exclude={"scorer", "modifier"}) == defaults.model_dump()
    assert from_file.compilation_cache_dir == defaults.compilation_cache_dir


if __name__ == "__main__":
    unittest.main()
