# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import gc
import importlib.metadata
import json
import os
import subprocess
import sys
import textwrap
import tomllib
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

from heretic_tpu import system
from heretic_tpu.config import Settings
from heretic_tpu.system import (
    configure_compilation_cache,
    empty_cache,
    get_accelerator_info,
    get_accelerator_info_dict,
    get_heretic_version_info,
    get_libtpu_version,
    get_requirements_dict,
)

GiB = 1024**3

PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


@dataclass
class FakeDevice:
    device_kind: str
    stats: dict[str, int] | None

    def memory_stats(self) -> dict[str, int] | None:
        return self.stats


def use_fake_devices(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    devices: list[FakeDevice],
):
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    monkeypatch.setattr(jax, "local_devices", lambda: devices)


def v6e_devices(count: int) -> list[FakeDevice]:
    return [
        FakeDevice("TPU v6 lite", {"bytes_limit": int(31.25 * GiB), "bytes_in_use": 0})
        for _ in range(count)
    ]


def require_backend(backend: str):
    if jax.default_backend() != backend:
        pytest.skip(f"requires the {backend} backend")


def test_empty_cache_collects_reference_cycles() -> None:
    class Node:
        pass

    # Automatic collection could free the cycle before empty_cache() does.
    gc.disable()
    try:
        node = Node()
        node.self = node  # ty:ignore[unresolved-attribute]
        reference = weakref.ref(node)
        del node
        assert reference() is not None

        empty_cache()
    finally:
        gc.enable()

    assert reference() is None


def test_accelerator_info_on_cpu() -> None:
    require_backend("cpu")

    assert get_accelerator_info_dict() == {"type": None}
    assert get_accelerator_info() == (
        "[bold yellow]No TPU or other accelerator detected. Operations will be slow.[/]"
    )
    assert get_accelerator_info(include_warnings=False) == (
        "[bold yellow]No TPU or other accelerator detected.[/]"
    )


def test_accelerator_info_on_tpu(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_devices(monkeypatch, "tpu", v6e_devices(4))
    monkeypatch.setattr(system, "get_libtpu_version", lambda: "0.0.48")

    assert get_accelerator_info_dict() == {
        "type": "TPU",
        "api_name": "libtpu",
        "api_version": "0.0.48",
        "driver_version": None,
        "devices": [{"name": "TPU v6 lite", "vram_gb": 31.25}] * 4,
    }

    assert get_accelerator_info().splitlines() == [
        "Detected [bold]4[/] TPU device(s) (125.00 GB total HBM)",
        "libtpu: [bold]0.0.48[/]",
        "* TPU 0: [bold]TPU v6 lite[/] (31.25 GB HBM)",
        "* TPU 1: [bold]TPU v6 lite[/] (31.25 GB HBM)",
        "* TPU 2: [bold]TPU v6 lite[/] (31.25 GB HBM)",
        "* TPU 3: [bold]TPU v6 lite[/] (31.25 GB HBM)",
    ]


def test_accelerator_info_on_tpu_without_libtpu_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake_devices(monkeypatch, "tpu", v6e_devices(1))
    monkeypatch.setattr(system, "get_libtpu_version", lambda: None)

    assert get_accelerator_info_dict()["api_version"] is None
    assert get_accelerator_info().splitlines() == [
        "Detected [bold]1[/] TPU device(s) (31.25 GB total HBM)",
        "* TPU 0: [bold]TPU v6 lite[/] (31.25 GB HBM)",
    ]


def test_accelerator_info_on_other_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    # Not every platform reports memory statistics.
    use_fake_devices(monkeypatch, "gpu", [FakeDevice("NVIDIA H100 80GB HBM3", None)])

    assert get_accelerator_info_dict() == {
        "type": "GPU",
        "api_name": None,
        "api_version": None,
        "driver_version": None,
        "devices": [{"name": "NVIDIA H100 80GB HBM3"}],
    }
    assert get_accelerator_info().splitlines() == [
        "Detected [bold]1[/] GPU device(s)",
        "* GPU 0: [bold]NVIDIA H100 80GB HBM3[/]",
    ]


@pytest.mark.tpu
def test_accelerator_info_on_real_tpu() -> None:
    require_backend("tpu")

    info = get_accelerator_info_dict()

    assert info["type"] == "TPU"
    assert info["api_version"] == get_libtpu_version()
    assert info["devices"]
    for device in info["devices"]:
        assert device["name"].startswith("TPU")
        assert device["vram_gb"] > 0


def test_libtpu_version(monkeypatch: pytest.MonkeyPatch) -> None:
    def not_installed(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", not_installed)
    assert get_libtpu_version() is None

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.0.48+nightly")
    assert get_libtpu_version() == "0.0.48"


def test_requirements_include_jax_but_not_pytorch() -> None:
    requirements = get_requirements_dict()

    assert requirements["jax"] == importlib.metadata.version("jax")
    assert requirements["jaxlib"] == importlib.metadata.version("jaxlib")
    # No runtime dependency of heretic-tpu requires PyTorch.
    assert "torch" not in requirements


class FakeDistribution:
    def __init__(self, version: str, requires: list[str] | None):
        self.version = version
        self.requires = requires


def test_requirements_include_installed_libtpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distribution = importlib.metadata.distribution
    version = importlib.metadata.version

    def fake_distribution(name: str) -> Any:
        if name == "jax":
            # libtpu is reachable only through the "tpu" extra, which the walk skips.
            return FakeDistribution(
                distribution("jax").version,
                ["jaxlib==0.11.2", 'libtpu==0.0.48; extra == "tpu"'],
            )
        if name == "libtpu":
            return FakeDistribution("0.0.48", None)
        return distribution(name)

    def fake_version(name: str) -> str:
        return "0.0.48" if name == "libtpu" else version(name)

    monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)
    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    assert get_requirements_dict()["libtpu"] == "0.0.48"


def test_requirements_skip_missing_libtpu(monkeypatch: pytest.MonkeyPatch) -> None:
    distribution = importlib.metadata.distribution

    def fake_distribution(name: str) -> Any:
        if name == "libtpu":
            raise importlib.metadata.PackageNotFoundError(name)
        return distribution(name)

    monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)

    requirements = get_requirements_dict()

    assert "libtpu" not in requirements
    assert "jax" in requirements


def test_dependency_bounds_exclude_untested_versions() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    requirements = {
        requirement.name: requirement
        for requirement in map(Requirement, project["dependencies"])
    }
    [tpu_requirement] = map(Requirement, project["optional-dependencies"]["tpu"])

    # jax 0.11 needs Python 3.12. Earlier jax releases compile abliteration with
    # more than one module in flight, and lm-eval releases before 0.4.13 lack
    # helpers that the JaxLM adapter imports.
    assert not SpecifierSet(project["requires-python"]).contains("3.11.9")
    for name, oldest_supported, newest_unsupported in [
        ("jax", "0.11.2", "0.11.1"),
        ("lm-eval", "0.4.13", "0.4.12"),
    ]:
        specifier = requirements[name].specifier
        assert specifier.contains(oldest_supported)
        assert not specifier.contains(newest_unsupported)
        assert specifier.contains(importlib.metadata.version(name))
    assert tpu_requirement.specifier == requirements["jax"].specifier


def test_dependencies_of_default_benchmarks_are_installed() -> None:
    # lm-eval lists the dependencies of some tasks only under an extra named after
    # the task, such as IFEval's langdetect and immutabledict, so heretic-tpu has
    # to depend on them itself.
    tasks = {
        benchmark.task
        for benchmark in Settings.model_fields["benchmarks"].get_default(
            call_default_factory=True
        )
    }
    extras = set(importlib.metadata.metadata("lm-eval").get_all("Provides-Extra") or [])
    assert "ifeval" in tasks & extras

    checked = set()
    for requirement in map(Requirement, importlib.metadata.requires("lm-eval") or []):
        marker = requirement.marker
        if marker is None or "extra" not in str(marker):
            continue
        for task in tasks & extras:
            if marker.evaluate({"extra": task}):
                version = importlib.metadata.version(requirement.name)
                assert requirement.specifier.contains(version, prereleases=True), (
                    requirement,
                    version,
                )
                checked.add(requirement.name)

    assert {"immutabledict", "langdetect"} <= checked


class FakeInstallation:
    """A heretic-tpu distribution with the given direct_url.json (None if absent)."""

    def __init__(self, direct_url: dict[str, Any] | None):
        self.version = "0.1.0"
        self.direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        if filename == "direct_url.json" and self.direct_url is not None:
            return json.dumps(self.direct_url)
        return None


@pytest.mark.parametrize(
    ("direct_url", "origin", "metadata"),
    [
        (None, "PyPI", {"type": "pypi"}),
        (
            {
                "url": "https://github.com/user/heretic-tpu.git",
                "vcs_info": {"vcs": "git", "commit_id": "abc123"},
            },
            "Git (https://github.com/user/heretic-tpu.git @ abc123)",
            {
                "type": "git",
                "url": "https://github.com/user/heretic-tpu.git",
                "commit_hash": "abc123",
                "requested_revision": None,
            },
        ),
        (
            {"url": "file:///home/user/heretic-tpu", "dir_info": {"editable": True}},
            "Local",
            {"type": "local"},
        ),
        # An archive, as `pip install https://.../archive/main.zip` records it.
        ({"url": "https://example.com/main.zip", "archive_info": {}}, None, {}),
        # Another version control system.
        (
            {
                "url": "https://example.com/heretic-tpu",
                "vcs_info": {"vcs": "hg", "commit_id": "abc123"},
            },
            None,
            {},
        ),
    ],
)
def test_heretic_version_info(
    monkeypatch: pytest.MonkeyPatch,
    direct_url: dict[str, Any] | None,
    origin: str | None,
    metadata: dict[str, Any],
) -> None:
    distribution = importlib.metadata.distribution

    def fake_distribution(name: str) -> Any:
        if name == "heretic-tpu":
            return FakeInstallation(direct_url)
        return distribution(name)

    monkeypatch.setattr(importlib.metadata, "distribution", fake_distribution)

    version_info = get_heretic_version_info()

    assert version_info.version == "0.1.0"
    assert version_info.origin == origin
    assert version_info.is_standard_pypi == (origin == "PyPI")
    # An unknown origin has no "type", which the readers of reproduce.json files,
    # here and in upstream Heretic, accept.
    assert version_info.metadata == metadata


def test_configure_compilation_cache_sets_and_clears_directory(
    tmp_path: Path,
) -> None:
    previous = jax.config.jax_compilation_cache_dir

    try:
        configure_compilation_cache(str(tmp_path))
        assert jax.config.jax_compilation_cache_dir == str(tmp_path)

        configure_compilation_cache("")
        assert jax.config.jax_compilation_cache_dir is None
    finally:
        jax.config.update("jax_compilation_cache_dir", previous)


@pytest.mark.parametrize("enabled", [True, False])
def test_configure_compilation_cache_persists_executables(
    enabled: bool,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"

    # JAX sets up the cache only once per process, at the first compilation,
    # so this runs in a fresh interpreter.
    script = textwrap.dedent(
        f"""
        import jax
        import jax.numpy as jnp

        from heretic_tpu.system import configure_compilation_cache

        configure_compilation_cache({str(cache_dir) if enabled else ""!r})
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.jit(lambda x: jnp.sin(x) @ x.T)(jnp.ones((8, 8))).block_until_ready()
        """
    )
    environment = {**os.environ, "JAX_PLATFORMS": "cpu"}
    environment.pop("JAX_COMPILATION_CACHE_DIR", None)

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=tmp_path,
        env=environment,
        timeout=120,
    )

    if enabled:
        assert any(cache_dir.iterdir())
    else:
        assert not cache_dir.exists()


def test_system_modules_do_not_import_pytorch(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        import sys

        import heretic_tpu.config
        import heretic_tpu.progress
        import heretic_tpu.reproduce
        import heretic_tpu.system
        import heretic_tpu.utils

        forbidden = {"torch", "accelerate", "peft", "bitsandbytes"}
        print(sorted(name for name in sys.modules if name.split(".")[0] in forbidden))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        cwd=tmp_path,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        text=True,
        timeout=120,
    )

    assert result.stdout.strip() == "[]"
