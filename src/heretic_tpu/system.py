# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import gc
import importlib.metadata
import json
import os
import platform
import re
import sys
from dataclasses import dataclass
from typing import Any

import cpuinfo
import jax


def empty_cache():
    """Collects garbage, so that device buffers that are no longer referenced are freed."""

    # JAX frees an array's device memory as soon as the last reference to it dies,
    # and there is no allocator cache to empty, so collecting reference cycles
    # is all that is needed.
    gc.collect()


def configure_compilation_cache(directory: str):
    """
    Enables JAX's persistent compilation cache in the given directory,
    or disables it if the directory is empty.

    Must be called before anything is compiled, because JAX sets up the cache
    when it compiles for the first time.
    """

    jax.config.update("jax_compilation_cache_dir", directory or None)


@dataclass
class HereticVersionInfo:
    """Detailed information about the heretic-tpu installation."""

    version: str
    origin: str | None
    is_standard_pypi: bool
    metadata: dict[str, Any]


def get_heretic_version_info() -> HereticVersionInfo:
    """Detects version and installation source (PyPI, Git, Local) of heretic-tpu."""

    package_name = "heretic-tpu"
    # Installations from other sources (archive URLs, other version control systems)
    # get no "type", which format_version_information, here and in upstream Heretic,
    # reads as an unknown origin. Upstream writes "type": "unknown" instead, which
    # its reader rejects.
    origin_metadata: dict[str, Any] = {}
    # This package must be installed for this code to run.
    distribution = importlib.metadata.distribution(package_name)

    base_version = distribution.version.lstrip("v")

    try:
        direct_url_content = distribution.read_text("direct_url.json")
    # Unreadable installation metadata only means that the origin is unknown.
    except Exception:  # noqa: BLE001
        direct_url_content = None

    if not direct_url_content:
        # Standard PyPI installation.
        origin_metadata["type"] = "pypi"

        return HereticVersionInfo(
            version=base_version,
            origin="PyPI",
            is_standard_pypi=True,
            metadata=origin_metadata,
        )

    data = json.loads(direct_url_content)

    # Check for Git source.
    if "vcs_info" in data and data["vcs_info"].get("vcs") == "git":
        vcs_info = data["vcs_info"]
        commit_hash = vcs_info.get("commit_id", "unknown")
        repo_url = data.get("url", "unknown_repo")
        requested_revision = vcs_info.get("requested_revision")

        if requested_revision:
            origin_str = (
                f"Git ({repo_url}@{requested_revision} - commit: {commit_hash})"
            )
        else:
            origin_str = f"Git ({repo_url} @ {commit_hash})"

        origin_metadata.update(
            {
                "type": "git",
                "url": repo_url,
                "commit_hash": commit_hash,
                "requested_revision": requested_revision,
            }
        )

        return HereticVersionInfo(
            version=base_version,
            origin=origin_str,
            is_standard_pypi=False,
            metadata=origin_metadata,
        )

    # Check for local file/wheel directory.
    if "url" in data and data["url"].startswith("file://"):
        origin_metadata["type"] = "local"

        return HereticVersionInfo(
            version=base_version,
            origin="Local",
            is_standard_pypi=False,
            metadata=origin_metadata,
        )

    return HereticVersionInfo(
        version=base_version,
        origin=None,
        is_standard_pypi=False,
        metadata=origin_metadata,
    )


def get_libtpu_version() -> str | None:
    """Gets the installed libtpu version, or None if libtpu is not installed."""

    try:
        return get_package_version("libtpu")
    except importlib.metadata.PackageNotFoundError:
        return None


def get_accelerator_info_dict() -> dict[str, Any]:
    """Retrieves raw info about the local JAX devices (TPU, etc) directly into structured keys."""

    backend = jax.default_backend()

    if backend == "cpu":
        return {"type": None}

    devices = []

    for device in jax.local_devices():
        device_info: dict[str, Any] = {"name": device.device_kind}

        # Not every platform reports memory statistics.
        memory_stats = device.memory_stats()
        if memory_stats is not None and "bytes_limit" in memory_stats:
            device_info["vram_gb"] = round(memory_stats["bytes_limit"] / (1024**3), 2)

        devices.append(device_info)

    # The keys are those of upstream Heretic, whose reproduce.json reader uses them.
    # JAX exposes no driver versions.
    if backend == "tpu":
        return {
            "type": "TPU",
            "api_name": "libtpu",
            "api_version": get_libtpu_version(),
            "driver_version": None,
            "devices": devices,
        }

    return {
        "type": backend.upper(),
        "api_name": None,
        "api_version": None,
        "driver_version": None,
        "devices": devices,
    }


def get_accelerator_info(include_warnings: bool = True) -> str:
    """Convenience wrapper for hardware detection and console-friendly formatting."""

    info = get_accelerator_info_dict()

    if info["type"] is None:
        suffix = " Operations will be slow." if include_warnings else ""
        return f"[bold yellow]No TPU or other accelerator detected.{suffix}[/]"

    devices = info["devices"]
    total_memory = sum(device.get("vram_gb", 0) for device in devices)

    memory_suffix = f" ({total_memory:.2f} GB total HBM)" if total_memory > 0 else ""
    report = (
        f"Detected [bold]{len(devices)}[/] {info['type']} device(s){memory_suffix}\n"
    )

    if info["api_name"] and info["api_version"]:
        report += f"{info['api_name']}: [bold]{info['api_version']}[/]\n"

    for i, device in enumerate(devices):
        memory = f" ({device['vram_gb']:.2f} GB HBM)" if device.get("vram_gb") else ""
        report += f"* {info['type']} {i}: [bold]{device['name']}[/]{memory}\n"

    return report.strip()


def get_cpu_info_dict() -> dict[str, str | int | None]:
    """Gets granular CPU identifiers using the py-cpuinfo library."""

    info = cpuinfo.get_cpu_info()

    return {
        "brand": info.get("brand_raw"),
        "vendor": info.get("vendor_id_raw"),
        "family": info.get("family"),
        "model": info.get("model"),
        "stepping": info.get("stepping"),
    }


def get_cpu_info() -> str:
    """Gets the CPU brand name."""

    info = get_cpu_info_dict()
    parts = []
    parts.append(
        f"Family {info['family']}, Model {info['model']}, Stepping {info['stepping']}"
    )

    details = f" ({'; '.join(parts)})" if parts else ""
    brand = info["brand"] or "Unknown CPU"
    return f"{brand}{details}"


def get_python_env_info_dict() -> dict[str, str]:
    implementation = platform.python_implementation()
    compiler = platform.python_compiler()

    # Check for Conda.
    if "CONDA_PREFIX" in os.environ:
        env_type = "Conda"
    # Check for Virtualenv/Venv.
    elif hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix:
        env_type = "Virtualenv/Venv"
    else:
        env_type = "System"

    return {
        "version": platform.python_version(),
        "implementation": implementation,
        "compiler": compiler,
        "environment": env_type,
    }


def get_python_env_info() -> str:
    """Detects the type of Python environment (Conda, Venv, etc.) and build info."""

    info = get_python_env_info_dict()
    return f"{info['version']} ({info['implementation']}, {info['compiler']}) [{info['environment']}]"


def get_package_version(name: str) -> str:
    """Gets the installed version of a package, stripping local suffixes like +cu128."""

    # Normalize name: pip considers hyphens and underscores equivalent.
    normalized_name = name.lower().replace("_", "-")
    version_str = importlib.metadata.version(normalized_name)
    return version_str.split("+")[0] if "+" in version_str else version_str


def get_requirements_dict() -> dict[str, str]:
    """Recursively finds all direct and transitive dependencies of heretic-tpu and core libraries."""

    # We start with heretic-tpu and the core compute libraries.
    # libtpu is a dependency of jax only through its "tpu" extra, which the walk
    # below skips, so it must be listed explicitly. Packages that are not installed
    # (such as libtpu on machines without a TPU) are skipped.
    packages_to_check = ["heretic-tpu", "jax", "jaxlib", "libtpu"]

    visited = set()
    required_packages = set()

    while packages_to_check:
        package = packages_to_check.pop(0)
        # Normalize name: pip considers hyphens and underscores equivalent.
        normalized_package = package.lower().replace("_", "-")
        if normalized_package in visited:
            continue
        visited.add(normalized_package)

        try:
            distribution = importlib.metadata.distribution(normalized_package)
            required_packages.add(normalized_package)
            if distribution.requires:
                for requirement in distribution.requires:
                    # Requirements can include environment markers like '; extra == "hf"'
                    # or version constraints. We should ignore optional 'extra' dependencies
                    # to keep the reproduction environment clean and relevant.
                    if ";" in requirement and "extra ==" in requirement:
                        continue

                    # We just want the base package name.
                    match = re.match(r"^([a-zA-Z0-9_\-]+)", requirement)
                    if match:
                        dep_name = match.group(0).lower().replace("_", "-")
                        if dep_name not in visited:
                            packages_to_check.append(dep_name)
        except importlib.metadata.PackageNotFoundError:
            # If a package is listed as a dependency but not installed, we skip it.
            continue

    required_packages_sorted = sorted(required_packages)

    # Lookup versions for all discovered packages.
    dependencies = {}
    version_info = get_heretic_version_info()

    for package in required_packages_sorted:
        # If heretic-tpu was installed from source (Git/Local), exclude it
        # from requirements.txt to prevent pip from downloading an unrelated
        # version from PyPI during reproduction.
        if package == "heretic-tpu" and not version_info.is_standard_pypi:
            continue

        dependencies[package] = get_package_version(package)

    return dependencies
