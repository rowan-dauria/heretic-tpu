# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import jax
import pytest


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    # Tests marked "tpu" only make sense on a TPU, so they are skipped elsewhere
    # rather than failing on a missing device.
    if jax.default_backend() == "tpu":
        return

    skip_tpu = pytest.mark.skip(reason="requires a TPU")
    for item in items:
        if "tpu" in item.keywords:
            item.add_marker(skip_tpu)


@pytest.fixture(
    params=[pytest.param("cpu"), pytest.param("tpu", marks=pytest.mark.tpu)]
)
def device(request: pytest.FixtureRequest) -> Iterator[jax.Device]:
    """Runs a test on the CPU, and also on the TPU when one is present."""

    try:
        device = jax.devices(request.param)[0]
    except RuntimeError:
        pytest.skip(f"no {request.param} device")
    with jax.default_device(device):
        yield device


@pytest.fixture
def isolated_settings_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """
    Keeps the test runner's command line and environment, and any config.toml
    in the working directory, out of the settings sources.
    """

    # Settings() parses sys.argv and reads ./config.toml and HERETIC_* variables.
    monkeypatch.setattr(sys, "argv", ["heretic-tpu"])
    monkeypatch.chdir(tmp_path)

    for name in list(os.environ):
        if name.startswith("HERETIC_"):
            monkeypatch.delenv(name)

    return tmp_path
