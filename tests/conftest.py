# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import os
import sys
from collections.abc import Generator, Iterator
from pathlib import Path

import jax
import pytest

# Set by scripts/tpu.sh on the VM, where the tests run: the tests that compare with
# upstream Heretic then fail, rather than skip, when the heretic/ submodule is missing,
# so that they cannot go unnoticed. It is read once here, because the
# isolated_settings_sources fixture removes HERETIC_* variables while a test runs.
REQUIRE_UPSTREAM = os.environ.get("HERETIC_TPU_REQUIRE_UPSTREAM") == "1"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo[None],
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield

    # A skipif mark skips during setup, and pytest.skip() in a fixture or test
    # during setup or the call. The reasons are "the upstream submodule is not
    # checked out" and "the upstream source is not checked out".
    if (
        REQUIRE_UPSTREAM
        and report.skipped
        and not hasattr(report, "wasxfail")
        and isinstance(report.longrepr, tuple)
    ):
        reason = report.longrepr[2].removeprefix("Skipped: ")
        if "upstream" in reason and "not checked out" in reason:
            report.outcome = "failed"
            report.longrepr = (
                f"{reason}, but HERETIC_TPU_REQUIRE_UPSTREAM=1 requires it "
                "(scripts/tpu.sh syncs the heretic/ submodule; run "
                "`git submodule update --init` locally)"
            )

    return report


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
