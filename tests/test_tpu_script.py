# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Tests of scripts/tpu.sh, with a fake gcloud that runs the remote commands locally,
# and of the HERETIC_TPU_REQUIRE_UPSTREAM switch in tests/conftest.py that it sets.

import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from xml.etree import ElementTree

import pytest

REPO_ROOT = Path(__file__).parents[1]

# Runs the --command=... payload of `gcloud ... ssh` locally, with the SSH session's
# standard input (the archive that `sync` sends).
FAKE_GCLOUD = """\
#!/usr/bin/env bash
for argument in "$@"; do
    case "$argument" in
        --command=*) exec bash -c "${argument#--command=}" ;;
    esac
done
exit 99
"""


def git(directory: Path, *arguments: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "protocol.file.allow=always",
            "-c",
            "init.defaultBranch=main",
            *arguments,
        ],
        check=True,
        capture_output=True,
        cwd=directory,
        timeout=60,
    )


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_checkout(root: Path, scratch: bool) -> Path:
    """
    A checkout with scripts/tpu.sh, an upstream submodule at heretic/ and, if
    `scratch` is set, a git-ignored .scratch/ directory.
    """
    upstream = root / "upstream"
    write(upstream / "README.md", "upstream\n")
    write(upstream / "src" / "heretic" / "config.py", "upstream = True\n")
    write(upstream / "z.py", "last = True\n")
    git(upstream, "init", "-q")
    git(upstream, "add", "-A")
    git(upstream, "commit", "-qm", "upstream")

    checkout = root / "checkout"
    write(checkout / ".gitignore", "/.scratch/\n/ignored.txt\n")
    write(checkout / "ignored.txt", "ignored\n")
    write(checkout / "src" / "heretic_tpu" / "a.py", "a = 1\n")
    write(checkout / "src" / "heretic_tpu" / "z.py", "z = 1\n")
    (checkout / "scripts").mkdir()
    shutil.copy(REPO_ROOT / "scripts" / "tpu.sh", checkout / "scripts" / "tpu.sh")
    git(checkout, "init", "-q")
    git(checkout, "submodule", "add", str(upstream), "heretic")
    git(checkout, "add", "-A")
    git(checkout, "commit", "-qm", "checkout")
    write(checkout / "untracked.py", "untracked = True\n")

    if scratch:
        write(checkout / ".scratch" / "label" / "x.py", "print(1)\n")

    return checkout


def run_tpu_script(
    root: Path,
    checkout: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    bin_dir = root / "bin"
    write(bin_dir / "gcloud", FAKE_GCLOUD)
    (bin_dir / "gcloud").chmod(stat.S_IRWXU)

    environment = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(root / "home"),
        "TPU_NAME": "tpu",
        "TPU_ZONE": "zone",
        "TPU_PROJECT": "project",
        "TPU_REMOTE_DIR": "remote",
        "TPU_LOCK": "0",
    }
    environment.pop("HERETIC_TPU_REQUIRE_UPSTREAM", None)

    return subprocess.run(
        ["bash", "scripts/tpu.sh", *arguments],
        check=False,
        capture_output=True,
        cwd=checkout,
        env=environment,
        text=True,
        timeout=120,
    )


def remote_files(remote: Path) -> set[str]:
    return {
        path.relative_to(remote).as_posix()
        for path in remote.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    ("deleted", "scratch"),
    [
        (None, True),
        ("src/heretic_tpu/z.py", True),
        # The last path listed, which must not drop .scratch/ or fail the sync.
        ("heretic/z.py", False),
    ],
)
def test_run_syncs_the_working_tree(
    tmp_path: Path,
    deleted: str | None,
    scratch: bool,
) -> None:
    checkout = make_checkout(tmp_path, scratch)
    # A tracked file deleted without staging the deletion, which git still lists.
    if deleted is not None:
        (checkout / deleted).unlink()

    # Files left by an earlier sync, which must be removed.
    remote = tmp_path / "home" / "remote"
    write(remote / "src" / "heretic_tpu" / "stale.py", "")
    write(remote / "heretic" / "stale.py", "")
    write(remote / ".scratch" / "stale.py", "")

    result = run_tpu_script(
        tmp_path,
        checkout,
        "run",
        'echo "REQUIRE_UPSTREAM=$HERETIC_TPU_REQUIRE_UPSTREAM in $PWD"',
    )

    assert result.returncode == 0, result.stderr
    assert f"REQUIRE_UPSTREAM=1 in {remote}" in result.stdout

    expected = {
        ".gitignore",
        ".gitmodules",
        "scripts/tpu.sh",
        "src/heretic_tpu/a.py",
        "src/heretic_tpu/z.py",
        "untracked.py",
        "heretic/README.md",
        "heretic/src/heretic/config.py",
        "heretic/z.py",
    }
    if scratch:
        expected.add(".scratch/label/x.py")
    expected.discard(deleted)
    assert remote_files(remote) == expected


def test_submit_and_logs(tmp_path: Path) -> None:
    checkout = make_checkout(tmp_path, scratch=True)

    result = run_tpu_script(
        tmp_path,
        checkout,
        "submit",
        "job",
        'cat heretic/README.md .scratch/label/x.py; echo "$HERETIC_TPU_REQUIRE_UPSTREAM"',
    )
    assert result.returncode == 0, result.stderr
    assert "Submitted job" in result.stdout

    # The job runs detached; it is finished once its exit status is written.
    exit_file = tmp_path / "home" / "remote" / "job.exit"
    for _ in range(100):
        if exit_file.is_file() and exit_file.read_text().strip():
            break
        time.sleep(0.1)

    result = run_tpu_script(tmp_path, checkout, "logs", "job")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "upstream",
        "print(1)",
        "1",
        "[finished with exit status 0]",
    ]


UPSTREAM_TESTS = """\
import pytest


@pytest.fixture
def upstream():
    pytest.skip("the upstream submodule is not checked out")


@pytest.mark.skipif(True, reason="the upstream submodule is not checked out")
def test_skipif():
    pass


def test_fixture(upstream):
    pass


def test_call():
    pytest.skip("the upstream source is not checked out")


def test_other_skip():
    pytest.skip("requires a TPU")


@pytest.mark.xfail(reason="the upstream submodule is not checked out")
def test_xfail():
    assert False
"""


@pytest.mark.parametrize("required", [False, True])
def test_missing_upstream_fails_when_required(tmp_path: Path, required: bool) -> None:
    shutil.copy(Path(__file__).with_name("conftest.py"), tmp_path / "conftest.py")
    write(tmp_path / "test_upstream.py", UPSTREAM_TESTS)

    environment = {**os.environ, "JAX_PLATFORMS": "cpu"}
    environment.pop("HERETIC_TPU_REQUIRE_UPSTREAM", None)
    if required:
        environment["HERETIC_TPU_REQUIRE_UPSTREAM"] = "1"

    junit = tmp_path / "junit.xml"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            f"--junitxml={junit}",
            "test_upstream.py",
        ],
        # The run fails when the upstream tests are required.
        check=False,
        capture_output=True,
        cwd=tmp_path,
        env=environment,
        timeout=120,
    )

    outcomes = {
        case.get("name"): [
            child.tag for child in case if child.tag in {"skipped", "failure", "error"}
        ]
        for case in ElementTree.parse(junit).iter("testcase")
    }

    # A skip during setup becomes an error, and one during the call a failure.
    expected_skip = ["error"] if required else ["skipped"]
    assert outcomes == {
        "test_skipif": expected_skip,
        "test_fixture": expected_skip,
        "test_call": ["failure"] if required else ["skipped"],
        "test_other_skip": ["skipped"],
        "test_xfail": ["skipped"],
    }
