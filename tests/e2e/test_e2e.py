# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""
Non-interactive heretic-tpu runs on tiny Hub checkpoints, adapted from upstream's
tests/*/config.toml. Each case directory holds the config.toml of one run, which
also runs by hand with `heretic-tpu` in that directory.

Each test runs the command line program in a copy of its case directory and checks
that the run completes and saves the model; that the saved model (a merged checkpoint
or a PEFT adapter) loads in transformers with no missing, unexpected or mismatched
keys and carries the trial's modification; and that a reproduction from a
reproduce.json made from the run's study, with the same seed, saves byte-identical
files, which `--reproduce` also verifies against the recorded hashes. An adapter must
also give, merged with PEFT, the merged export of the same trial.

The program runs in the test process: only one process can use a TPU, and the test
process already holds it.
"""

import json
import re
import shutil
import sys
import tomllib
from pathlib import Path

import ml_dtypes  # noqa: F401 (registers bfloat16 with NumPy for safe_open)
import numpy as np
import optuna
import pytest
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from safetensors import safe_open

from heretic_tpu import main
from heretic_tpu.backend import weights
from heretic_tpu.config import Settings
from heretic_tpu.utils import generate_reproduce_json, get_file_sha256
from tests.backend.tiny import model_class

torch = pytest.importorskip("torch")

CASES = sorted(
    path.name
    for path in Path(__file__).parent.iterdir()
    if (path / "config.toml").is_file()
)

pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("isolated_settings_sources")]


def run_cli(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *arguments: str,
) -> str:
    """Runs heretic-tpu in a directory, and returns what it printed."""

    monkeypatch.chdir(directory)
    monkeypatch.setattr(sys, "argv", ["heretic-tpu", *arguments])
    # main() installs Rich's traceback handler.
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)

    main.main()

    output = capsys.readouterr().out
    # The model actions report their errors instead of raising them.
    assert not re.search(r"^Error", output, re.MULTILINE), output
    assert "Model saved to model." in output, output
    return output


def file_hashes(directory: Path) -> dict[str, str]:
    return {path.name: get_file_sha256(path) for path in sorted(directory.iterdir())}


def read_tensors(path: Path) -> dict[str, np.ndarray]:
    with safe_open(path, framework="np") as file:
        keys = file.keys()
        return {key: file.get_tensor(key) for key in keys}


def check_merged_model(directory: Path, source: weights.Checkpoint) -> None:
    _, info = model_class(source.raw_config).from_pretrained(
        directory,
        dtype=torch.float32,
        output_loading_info=True,
    )
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    assert not info["mismatched_keys"]

    # Only abliterable weights differ from the source checkpoint, and some do.
    modified = []
    for shard in source.shard_files:
        original = read_tensors(Path(source.snapshot_dir) / shard)
        merged = read_tensors(directory / shard)
        assert merged.keys() == original.keys()
        modified += [
            key
            for key, tensor in merged.items()
            if tensor.tobytes() != original[key].tobytes()
        ]
    assert modified
    assert all(
        key.endswith(("self_attn.o_proj.weight", "mlp.down_proj.weight"))
        for key in modified
    ), modified


def check_adapter(directory: Path, source: weights.Checkpoint) -> None:
    from peft import PeftModel

    base = model_class(source.raw_config).from_pretrained(
        source.snapshot_dir,
        dtype=torch.float32,
    )
    peft_model = PeftModel.from_pretrained(base, directory)
    result = peft_model.load_adapter(str(directory), adapter_name="check")
    assert not result.missing_keys
    assert not result.unexpected_keys

    adapter_config = json.loads((directory / "adapter_config.json").read_text())
    assert adapter_config["revision"] == source.sha

    # Some module is modified.
    adapter = read_tensors(directory / "adapter_model.safetensors")
    assert any(
        np.any(tensor != 0) for key, tensor in adapter.items() if "lora_B" in key
    )


def check_adapter_matches_merged_model(
    adapter: Path,
    merged: Path,
    source: weights.Checkpoint,
) -> None:
    """Merged with PEFT, the adapter gives the merged export of the same trial."""

    from peft import PeftModel

    model_type = model_class(source.raw_config)
    base = model_type.from_pretrained(source.snapshot_dir, dtype=torch.float32)
    adapted = PeftModel.from_pretrained(base, adapter).merge_and_unload().state_dict()
    exported = model_type.from_pretrained(merged, dtype=torch.float32).state_dict()

    names = [
        name
        for name in adapted
        if name.endswith(("self_attn.o_proj.weight", "mlp.down_proj.weight"))
    ]
    assert names
    for name in names:
        # The merged export rounds W + B @ A to the stored dtype (here bfloat16,
        # whose relative spacing is 2⁻⁷).
        np.testing.assert_allclose(
            adapted[name].numpy(),
            exported[name].numpy(),
            rtol=2**-7,
            atol=1e-12,
            err_msg=name,
        )


def reproduce_json(directory: Path, output: str) -> tuple[str, dict[str, str]]:
    """
    reproduce.json for the trial that a run saved, as uploading it would write it,
    and the hashes of the saved weights that it records.
    """

    (journal,) = (directory / "checkpoints").glob("*.jsonl")
    study = optuna.load_study(
        study_name="heretic",
        storage=JournalStorage(JournalFileBackend(str(journal))),
    )
    settings = Settings.model_validate_json(study.user_attrs["settings"])

    match = re.search(r"Restoring model from trial (\d+)", output)
    assert match is not None
    (trial,) = [
        trial
        for trial in study.trials
        if trial.user_attrs["index"] == int(match.group(1))
    ]

    hashes = {
        path.name: get_file_sha256(path)
        for path in sorted((directory / "model").glob("*.safetensors"))
    }
    information = generate_reproduce_json(
        settings,
        trial,
        timestamp="2026-01-01T00:00:00",
        uploaded_model_hashes=hashes,
        include_system_information=True,
    )
    return information, hashes


def reproduce(
    directory: Path,
    information: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> str:
    """Runs heretic-tpu --reproduce in a new directory, and returns what it printed."""

    directory.mkdir()
    (directory / "reproduce.json").write_text(information)
    # Stored settings don't contain paths.
    (directory / "config.toml").write_text('save_directory = "model"\n')
    return run_cli(directory, monkeypatch, capsys, "--reproduce", "reproduce.json")


@pytest.mark.parametrize("case", CASES)
def test_run(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Compiled programs are not kept between runs.
    monkeypatch.setenv("HERETIC_COMPILATION_CACHE_DIR", "")

    run = tmp_path / "run"
    shutil.copytree(Path(__file__).parent / case, run)
    output = run_cli(run, monkeypatch, capsys)

    config = tomllib.loads((run / "config.toml").read_text())
    source = weights.resolve_checkpoint(config["model"], config["model_commit"])
    if config["export_strategy"] == "merge":
        check_merged_model(run / "model", source)
    else:
        check_adapter(run / "model", source)

    # A reproduction with the same seed on the same machine gives identical files.
    information, hashes = reproduce_json(run, output)
    # Local installations of heretic-tpu never count as the same version, as in
    # upstream, so the environment check always reports a mismatch.
    monkeypatch.setenv("HERETIC_IGNORE_MISMATCHES", "true")
    output = reproduce(tmp_path / "reproduction", information, monkeypatch, capsys)

    assert output.count("Hash matches") == len(hashes)
    assert file_hashes(tmp_path / "reproduction/model") == file_hashes(run / "model")

    if config["export_strategy"] == "adapter":
        # The same trial, exported as a merged model.
        merged_information = json.loads(information)
        merged_information["settings"]["export_strategy"] = "merge"
        merged_information["hashes"] = {}
        reproduce(
            tmp_path / "merged",
            json.dumps(merged_information),
            monkeypatch,
            capsys,
        )
        check_adapter_matches_merged_model(
            run / "model",
            tmp_path / "merged/model",
            source,
        )
