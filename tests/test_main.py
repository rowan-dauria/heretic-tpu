# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import ast
import json
import os
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import huggingface_hub
import jax
import lm_eval
import pytest
import questionary
import tomli_w
from optuna import Trial
from optuna.study import StudyDirection

import heretic_tpu
from heretic_tpu import main, utils
from heretic_tpu.backend.errors import DeviceMemoryError
from heretic_tpu.config import ExportStrategy, ModifierConfig, Settings
from heretic_tpu.modifier import Modifier, ModifierEntry
from heretic_tpu.plugin import Context
from heretic_tpu.scorer import Score
from heretic_tpu.utils import Prompt

# Modules that the command line help must not wait for.
HEAVY_MODULES = ["datasets", "jax", "lm_eval", "optuna", "torch", "transformers"]

# Dependencies of upstream Heretic that heretic-tpu dropped.
DROPPED_MODULES = ["accelerate", "bitsandbytes", "peft", "torch"]


def out_of_memory_error() -> jax.errors.JaxRuntimeError:
    return jax.errors.JaxRuntimeError(
        "RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1.00GiB."
    )


class StopRun(Exception):
    """Ends `run()` once the part under test has finished."""


class FakeQuestion:
    def __init__(self, answer: Any):
        self.answer = answer

    def ask(self) -> Any:
        return self.answer


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [0] * len(text.split())


@dataclass
class FakeParameters:
    value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_presentation_dict(self) -> dict[str, str]:
        return {"value": f"{self.value:.3f}"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FakeParameters":
        return cls(**data)


class FakeModifier(Modifier[FakeParameters]):
    events: list[Any]

    def suggest_parameters(self, ctx: Context, trial: Trial) -> FakeParameters:
        return FakeParameters(value=trial.suggest_float("value", 0.0, 1.0))

    def modify_model(self, ctx: Context, parameters: FakeParameters) -> None:
        self.events.append("modify_model")

    def reset_model(self, ctx: Context) -> None:
        self.events.append("reset_model")


@dataclass
class Harness:
    """Fakes for everything that `run()` uses, and a record of the calls they get."""

    events: list[Any] = field(default_factory=list)
    # Batch size -> error raised by get_responses for that many prompts.
    response_failures: dict[int, BaseException] = field(default_factory=dict)
    stop_at_model: bool = False
    stop_at_evaluator: bool = False
    batch_size: int | None = None
    model: "FakeModel | None" = None

    @property
    def response_batch_sizes(self) -> list[int]:
        return [
            event[1]
            for event in self.events
            if isinstance(event, tuple) and event[0] == "get_responses"
        ]


class FakeModel:
    """The parts of the model facade that `main.py` uses."""

    def __init__(self, harness: Harness, settings: Settings):
        self.harness = harness
        self.settings = settings
        self.tokenizer = FakeTokenizer()
        self.lora_enabled = True

    def get_responses(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        self.harness.events.append(("get_responses", len(prompts)))

        error = self.harness.response_failures.get(len(prompts))
        if error is not None:
            raise error

        # A constant latency, so that larger batches have a higher throughput.
        time.sleep(0.02)
        return ["one two three four"] * len(prompts)

    @contextmanager
    def lora_disabled(self) -> Iterator[None]:
        self.lora_enabled = False
        try:
            yield
        finally:
            self.lora_enabled = True

    def lm_eval_state(self) -> SimpleNamespace:
        return SimpleNamespace(lora_enabled=self.lora_enabled)

    def save_merged(self, directory: str) -> None:
        self.harness.events.append(("save_merged", directory))

    def save_adapter(self, directory: str) -> None:
        self.harness.events.append(("save_adapter", directory))

    def push_to_hub(
        self,
        repo_id: str,
        *,
        private: bool,
        token: str,
        strategy: ExportStrategy,
    ) -> None:
        self.harness.events.append(("push_to_hub", repo_id, private, token, strategy))


class FakeEvaluator:
    """An evaluator with a single, constant objective."""

    def __init__(self, harness: Harness, settings: Settings):
        harness.batch_size = settings.batch_size
        if harness.stop_at_evaluator:
            raise StopRun

        self.score = Score(value=0.5, rich_display="[bold]0.5[/]", md_display="0.5")

    def get_scores(self) -> list[tuple[str, Score]]:
        return [("Fake score", self.score)]

    def get_objective_names(self) -> list[str]:
        return ["Fake score"]

    def get_objective_directions(self) -> list[StudyDirection]:
        return [StudyDirection.MINIMIZE]

    def get_objective_values(self, scores: list[tuple[str, Score]]) -> tuple[float]:
        return (scores[0][1].value,)

    def get_paired_score_records(
        self,
        scores: list[tuple[str, Score]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "score": vars(score),
                "baseline": vars(self.score),
            }
            for name, score in scores
        ]

    def get_dataset_specifications(self) -> list[Any]:
        return []

    def all_scorers_reproducible(self) -> bool:
        return True

    def all_scorers_builtin(self) -> bool:
        return True


pytestmark = pytest.mark.usefixtures("isolated_settings_sources")


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Harness:
    harness = Harness()

    def create_model(settings: Settings) -> FakeModel:
        harness.events.append("Model")
        if harness.stop_at_model:
            raise StopRun
        harness.model = FakeModel(harness, settings)
        return harness.model

    def load_and_init_modifiers(settings: Settings, model: Any) -> list[ModifierEntry]:
        modifier = FakeModifier(heretic_settings=settings)
        modifier.events = harness.events
        return [
            ModifierEntry(
                modifier=modifier,
                name=modifier.modifier_name,
                config=ModifierConfig(plugin="test_main.FakeModifier"),
            )
        ]

    monkeypatch.setattr(main, "Model", create_model)
    monkeypatch.setattr(
        main,
        "Evaluator",
        lambda settings, model: FakeEvaluator(harness, settings),
    )
    monkeypatch.setattr(main, "load_and_init_modifiers", load_and_init_modifiers)
    monkeypatch.setattr(
        main,
        "load_prompts",
        lambda settings, specification: [
            Prompt(system="You are a helpful assistant.", user=f"Question {i}")
            for i in range(3)
        ],
    )
    monkeypatch.setattr(
        main,
        "configure_compilation_cache",
        lambda directory: harness.events.append(
            ("configure_compilation_cache", directory)
        ),
    )

    # The distribution may not be installed (the source tree is on the path).
    monkeypatch.setattr(main, "version", lambda name: "0.0.0")
    monkeypatch.setattr(utils, "version", lambda name: "0.0.0")

    return harness


def write_config(tmp_path: Path, **settings: Any) -> None:
    config = {
        "model": "test-org/test-model",
        "seed": 0,
        "response_prefix": "",
        "study_checkpoint_dir": str(tmp_path / "checkpoints"),
        "compilation_cache_dir": str(tmp_path / "xla"),
        "n_trials": 1,
        "n_startup_trials": 1,
        **settings,
    }
    (tmp_path / "config.toml").write_text(tomli_w.dumps(config))


def write_trial_config(tmp_path: Path, **settings: Any) -> None:
    """Configuration for a run of one trial, followed by one action on that trial."""
    write_config(tmp_path, batch_size=1, trial_index=0, **settings)


# One trial is run, then the chosen trial is restored.
TRIAL_EVENTS = ["Model", "reset_model", "modify_model", "reset_model", "modify_model"]


def run_python(script: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        timeout=300,
    )


def test_help_does_not_import_heavy_modules(tmp_path: Path) -> None:
    result = run_python(
        f"""
        import contextlib
        import io
        import json
        import sys

        sys.argv = ["heretic-tpu", "--help"]
        output = io.StringIO()

        try:
            with contextlib.redirect_stdout(output):
                import heretic_tpu.main
        except SystemExit as exit:
            exit_code = exit.code
        else:
            exit_code = None

        heavy_modules = [name for name in {HEAVY_MODULES!r} if name in sys.modules]
        print(json.dumps([exit_code, output.getvalue(), heavy_modules]))
        """,
        tmp_path,
    )

    exit_code, help_text, heavy_modules = json.loads(result.stdout)

    assert exit_code == 0
    assert "--model" in help_text
    assert "--parallelism" in help_text
    assert "--compilation-cache-dir" in help_text
    assert heavy_modules == []


def test_cli_modules_do_not_import_pytorch(tmp_path: Path) -> None:
    # With the modules blocked, importing them fails, as without PyTorch installed.
    result = run_python(
        f"""
        import sys

        for name in {DROPPED_MODULES!r}:
            sys.modules[name] = None

        sys.argv = ["heretic-tpu"]

        import heretic_tpu.backend.lm_eval_adapter
        import heretic_tpu.evaluator
        import heretic_tpu.main
        import heretic_tpu.modifier
        import heretic_tpu.modifiers.abliteration
        import heretic_tpu.modifiers.ara
        import heretic_tpu.plugin
        import heretic_tpu.scorer
        import heretic_tpu.scorers.benchmark_score
        import heretic_tpu.scorers.keyword_rate
        import heretic_tpu.scorers.kl_divergence

        forbidden = {DROPPED_MODULES!r} + ["lm_eval.models.huggingface"]
        print([name for name in forbidden if sys.modules.get(name) is not None])
        """,
        tmp_path,
    )

    assert result.stdout.strip().splitlines()[-1] == "[]"


def test_package_never_imports_dropped_modules() -> None:
    # Where PyTorch is installed, transformers imports it (and AutoTokenizer imports
    # accelerate) on its own, so what matters is that heretic-tpu itself never does.
    forbidden = [*DROPPED_MODULES, "lm_eval.models.huggingface"]
    package_directory = Path(heretic_tpu.__file__).parent

    imports = []
    for path in sorted(package_directory.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
                # "from lm_eval.models import huggingface"
                names += [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                continue
            for name in names:
                if any(
                    name == module or name.startswith(f"{module}.")
                    for module in forbidden
                ):
                    imports.append((path.relative_to(package_directory), name))

    assert imports == []


def test_compilation_cache_is_configured_before_model(
    harness: Harness,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_config(tmp_path, print_debug_information=True)
    harness.stop_at_model = True

    with pytest.raises(StopRun):
        main.run()

    assert harness.events == [
        ("configure_compilation_cache", str(tmp_path / "xla")),
        "Model",
    ]

    # The debug information describes JAX instead of PyTorch.
    # Rich wraps long lines, so whitespace is normalised before comparing.
    output = " ".join(capsys.readouterr().out.split())
    assert "jaxlib" in output
    assert " ".join(f"jax.devices() = {jax.devices()!r}".split()) in output
    assert "torch" not in output


@pytest.mark.parametrize(
    "error",
    [out_of_memory_error(), DeviceMemoryError("The batch doesn't fit.")],
    ids=["resource-exhausted", "device-memory"],
)
def test_batch_size_tuning_stops_at_first_out_of_memory_error(
    harness: Harness,
    tmp_path: Path,
    error: BaseException,
) -> None:
    write_config(tmp_path, max_batch_size=16)
    harness.response_failures = {4: error}
    harness.stop_at_evaluator = True

    with pytest.raises(StopRun):
        main.run()

    # Each candidate gets a warm-up run and a timed run, with exactly that many prompts.
    assert harness.response_batch_sizes == [1, 1, 2, 2, 4]
    assert harness.batch_size == 2


def test_batch_size_tuning_tries_up_to_max_batch_size(
    harness: Harness,
    tmp_path: Path,
) -> None:
    write_config(tmp_path, max_batch_size=4)
    harness.stop_at_evaluator = True

    with pytest.raises(StopRun):
        main.run()

    assert harness.response_batch_sizes == [1, 1, 2, 2, 4, 4]
    assert harness.batch_size == 4


@pytest.mark.parametrize(
    "error",
    [ValueError("Malformed prompt."), jax.errors.JaxRuntimeError("INTERNAL: Failed.")],
    ids=["value-error", "other-runtime-error"],
)
def test_batch_size_tuning_propagates_other_errors(
    harness: Harness,
    tmp_path: Path,
    error: BaseException,
) -> None:
    write_config(tmp_path, max_batch_size=16)
    harness.response_failures = {2: error}

    with pytest.raises(type(error)) as raised:
        main.run()

    assert raised.value is error
    assert harness.response_batch_sizes == [1, 1, 2]


@pytest.mark.parametrize(
    "error",
    [out_of_memory_error(), DeviceMemoryError("The batch doesn't fit.")],
    ids=["resource-exhausted", "device-memory"],
)
def test_batch_size_tuning_reraises_out_of_memory_error_at_batch_size_1(
    harness: Harness,
    tmp_path: Path,
    error: BaseException,
) -> None:
    write_config(tmp_path, max_batch_size=16)
    harness.response_failures = {1: error}

    with pytest.raises(type(error)) as raised:
        main.run()

    assert raised.value is error
    assert harness.response_batch_sizes == [1]


@pytest.mark.parametrize(
    ("strategy", "method"),
    [(ExportStrategy.MERGE, "save_merged"), (ExportStrategy.ADAPTER, "save_adapter")],
)
def test_save_action_exports_without_restoring_model(
    harness: Harness,
    tmp_path: Path,
    strategy: ExportStrategy,
    method: str,
) -> None:
    save_directory = str(tmp_path / "export")
    write_trial_config(
        tmp_path,
        model_action="save",
        save_directory=save_directory,
        export_strategy=strategy.value,
    )

    main.run()

    assert harness.events == [
        ("configure_compilation_cache", str(tmp_path / "xla")),
        *TRIAL_EVENTS,
        (method, save_directory),
    ]


def test_restored_settings_keep_the_compilation_cache(
    harness: Harness,
    tmp_path: Path,
) -> None:
    write_trial_config(tmp_path, model_action="")
    main.run()

    # Settings restored from the finished study don't contain the cache directory,
    # which the settings sources (here config.toml) of the current invocation give.
    harness.events.clear()
    write_trial_config(
        tmp_path,
        model_action="",
        checkpoint_action="continue",
        compilation_cache_dir=str(tmp_path / "other"),
    )
    main.run()

    assert harness.events[0] == ("configure_compilation_cache", str(tmp_path / "other"))


@pytest.mark.parametrize(
    "strategy",
    [ExportStrategy.MERGE, ExportStrategy.ADAPTER],
)
def test_upload_action_exports_without_restoring_model(
    harness: Harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    strategy: ExportStrategy,
) -> None:
    # A local model with a model card, which is extended and uploaded too.
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    (model_directory / "README.md").write_text(
        "---\nlicense: apache-2.0\n---\nOriginal model card.\n"
    )

    write_trial_config(
        tmp_path,
        model=str(model_directory),
        model_action="upload",
        upload_repo_id="user/model-heretic",
        upload_repo_private=True,
        export_strategy=strategy.value,
    )

    uploaded_cards: list[tuple[str, str, list[str]]] = []

    def push_card(card: huggingface_hub.ModelCard, repo_id: str, **kwargs: Any) -> None:
        harness.events.append("push_card")
        uploaded_cards.append((repo_id, card.text, card.data.to_dict()["tags"]))

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_test_token")
    monkeypatch.setattr(
        huggingface_hub,
        "whoami",
        lambda token: {"name": "user", "fullname": "Test User"},
    )
    monkeypatch.setattr(huggingface_hub.ModelCard, "push_to_hub", push_card)

    main.run()

    assert harness.events == [
        ("configure_compilation_cache", str(tmp_path / "xla")),
        *TRIAL_EVENTS,
        ("push_to_hub", "user/model-heretic", True, "hf_test_token", strategy),
        "push_card",
    ]

    [(repo_id, text, tags)] = uploaded_cards
    assert repo_id == "user/model-heretic"
    assert text.startswith(
        "# This is a decensored version of a model, made using heretic-tpu v0.0.0, "
        "a TPU port of [Heretic](https://heretic-project.org)"
    )
    assert text.endswith("Original model card.\n")
    assert "heretic" in tags
    assert "reproducible" not in tags


def test_benchmark_action_uses_jax_lm(
    harness: Harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    write_trial_config(tmp_path, model_action="benchmark")

    jax_lm_instances: list[Any] = []

    class JaxLM:
        def __init__(self, tokenizer: Any, state_fn: Callable[[], Any]):
            self.tokenizer = tokenizer
            self.state_fn = state_fn
            jax_lm_instances.append(self)

    adapter = ModuleType("heretic_tpu.backend.lm_eval_adapter")
    adapter.JaxLM = JaxLM  # ty:ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)

    evaluations: list[tuple[Any, list[str], bool]] = []

    def simple_evaluate(model: Any, tasks: list[str]) -> dict[str, Any]:
        # The adapter resolves the model state when it runs a request.
        lora_enabled = model.state_fn().lora_enabled
        evaluations.append((model, tasks, lora_enabled))
        return {
            "results": {
                tasks[0]: {"alias": tasks[0], "acc,none": 0.75 if lora_enabled else 0.5}
            }
        }

    monkeypatch.setattr(lm_eval, "simple_evaluate", simple_evaluate)
    monkeypatch.setattr(
        questionary,
        "checkbox",
        lambda message, choices, **kwargs: FakeQuestion(
            [choice.value for choice in choices[:2]]
        ),
    )
    monkeypatch.setattr(
        questionary,
        "select",
        lambda message, **kwargs: FakeQuestion("Benchmark both models"),
    )

    main.run()

    # One adapter serves every benchmark, for both the decensored and the original model.
    [lm] = jax_lm_instances
    model = harness.model
    assert model is not None
    assert lm.tokenizer is model.tokenizer
    assert lm.state_fn == model.lm_eval_state

    tasks = [benchmark.task for benchmark in model.settings.benchmarks[:2]]
    assert evaluations == [
        (lm, [tasks[0]], True),
        (lm, [tasks[0]], False),
        (lm, [tasks[1]], True),
        (lm, [tasks[1]], False),
    ]

    output = capsys.readouterr().out
    assert "0.7500" in output
    assert "0.5000" in output
    assert "Error" not in output


def test_export_strategy_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    questions: list[tuple[str, list[Any]]] = []

    def select(message: str, choices: list[Any], **kwargs: Any) -> FakeQuestion:
        questions.append((message, choices))
        return FakeQuestion(ExportStrategy.ADAPTER)

    monkeypatch.setattr(questionary, "select", select)

    settings = Settings(model="test-org/test-model")
    assert main.obtain_export_strategy(settings) == ExportStrategy.ADAPTER

    [(message, choices)] = questions
    assert message == "How do you want to export the model?"
    assert [(choice.title, choice.value) for choice in choices] == [
        (
            "Merge the abliteration LoRA and export the full model",
            ExportStrategy.MERGE,
        ),
        (
            "Export the abliteration LoRA only (can be merged later)",
            ExportStrategy.ADAPTER,
        ),
    ]

    settings.export_strategy = ExportStrategy.MERGE
    assert main.obtain_export_strategy(settings) == ExportStrategy.MERGE
