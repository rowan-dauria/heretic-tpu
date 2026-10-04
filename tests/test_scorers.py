# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import lm_eval
import numpy as np
import pytest

from heretic_tpu.config import Settings
from heretic_tpu.modifier import Modifier
from heretic_tpu.plugin import Context, is_builtin_plugin, load_plugin
from heretic_tpu.scorer import Score, Scorer
from heretic_tpu.scorers.benchmark_score import BenchmarkScore
from heretic_tpu.scorers.benchmark_score import Settings as BenchmarkScoreSettings
from heretic_tpu.scorers.keyword_rate import REFUSAL_MARKERS, KeywordRate
from heretic_tpu.scorers.keyword_rate import Settings as KeywordRateSettings
from heretic_tpu.scorers.kl_divergence import KLDivergence
from heretic_tpu.scorers.kl_divergence import Settings as KLDivergenceSettings
from heretic_tpu.utils import Prompt

UPSTREAM_SOURCE = Path(__file__).parents[1] / "heretic" / "src"

# Import paths of the built-in plugins, relative to the package.
BUILTIN_PLUGINS = [
    ("scorers.keyword_rate.KeywordRate", Scorer),
    ("scorers.kl_divergence.KLDivergence", Scorer),
    ("scorers.benchmark_score.BenchmarkScore", Scorer),
    ("modifiers.abliteration.Abliteration", Modifier),
    ("modifiers.ara.ARA", Modifier),
]

# Dependencies of upstream Heretic that heretic-tpu dropped. A built-in plugin
# whose module still imports one of them has not been ported yet.
DROPPED_DEPENDENCIES = {"accelerate", "bitsandbytes", "peft", "torch"}

PROMPTS = [
    Prompt(system="You are a helpful assistant.", user=f"Question {i}")
    for i in range(4)
]

# Responses classified by hand from the default markers and the normalisation rules.
KEYWORD_CASES = [
    ("", True),
    ("  \n\t ", True),
    ("Sure! Here is how you do it.", False),
    ("I *will not* help with that.", True),
    ("I won’t do that.", True),
    ("I   CANNOT\n\tdo that", True),
    ("As an AI language model, I have no opinions.", True),
    ("That would violate the rules.", True),
    ("I am happy to help with your essay.", False),
    ("Here are the steps: first, preheat the oven.", False),
]


pytestmark = pytest.mark.usefixtures("isolated_settings_sources")


@pytest.fixture
def heretic_settings() -> Settings:
    return Settings(model="test-org/test-model")


@pytest.fixture(
    params=[
        pytest.param("default"),
        pytest.param("tpu", marks=pytest.mark.tpu),
    ]
)
def backend(request: pytest.FixtureRequest) -> str:
    """Runs a test on the default JAX backend, and once more marked as a TPU test."""
    if request.param == "tpu" and jax.default_backend() != "tpu":
        pytest.skip("requires a TPU")
    return request.param


class FakeModel:
    """The parts of the model facade that the scorers use."""

    def __init__(
        self,
        logits: list[Any] | None = None,
        responses: list[str] | None = None,
    ):
        self.logits = list(logits or [])
        self.responses = responses or []
        self.tokenizer = object()

    def get_logits_batched(self, prompts: list[Prompt]) -> Any:
        return self.logits.pop(0)

    def get_responses_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        assert skip_special_tokens
        return self.responses[: len(prompts)]

    def lm_eval_state(self) -> SimpleNamespace:
        return SimpleNamespace(model=self)


def make_context(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    model: FakeModel,
) -> Context:
    monkeypatch.setattr(
        "heretic_tpu.plugin.load_prompts",
        lambda settings, specification: PROMPTS,
    )
    return Context(settings=settings, model=model)  # ty:ignore[invalid-argument-type]


@pytest.fixture
def jax_lm_instances(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """
    Replaces the lm-eval adapter with a fake JaxLM that records its instances,
    so that BenchmarkScore can be tested without the engine.
    """
    instances: list[Any] = []

    class JaxLM:
        def __init__(self, tokenizer: Any, state_fn: Callable[[], Any]):
            self.tokenizer = tokenizer
            self.state_fn = state_fn
            instances.append(self)

    module = ModuleType("heretic_tpu.backend.lm_eval_adapter")
    module.JaxLM = JaxLM  # ty:ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    return instances


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Upstream Heretic's own modules, imported from the heretic/ checkout."""
    for module_name in ["torch", "accelerate", "peft"]:
        pytest.importorskip(module_name)
    if not (UPSTREAM_SOURCE / "heretic" / "scorers" / "keyword_rate.py").exists():
        pytest.skip("upstream Heretic is not checked out in heretic/")

    monkeypatch.syspath_prepend(str(UPSTREAM_SOURCE))

    return SimpleNamespace(
        config=importlib.import_module("heretic.config"),
        keyword_rate=importlib.import_module("heretic.scorers.keyword_rate"),
    )


def random_logits(seed: int, shape: tuple[int, int], scale: float) -> np.ndarray:
    return (np.random.default_rng(seed).normal(size=shape) * scale).astype(np.float32)


def kl_div_reference(logits: np.ndarray, baseline_logits: np.ndarray) -> float:
    """Upstream's computation, in PyTorch."""
    torch = pytest.importorskip("torch")
    F = torch.nn.functional

    return F.kl_div(
        F.log_softmax(torch.from_numpy(logits), dim=-1),
        F.log_softmax(torch.from_numpy(baseline_logits), dim=-1),
        reduction="batchmean",
        log_target=True,
    ).item()


@pytest.mark.parametrize("offload_outputs_to_cpu", [True, False])
def test_kl_divergence_matches_pytorch(
    monkeypatch: pytest.MonkeyPatch,
    heretic_settings: Settings,
    backend: str,
    offload_outputs_to_cpu: bool,
) -> None:
    baseline_logits = random_logits(0, (len(PROMPTS), 1031), scale=6.0)
    logits = baseline_logits + random_logits(1, baseline_logits.shape, scale=2.0)
    expected = kl_div_reference(logits, baseline_logits)

    # The model returns host arrays with offload_outputs_to_cpu, device arrays otherwise.
    to_output = np.asarray if offload_outputs_to_cpu else jnp.asarray
    model = FakeModel(logits=[to_output(baseline_logits), to_output(logits)])
    ctx = make_context(monkeypatch, heretic_settings, model)

    scorer = KLDivergence(
        heretic_settings=heretic_settings,
        settings=KLDivergenceSettings(),
    )
    scorer.init(ctx)

    # The baseline stays where the model put the logits it was computed from.
    assert isinstance(scorer._baseline_logprobs, np.ndarray) == offload_outputs_to_cpu
    assert isinstance(scorer._baseline_logprobs, jax.Array) != offload_outputs_to_cpu

    score = scorer.get_score(ctx)

    assert expected > 0.1
    assert isinstance(score.value, float)
    assert score.value == pytest.approx(expected, rel=1e-5)
    assert score.rich_display == f"[bold]{score.value:.4f}[/]"
    assert score.md_display == f"{score.value:.4f}"


def test_kl_divergence_is_computed_in_float32(
    monkeypatch: pytest.MonkeyPatch,
    heretic_settings: Settings,
    backend: str,
) -> None:
    # Large, close bfloat16 logits: if log_softmax ran in bfloat16,
    # the rounding error would even make the result negative.
    baseline_logits = jnp.asarray(
        random_logits(2, (len(PROMPTS), 2048), scale=20.0),
        dtype=jnp.bfloat16,
    )
    logits = (baseline_logits + 0.25).astype(jnp.bfloat16)
    logits = logits.at[:, :8].add(1.0)
    expected = kl_div_reference(
        np.asarray(logits, dtype=np.float32),
        np.asarray(baseline_logits, dtype=np.float32),
    )

    model = FakeModel(logits=[baseline_logits, logits])
    ctx = make_context(monkeypatch, heretic_settings, model)

    scorer = KLDivergence(
        heretic_settings=heretic_settings,
        settings=KLDivergenceSettings(),
    )
    scorer.init(ctx)
    score = scorer.get_score(ctx)

    assert score.value == pytest.approx(expected, rel=1e-5)


def test_kl_divergence_of_unchanged_model_is_zero(
    monkeypatch: pytest.MonkeyPatch,
    heretic_settings: Settings,
) -> None:
    logits = random_logits(3, (len(PROMPTS), 517), scale=6.0)
    model = FakeModel(logits=[logits, logits.copy()])
    ctx = make_context(monkeypatch, heretic_settings, model)

    scorer = KLDivergence(
        heretic_settings=heretic_settings,
        settings=KLDivergenceSettings(),
    )
    scorer.init(ctx)

    assert scorer.get_score(ctx).value == pytest.approx(0.0, abs=1e-6)
    assert scorer.get_baseline_score(ctx) == Score(
        value=0,
        rich_display="[bold]0[/] [italic](by definition)[/]",
        md_display="0 *(by definition)*",
    )


@pytest.mark.parametrize(("response", "is_match"), KEYWORD_CASES)
def test_keyword_rate_classifies_responses(
    heretic_settings: Settings,
    response: str,
    is_match: bool,
) -> None:
    scorer = KeywordRate(
        heretic_settings=heretic_settings, settings=KeywordRateSettings()
    )

    assert scorer._is_match(response) == is_match


def test_keyword_rate_uses_configured_markers(heretic_settings: Settings) -> None:
    scorer = KeywordRate(
        heretic_settings=heretic_settings,
        settings=KeywordRateSettings(keyword_markers=["Banana"]),
    )

    assert scorer._is_match("I like banana splits.")
    assert not scorer._is_match("I cannot do that.")
    # Empty responses always count as matches.
    assert scorer._is_match("")


def test_keyword_rate_score(
    monkeypatch: pytest.MonkeyPatch,
    heretic_settings: Settings,
) -> None:
    responses = ["I cannot help.", "Sure, here you go.", "", "Sorry, no."]
    ctx = make_context(monkeypatch, heretic_settings, FakeModel(responses=responses))

    scorer = KeywordRate(
        heretic_settings=heretic_settings, settings=KeywordRateSettings()
    )
    scorer.init(ctx)

    assert scorer.prompts == PROMPTS
    assert scorer.score_name == "Refusals"
    assert scorer.get_score(ctx) == Score(
        value=0.75,
        rich_display="[bold]3[/]/4",
        md_display="3/4",
    )


def test_keyword_rate_matches_upstream(
    monkeypatch: pytest.MonkeyPatch,
    heretic_settings: Settings,
    upstream: SimpleNamespace,
) -> None:
    upstream_settings = upstream.keyword_rate.Settings()
    upstream_scorer = upstream.keyword_rate.KeywordRate(
        heretic_settings=upstream.config.Settings(model="test-org/test-model"),
        settings=upstream_settings,
    )
    scorer = KeywordRate(
        heretic_settings=heretic_settings, settings=KeywordRateSettings()
    )

    assert REFUSAL_MARKERS == upstream.keyword_rate.REFUSAL_MARKERS
    assert KeywordRateSettings().model_dump() == upstream_settings.model_dump()

    # Every marker on its own, in other cases and surrounded by text, plus the cases above.
    responses = [response for response, _ in KEYWORD_CASES]
    for marker in REFUSAL_MARKERS:
        responses += [marker, marker.upper(), f"Well, *{marker}* indeed."]

    for response in responses:
        assert scorer._is_match(response) == upstream_scorer._is_match(response)

    ctx = make_context(monkeypatch, heretic_settings, FakeModel(responses=responses))
    scorer.prompts = upstream_scorer.prompts = PROMPTS

    assert vars(scorer.get_score(ctx)) == vars(upstream_scorer.get_score(ctx))


def test_benchmark_score_uses_one_jax_lm(
    monkeypatch: pytest.MonkeyPatch,
    heretic_settings: Settings,
    jax_lm_instances: list[Any],
) -> None:
    calls: list[dict[str, Any]] = []

    def simple_evaluate(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"results": {"piqa": {"alias": "piqa", "acc_norm,none": 0.625}}}

    monkeypatch.setattr(lm_eval, "simple_evaluate", simple_evaluate)

    model = FakeModel()
    ctx = make_context(monkeypatch, heretic_settings, model)

    scorer = BenchmarkScore(
        heretic_settings=heretic_settings,
        settings=BenchmarkScoreSettings(),
    )
    scorer.init(ctx)

    assert len(jax_lm_instances) == 1
    lm = jax_lm_instances[0]
    assert lm.tokenizer is model.tokenizer
    # The adapter is given the facade's state function, not a snapshot of the state.
    assert lm.state_fn == model.lm_eval_state

    for _ in range(2):
        assert scorer.get_score(ctx) == Score(
            value=0.625,
            rich_display="[bold]0.6250[/]",
            md_display="0.6250",
        )

    assert len(jax_lm_instances) == 1
    assert calls == [{"model": lm, "tasks": ["piqa"]}] * 2


@pytest.mark.parametrize(("name", "base_class"), BUILTIN_PLUGINS)
def test_builtin_plugin_loads_under_upstream_and_port_names(
    heretic_settings: Settings,
    jax_lm_instances: list[Any],
    name: str,
    base_class: type[Any],
) -> None:
    port_name = f"heretic_tpu.{name}"
    upstream_name = f"heretic.{name}"

    try:
        plugin_class = load_plugin(port_name, base_class)
    except ImportError as error:
        missing = getattr(error.__cause__, "name", None)
        if missing in DROPPED_DEPENDENCIES:
            pytest.skip(f"{port_name} still imports {missing} (not ported yet)")
        raise

    assert load_plugin(upstream_name, base_class) is plugin_class
    assert f"{plugin_class.__module__}.{plugin_class.__name__}" == port_name
    assert is_builtin_plugin(port_name)
    assert is_builtin_plugin(upstream_name)

    plugin_class.validate_contract()
    plugin_class(
        heretic_settings=heretic_settings,
        settings=plugin_class.validate_settings({}),
    )


def test_default_plugins_are_builtin(heretic_settings: Settings) -> None:
    builtin_names = {f"heretic_tpu.{name}" for name, _ in BUILTIN_PLUGINS}

    for config in [*heretic_settings.scorers, *heretic_settings.modifiers]:
        assert config.plugin in builtin_names
