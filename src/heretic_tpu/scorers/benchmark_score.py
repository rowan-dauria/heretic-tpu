# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import lm_eval
from pydantic import BaseModel, Field

from heretic_tpu.scorer import Context, Score, Scorer


class Settings(BaseModel):
    score_name: str = Field(
        default="PIQA acc_norm",
        description="Name that describes what the configured benchmark score measures.",
    )

    task: str = Field(
        default="piqa",
        description="Task ID of the benchmark in the Language Model Evaluation Harness.",
    )

    metric: str = Field(
        default="acc_norm,none",
        description="Task metric to use as the benchmark score.",
    )


class BenchmarkScore(Scorer):
    """
    Calculates the score of a benchmark from the Language Model Evaluation Harness.
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return self.settings.score_name

    def init(self, ctx: Context) -> None:
        # Imported on first use, so that the plugin can be loaded and validated
        # without the lm-eval model API.
        from heretic_tpu.backend.lm_eval_adapter import JaxLM

        model = ctx.get_model()

        # JaxLM fetches the model's current state on every request, so the same
        # object scores every trial and survives model reloads, e.g. when using
        # --evaluate-model.
        self.lm = JaxLM(model.tokenizer, model.lm_eval_state)

    def get_score(self, ctx: Context) -> Score:
        results = lm_eval.simple_evaluate(
            model=self.lm,
            tasks=[self.settings.task],
        )

        benchmark_score = float(
            results["results"][self.settings.task][self.settings.metric]
        )

        return Score(
            value=benchmark_score,
            rich_display=f"[bold]{benchmark_score:.4f}[/]",
            md_display=f"{benchmark_score:.4f}",
        )
