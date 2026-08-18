"""Resilient inference wrapper for genuinely unseen or incomplete formats."""

from __future__ import annotations

from dataclasses import dataclass

from .markdown_diagnosis import diagnose_markdown
from .options import parse_options
from .pipeline import ChoiceBackend, UnifiedHybrid, UnifiedPrediction
from .routing import Route, route_question


@dataclass(frozen=True)
class ResilientPrediction:
    prediction: UnifiedPrediction
    used_fallback: bool
    fallback_reason: str | None


class ResilientHybrid:
    """Use specialized diagnosis when supported and constrained Qwen otherwise."""

    def __init__(self, standard_pipeline: UnifiedHybrid, choice_backend: ChoiceBackend) -> None:
        self.standard_pipeline = standard_pipeline
        self.choice_backend = choice_backend

    def predict(self, question: str) -> ResilientPrediction:
        options = parse_options(question)
        allowed = {option.label for option in options}
        if not 2 <= len(allowed) <= 9:
            raise ValueError(f"Unsupported offered-choice domain: {sorted(allowed)}")
        try:
            decision = route_question(question)
        except ValueError as exc:
            answer = self.choice_backend.answer(question, allowed)
            return ResilientPrediction(
                UnifiedPrediction(
                    route=Route.GENERAL_KNOWLEDGE,
                    answer=answer,
                    boxed_text=rf"\boxed{{{answer}}}",
                    semantic_label=None,
                    evidence="Constrained local-model fallback for unrecognized structure",
                ),
                True,
                f"routing: {exc}",
            )

        if decision.route == Route.STANDARD_TELECOM:
            return ResilientPrediction(self.standard_pipeline.predict(question), False, None)
        if decision.route == Route.MARKDOWN_TELECOM:
            try:
                diagnosis = diagnose_markdown(question)
                prediction = UnifiedPrediction(
                    route=decision.route,
                    answer=diagnosis.displayed_answer,
                    boxed_text=rf"\boxed{{{diagnosis.displayed_answer}}}",
                    semantic_label=diagnosis.semantic_cause,
                    evidence=diagnosis.evidence,
                )
                return ResilientPrediction(prediction, False, None)
            except (ValueError, KeyError) as exc:
                answer = self.choice_backend.answer(question, allowed)
                return ResilientPrediction(
                    UnifiedPrediction(
                        route=decision.route,
                        answer=answer,
                        boxed_text=rf"\boxed{{{answer}}}",
                        semantic_label=None,
                        evidence="Constrained local-model fallback for incomplete Markdown evidence",
                    ),
                    True,
                    f"markdown diagnosis: {exc}",
                )
        return ResilientPrediction(self.standard_pipeline.predict(question), False, None)
