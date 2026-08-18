"""Unified routed hybrid inference without submission generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Protocol

import joblib

from .diagnosis import Diagnosis
from .features import extract_diagnostic_features
from .ml import C13Resolver
from .options import map_standard_cause, standard_option_map
from .qwen import LocalQwen, extract_boxed_answer
from .routing import Route, route_question


class ChoiceBackend(Protocol):
    """Backend contract for non-standard multiple-choice routes."""

    def answer(self, question: str, allowed: set[str]) -> str: ...


class LazyQwenChoiceBackend:
    """Load Qwen only when a GK or Markdown route actually needs it."""

    def __init__(self, model_path: Path, device: str | None = None) -> None:
        self.model_path = model_path
        self.device = device
        self._runtime: LocalQwen | None = None

    @property
    def loaded(self) -> bool:
        return self._runtime is not None

    def answer(self, question: str, allowed: set[str]) -> str:
        if not allowed:
            raise ValueError("Allowed choice domain cannot be empty")
        if self._runtime is None:
            self._runtime = LocalQwen(self.model_path, device=self.device)
        example = sorted(allowed)[0]
        result = self._runtime.generate(question, example_choice=example)
        return extract_boxed_answer(result.text, allowed)


@dataclass(frozen=True)
class UnifiedPrediction:
    route: Route
    answer: str
    boxed_text: str
    semantic_label: str | None
    evidence: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_c13_resolver(path: Path, expected_sha256: str | None = None) -> C13Resolver:
    """Load only our own Stage 4 artifact, optionally verifying its hash."""
    if not path.is_file():
        raise FileNotFoundError(f"C1/C3 resolver not found: {path}")
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"Resolver hash mismatch: expected {expected_sha256}, found {actual}")
    artifact = joblib.load(path)
    resolver = artifact.get("resolver") if isinstance(artifact, dict) else None
    if not isinstance(resolver, C13Resolver):
        raise TypeError("Artifact does not contain a valid independent C13Resolver")
    return resolver


class UnifiedHybrid:
    """Route standard telecom to the hybrid and other choices to Qwen."""

    def __init__(self, resolver: C13Resolver, choice_backend: ChoiceBackend) -> None:
        self.resolver = resolver
        self.choice_backend = choice_backend

    def predict(self, question: str) -> UnifiedPrediction:
        decision = route_question(question)
        if decision.route == Route.STANDARD_TELECOM:
            features = extract_diagnostic_features(question, decision.parsed)
            allowed_semantics = set(standard_option_map(question))
            diagnosis = self._diagnose_constrained(features, allowed_semantics)
            answer = map_standard_cause(question, diagnosis.label)
            return UnifiedPrediction(
                route=decision.route,
                answer=answer,
                boxed_text=rf"\boxed{{{answer}}}",
                semantic_label=diagnosis.label,
                evidence=diagnosis.reason,
            )

        allowed = {option.label for option in decision.options}
        answer = self.choice_backend.answer(question, allowed)
        return UnifiedPrediction(
            route=decision.route,
            answer=answer,
            boxed_text=rf"\boxed{{{answer}}}",
            semantic_label=None,
            evidence=decision.evidence,
        )

    def _diagnose_constrained(
        self,
        features: dict[str, float],
        allowed: set[str],
    ) -> Diagnosis:
        """Apply ordered evidence while never selecting an absent cause."""
        rules = (
            ("C7", features["low_speed_max"] > 40, "Vehicle speed exceeds 40 km/h"),
            ("C2", features["low_distance_max_km"] > 1, "Serving distance exceeds 1 km"),
            ("C8", features["low_rbs_mean"] < 160, "Mean scheduled RBs are below 160"),
            (
                "C5",
                features["ping_pong_count"] >= 1 or features["handover_count"] >= 3,
                "Repeated serving-cell transitions",
            ),
            ("C6", features["low_mod30_conflict_fraction"] >= 1, "PCI modulo-30 conflict"),
            (
                "C4",
                features["low_close_noncolocated_fraction"] >= 1,
                "Close non-colocated overlapping coverage",
            ),
        )
        for label, active, reason in rules:
            if active and label in allowed:
                return Diagnosis(label, reason, "high")

        unresolved = allowed & {"C1", "C3"}
        if unresolved:
            learned = self.resolver.predict_one(features)
            if learned.label in unresolved:
                return learned
            if len(unresolved) == 1:
                label = next(iter(unresolved))
                return Diagnosis(label, "Only remaining compatible C1/C3 candidate", "low")

        if len(allowed) == 1:
            label = next(iter(allowed))
            return Diagnosis(label, "Only offered semantic candidate", "low")
        raise ValueError(f"Evidence did not resolve offered semantic causes: {sorted(allowed)}")
