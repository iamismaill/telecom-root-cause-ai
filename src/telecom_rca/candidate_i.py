"""Candidate I experiments inspired by robust prior telco solutions.

This module is deliberately separate from Candidate F.  It adds three pieces
that can be evaluated independently: evidence-quality routing, retrieval from
the official training set, and deterministic multi-prompt consensus.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .features import extract_diagnostic_features
from .options import standard_option_map
from .parser import ParsedQuestion, parse_question
from .pipeline import ChoiceBackend


RETRIEVAL_FEATURES = (
    "throughput_min",
    "low_throughput_mean",
    "low_serving_rsrp_mean",
    "low_serving_sinr_mean",
    "low_speed_max",
    "low_rbs_mean",
    "low_distance_max_km",
    "low_distance_mean_km",
    "low_serving_total_tilt_max",
    "low_tilt_excess_max_deg",
    "low_neighbor_margin_max_db",
    "low_neighbor_stronger_fraction",
    "low_neighbor_throughput_gain_max",
    "low_neighbor_throughput_gain_positive_fraction",
    "low_close_noncolocated_fraction",
    "low_mod30_conflict_fraction",
    "handover_count",
    "ping_pong_count",
    "engineering_match_fraction",
)


@dataclass(frozen=True)
class EvidenceQuality:
    complete: bool
    score: float
    missing: tuple[str, ...]
    reason: str


def assess_standard_evidence(question: str, parsed: ParsedQuestion | None = None) -> EvidenceQuality:
    """Assess whether deterministic telecom logic has its required evidence."""
    parsed = parsed or parse_question(question)
    if not parsed.is_telecom or "engineering" not in parsed.tables:
        return EvidenceQuality(False, 0.0, ("telecom_tables",), "Required telecom tables are absent")
    features = extract_diagnostic_features(question, parsed)
    required = {
        "drive_rows": features.get("row_count", math.nan),
        "degraded_rows": features.get("low_row_count", math.nan),
        "throughput": features.get("throughput_min", math.nan),
        "serving_signal": features.get("low_serving_rsrp_mean", math.nan),
        "scheduled_rbs": features.get("low_rbs_mean", math.nan),
        "engineering_match": features.get("engineering_match_fraction", math.nan),
    }
    missing = tuple(name for name, value in required.items() if not np.isfinite(value))
    positive_rows = required["drive_rows"] > 0 and required["degraded_rows"] > 0
    matched = required["engineering_match"] > 0
    complete = not missing and positive_rows and matched
    score = sum(np.isfinite(value) for value in required.values()) / len(required)
    if not positive_rows:
        score *= 0.5
    if not matched:
        score *= 0.75
    reason = "Complete drive-test and matched engineering evidence" if complete else (
        "Incomplete deterministic evidence: " + ", ".join(missing or ("no usable rows or engineering match",))
    )
    return EvidenceQuality(complete, float(score), missing, reason)


@dataclass(frozen=True)
class RetrievalPrediction:
    label: str
    confidence: float
    neighbor_labels: tuple[str, ...]
    neighbor_distances: tuple[float, ...]


class TrainingCaseRetriever:
    """K-nearest case evidence fitted exclusively on official labelled training rows."""

    def __init__(self, k: int = 7) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.matrix: np.ndarray | None = None
        self.labels: np.ndarray | None = None

    @staticmethod
    def _row(question: str) -> list[float]:
        parsed = parse_question(question)
        features = extract_diagnostic_features(question, parsed)
        return [features.get(name, math.nan) for name in RETRIEVAL_FEATURES]

    def fit(self, questions: Iterable[str], labels: Iterable[str]) -> "TrainingCaseRetriever":
        rows = list(questions)
        y = np.asarray(list(labels), dtype=str)
        if len(rows) != len(y) or not rows:
            raise ValueError("Questions and labels must be non-empty and aligned")
        raw = np.asarray([self._row(question) for question in rows], dtype=float)
        self.matrix = self.scaler.fit_transform(self.imputer.fit_transform(raw))
        self.labels = y
        return self

    def predict(self, question: str, allowed: set[str] | None = None) -> RetrievalPrediction:
        if self.matrix is None or self.labels is None:
            raise RuntimeError("Retriever must be fitted before prediction")
        raw = np.asarray([self._row(question)], dtype=float)
        query = self.scaler.transform(self.imputer.transform(raw))[0]
        distances = np.linalg.norm(self.matrix - query, axis=1)
        order = np.argsort(distances)[: min(self.k, len(distances))]
        labels = self.labels[order]
        if allowed is not None:
            keep = np.asarray([label in allowed for label in labels])
            order, labels = order[keep], labels[keep]
        if not len(labels):
            raise ValueError("No retrieved labels belong to the offered cause domain")
        weights = 1.0 / np.maximum(distances[order], 1e-6)
        totals = {label: float(weights[labels == label].sum()) for label in set(labels)}
        winner = max(totals, key=totals.get)
        confidence = totals[winner] / sum(totals.values())
        return RetrievalPrediction(
            winner,
            confidence,
            tuple(str(label) for label in labels),
            tuple(float(distances[index]) for index in order),
        )


class ConsensusChoiceBackend:
    """Accept an answer only when deterministic prompt variants reach quorum."""

    PROMPTS = (
        "\n\nSolve independently. Return exactly one offered label in \\boxed{}.",
        "\n\nCheck the evidence and eliminate incompatible choices. Return one label in \\boxed{}.",
        "\n\nRecalculate carefully, then return only the final offered label in \\boxed{}.",
    )

    def __init__(self, backend: ChoiceBackend, quorum: int = 2) -> None:
        if not 1 <= quorum <= len(self.PROMPTS):
            raise ValueError("Invalid consensus quorum")
        self.backend = backend
        self.quorum = quorum

    def answer(self, question: str, allowed: set[str]) -> str:
        answers = [self.backend.answer(question + suffix, allowed) for suffix in self.PROMPTS]
        counts = {answer: answers.count(answer) for answer in set(answers)}
        winner = max(counts, key=counts.get)
        if counts[winner] < self.quorum:
            raise ValueError(f"No Qwen consensus: {answers}")
        return winner


def offered_semantics(question: str) -> set[str]:
    """Return the standard semantic cause domain for retrieval constraints."""
    return set(standard_option_map(question))
