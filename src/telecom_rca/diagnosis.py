"""Transparent deterministic baseline for the eight training root causes."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Diagnosis:
    """Semantic root-cause decision with auditable evidence."""

    label: str
    reason: str
    confidence_tier: str


def diagnose_high_confidence(features: dict[str, float]) -> Diagnosis | None:
    """Return one of six high-confidence causes, or ``None`` for C1/C3."""
    if features["low_speed_max"] > 40:
        return Diagnosis("C7", "Vehicle speed exceeds 40 km/h during degradation", "high")
    if features["low_distance_max_km"] > 1:
        return Diagnosis("C2", "Serving distance exceeds 1 km during degradation", "high")
    if features["low_rbs_mean"] < 160:
        return Diagnosis("C8", "Mean scheduled RBs are below 160 during degradation", "high")
    if features["ping_pong_count"] >= 1 or features["handover_count"] >= 3:
        return Diagnosis("C5", "Repeated serving-cell transitions indicate frequent handovers", "high")
    if features["low_mod30_conflict_fraction"] >= 1:
        return Diagnosis("C6", "All degraded rows expose a serving-neighbor PCI modulo-30 conflict", "high")
    if features["low_close_noncolocated_fraction"] >= 1:
        return Diagnosis("C4", "All degraded rows have close non-colocated overlapping coverage", "high")
    return None


def diagnose_standard(features: dict[str, float]) -> Diagnosis:
    """Apply domain-ordered rules to standard C1-C8 telecom features.

    Thresholds come from the definitions stated in the question (40 km/h,
    1 km, and 160 RBs) plus training-only structural evidence for handovers,
    overlap, and tilt. C3 is the explicit residual after stronger causes are
    eliminated.
    """
    high_confidence = diagnose_high_confidence(features)
    if high_confidence is not None:
        return high_confidence
    if features["low_tilt_excess_max_deg"] > 5:
        return Diagnosis("C1", "Serving downtilt exceeds the estimated beam-edge geometry", "medium")
    return Diagnosis("C3", "No stronger physical cause remains; neighboring-cell performance is favored", "low")
