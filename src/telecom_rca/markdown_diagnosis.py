"""Deterministic diagnosis of the nine synthetic Markdown telecom anomalies."""

from __future__ import annotations

from dataclasses import dataclass

from .markdown_features import markdown_option_map
from .routing import Route, route_question


@dataclass(frozen=True)
class MarkdownDiagnosis:
    semantic_cause: str
    displayed_answer: str
    evidence: str


def diagnose_markdown(question: str) -> MarkdownDiagnosis:
    """Decode one injected anomaly using mutually interpretable evidence."""
    from .markdown_features import extract_markdown_evidence

    decision = route_question(question)
    if decision.route != Route.MARKDOWN_TELECOM:
        raise ValueError(f"Expected Markdown telecom question, found {decision.route.value}")
    features = extract_markdown_evidence(decision.parsed)
    rules = (
        (
            "missing_neighbor",
            features["rows_with_missing_neighbor_fraction"] > 0,
            "Observed neighbor PCI is absent from configured neighbor relations",
        ),
        (
            "inter_frequency_threshold",
            features["a2_threshold_max_dbm"] > -100,
            "A2/A5 inter-frequency threshold is shifted from the common -105 dBm setting",
        ),
        (
            "intra_frequency_high",
            features["a3_offset_max_db"] >= 5,
            "A3 offset is elevated to 5 dB",
        ),
        (
            "intra_frequency_low",
            features["a3_offset_min_db"] <= 1,
            "A3 offset is reduced to 1 dB",
        ),
        (
            "pdcch",
            features["low_cce_fail_max"] > 0.5,
            "CCE failure rate has a distinct high excursion",
        ),
        (
            "capacity",
            features["low_rb_below_100_fraction"] >= 0.5,
            "Low-throughput interval receives fewer than 100 RBs",
        ),
        (
            "transport",
            features["low_grant_mean"] < 1000,
            "Low-throughput interval has anomalously low grants despite high RB allocation",
        ),
        (
            "weak_coverage",
            features["low_rsrp_mean"] < -100,
            "Low-throughput interval has weak mean serving RSRP",
        ),
        (
            "overlap",
            features["low_overlap_3db_fraction"] >= 0.5,
            "Neighbor signal remains within 3 dB of serving signal across the interval",
        ),
    )
    for semantic, active, evidence in rules:
        if active:
            return MarkdownDiagnosis(semantic, markdown_option_map(question)[semantic], evidence)
    raise ValueError("No supported injected Markdown anomaly was detected")
