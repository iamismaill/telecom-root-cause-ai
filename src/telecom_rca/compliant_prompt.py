"""Compact, model-only decision prompts derived from one question at a time."""

from __future__ import annotations

import math

from .features import extract_diagnostic_features
from .options import parse_options
from .parser import parse_question


FEATURES = (
    ("low_row_count", "degraded rows"),
    ("throughput_threshold", "throughput threshold Mbps"),
    ("low_throughput_mean", "degraded throughput mean Mbps"),
    ("low_speed_max", "degraded speed maximum km/h"),
    ("low_rbs_mean", "degraded scheduled RB mean"),
    ("low_rbs_below_160_fraction", "degraded RB-below-160 fraction"),
    ("handover_count", "serving-cell transitions"),
    ("ping_pong_count", "ping-pong transitions"),
    ("low_distance_max_km", "degraded serving distance maximum km"),
    ("low_tilt_excess_max_deg", "degraded tilt excess maximum degrees"),
    ("low_neighbor_margin_max_db", "degraded best-neighbor RSRP margin dB"),
    (
        "low_neighbor_throughput_gain_positive_fraction",
        "degraded rows with positive neighbor throughput gain fraction",
    ),
    ("low_mod30_conflict_fraction", "degraded PCI-mod-30 conflict fraction"),
    (
        "low_close_noncolocated_fraction",
        "degraded close non-colocated neighbor fraction",
    ),
    ("engineering_match_fraction", "engineering-record match fraction"),
)


CAUSE_HINTS = {
    "C1": "Check whether downtilt exceeds the beam-edge geometry.",
    "C2": "Check whether degraded serving distance exceeds 1 km.",
    "C3": "Check whether a neighbor offers better radio/throughput evidence.",
    "C4": "Check close, non-colocated co-frequency overlap.",
    "C5": "Check repeated serving-cell changes and ping-pong behavior.",
    "C6": "Check serving-neighbor PCI modulo-30 conflicts.",
    "C7": "Check whether degraded vehicle speed exceeds 40 km/h.",
    "C8": "Check whether degraded scheduled RB mean is below 160.",
}


def _display(value: float) -> str:
    if math.isnan(value):
        return "unavailable"
    return f"{value:.4g}"


def compact_model_prompt(question: str) -> str:
    """Represent one telecom question without selecting its answer."""
    parsed = parse_question(question)
    features = extract_diagnostic_features(question, parsed)
    options = parse_options(question)
    if len(options) != 8:
        raise ValueError(f"Expected eight answer options, found {len(options)}")
    option_lines = "\n".join(f"- {item.label}: {item.description}" for item in options)
    evidence_lines = "\n".join(
        f"- {description}: {_display(float(features[name]))}"
        for name, description in FEATURES
    )
    checks = "\n".join(f"- {hint}" for hint in CAUSE_HINTS.values())
    return (
        "Diagnose the root cause of the degraded 5G throughput using only this "
        "question's derived evidence.\n\n"
        f"Offered choices:\n{option_lines}\n\n"
        f"Evidence from degraded rows:\n{evidence_lines}\n\n"
        f"Diagnostic checks:\n{checks}\n\n"
        "Compare all eight causes, select the best offered choice, and finish with "
        "its exact displayed label."
    )


def rubric_model_prompt(question: str) -> str:
    """Add the challenge's engineering decision rubric for model reasoning."""
    base = compact_model_prompt(question)
    rubric = (
        "\n\nApply this ordered engineering rubric yourself:\n"
        "1. If degraded speed maximum exceeds 40 km/h, select the vehicle-speed cause.\n"
        "2. Otherwise, if degraded serving distance exceeds 1 km, select overshooting.\n"
        "3. Otherwise, if degraded scheduled RB mean is below 160, select low RBs.\n"
        "4. Otherwise, if there is ping-pong or at least three serving-cell transitions, "
        "select frequent handovers.\n"
        "5. Otherwise, if the degraded PCI-mod-30 conflict fraction is 1, select "
        "PCI-mod-30 interference.\n"
        "6. Otherwise, if the degraded close non-colocated neighbor fraction is 1, "
        "select overlapping coverage.\n"
        "7. Otherwise, if degraded tilt excess is above 5 degrees, select excessive "
        "downtilt.\n"
        "8. Otherwise select the better-neighbor cause.\n"
        "Perform these comparisons internally. Your response must only contain the "
        "displayed label in \\boxed{}."
    )
    return base + rubric


def condition_model_prompt(question: str) -> str:
    """Present current-question condition results without computing a final label."""
    parsed = parse_question(question)
    f = extract_diagnostic_features(question, parsed)
    options = parse_options(question)
    if len(options) != 8:
        raise ValueError(f"Expected eight answer options, found {len(options)}")

    def flag(value: bool) -> str:
        return "TRUE" if value else "FALSE"

    conditions = (
        ("C7 vehicle speed above 40 km/h", flag(f["low_speed_max"] > 40)),
        ("C2 serving distance above 1 km", flag(f["low_distance_max_km"] > 1)),
        ("C8 scheduled RB mean below 160", flag(f["low_rbs_mean"] < 160)),
        (
            "C5 frequent/ping-pong handovers",
            flag(f["ping_pong_count"] >= 1 or f["handover_count"] >= 3),
        ),
        (
            "C6 PCI modulo-30 conflict throughout degraded rows",
            flag(f["low_mod30_conflict_fraction"] >= 1),
        ),
        (
            "C4 close non-colocated overlap throughout degraded rows",
            flag(f["low_close_noncolocated_fraction"] >= 1),
        ),
        ("C1 excessive downtilt geometry", flag(f["low_tilt_excess_max_deg"] > 5)),
    )
    option_lines = "\n".join(f"- {o.label}: {o.description}" for o in options)
    condition_lines = "\n".join(f"- {name}: {value}" for name, value in conditions)
    return (
        "Use only this question's computed engineering evidence.\n\n"
        f"Offered choices:\n{option_lines}\n\n"
        f"Condition results:\n{condition_lines}\n\n"
        "Select the first TRUE condition in the order shown. If every condition is "
        "FALSE, select C3 (better neighboring cell). The option descriptions determine "
        "the displayed label. Generate only that displayed label in \\boxed{}."
    )


def enhanced_condition_model_prompt(question: str) -> str:
    """Add continuous C1/C3 evidence to the otherwise discrete condition prompt."""
    parsed = parse_question(question)
    f = extract_diagnostic_features(question, parsed)
    comparison = (
        "\n\nC1 versus C3 continuous evidence:\n"
        f"- degraded tilt excess maximum degrees: {_display(f['low_tilt_excess_max_deg'])}\n"
        f"- degraded tilt excess mean degrees: {_display(f['low_tilt_excess_mean_deg'])}\n"
        f"- serving total tilt maximum degrees: {_display(f['low_serving_total_tilt_max'])}\n"
        f"- mechanical tilt maximum degrees: {_display(f['low_mechanical_tilt_max'])}\n"
        f"- degraded serving RSRP mean dBm: {_display(f['low_serving_rsrp_mean'])}\n"
        f"- best-neighbor RSRP margin maximum dB: {_display(f['low_neighbor_margin_max_db'])}\n"
        f"- stronger-neighbor fraction: {_display(f['low_neighbor_stronger_fraction'])}\n"
        f"- positive neighbor-throughput-gain fraction: "
        f"{_display(f['low_neighbor_throughput_gain_positive_fraction'])}\n\n"
        "If an earlier TRUE condition identifies C2/C4/C5/C6/C7/C8, retain it. "
        "Otherwise distinguish C1 from C3 using the continuous evidence: C1 is "
        "supported by unusually excessive serving-cell tilt/beam-edge geometry; "
        "C3 is the residual better-neighbor diagnosis when excessive tilt is not "
        "supported. Generate only the displayed option label in \\boxed{}."
    )
    return condition_model_prompt(question) + comparison


def supervised_rationale(semantic_label: str, displayed_label: str) -> str:
    """Return a decision-only target so every trained token supports the choice."""
    if semantic_label not in CAUSE_HINTS:
        raise ValueError(f"Unsupported semantic label: {semantic_label}")
    return f"\\boxed{{{displayed_label}}}"
