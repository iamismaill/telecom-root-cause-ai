"""Clean-room benchmark of Gopher's published rule/router ideas, never its artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.features import extract_diagnostic_features  # noqa: E402
from telecom_rca.parser import parse_question  # noqa: E402
from telecom_rca.pipeline import UnifiedHybrid, load_c13_resolver  # noqa: E402
from telecom_rca.routing import Route, route_question  # noqa: E402


class RejectBackend:
    def answer(self, question: str, allowed: set[str]) -> str:
        raise AssertionError("Validation should contain only standard telecom questions")


def published_router(question: str) -> str:
    """Independent transcription of the documented keyword router."""
    lowered = question.lower()
    signals = [
        "rsrp", "sinr", "throughput", "mhz", "tilt", "antenna", "cell", "rbs",
        "handover", "coverage", "interference", "pci",
    ]
    network_score = sum(term in lowered for term in signals)
    if network_score < 2:
        return "general_knowledge"
    if ("how many" in lowered or "value of" in lowered) and network_score < 3:
        return "general_knowledge"
    return "standard_telecom" if re.search(r"\bC[1-8]:", question) else "markdown_telecom"


def published_rule_prediction(features: dict[str, float]) -> str:
    """Independent transcription of the published priority decision tree."""
    if features["speed_max"] > 40:
        return "C7"
    if features["distance_max_km"] > 1:
        return "C2"
    if features["rbs_mean"] < 160:
        return "C8"
    if features["serving_cell_count"] > 2:
        return "C5"
    if features["strong_noncolocated_fraction"] > 0:
        return "C4"
    if features["serving_total_tilt_max"] > 40:
        return "C1"
    if features["mod30_conflict_fraction"] > 0:
        return "C6"
    return "C3"


def main() -> None:
    validation = load_current_csv("validation_questions.csv")
    targets = load_current_csv("validation_target.csv").iloc[::4]["Target"].tolist()
    reference_predictions = []
    for question in validation["question"]:
        parsed = parse_question(question)
        reference_predictions.append(
            published_rule_prediction(extract_diagnostic_features(question, parsed))
        )
    resolver = load_c13_resolver(
        ROOT / "outputs/models/c13_resolver.joblib",
        "62e07383991c679878552b90f187b1948daf1d11a63675a9f06b9f4fe1a9ce26",
    )
    ours = UnifiedHybrid(resolver, RejectBackend())
    our_predictions = [ours.predict(question).semantic_label for question in validation["question"]]

    test = load_current_csv("test.csv")
    expected_routes = [route_question(question).route.value for question in test["question"]]
    reference_routes = [published_router(question) for question in test["question"]]
    reference_accuracy = sum(a == b for a, b in zip(reference_predictions, targets)) / len(targets)
    our_accuracy = sum(a == b for a, b in zip(our_predictions, targets)) / len(targets)
    router_accuracy = sum(a == b for a, b in zip(reference_routes, expected_routes)) / len(test)
    confusion = pd.crosstab(
        pd.Series(targets, name="true"), pd.Series(reference_predictions, name="published_rule")
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "clean-room published router and rule tree only; no Gopher code/model/predictions executed or loaded",
        "official_gopher_external_results": {
            "reproduced_zindi_score": 0.792,
            "unseen_private_score": 0.205,
            "source": "https://zindi.africa/competitions/the-ai-telco-troubleshooting-challenge/discussions/30730",
        },
        "published_rule_validation_accuracy": reference_accuracy,
        "our_hybrid_validation_accuracy": our_accuracy,
        "published_router_test_route_accuracy": router_accuracy,
        "published_router_mismatches": len(test) - sum(a == b for a, b in zip(reference_routes, expected_routes)),
        "confusion": confusion.to_dict(),
    }
    reports = ROOT / "reports"
    (reports / "gopher_reference_benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    markdown = "\n".join(
        [
            "# Gopher clean-room reference benchmark",
            "",
            f"Generated: {report['generated_at_utc']}",
            "",
            "This is not an execution of Gopher's complete submission. It independently",
            "transcribes only the published keyword router and rule priority tree. The",
            "downloaded pickle and submission predictions were never loaded.",
            "",
            "| Measurement | Result |",
            "|---|---:|",
            f"| Published rule tree on our labelled validation | {reference_accuracy:.4%} |",
            f"| Our hybrid on the same validation | {our_accuracy:.4%} |",
            f"| Published keyword router agreement on current test | {router_accuracy:.4%} |",
            f"| Router mismatches | {report['published_router_mismatches']} / {len(test)} |",
            "",
            "Official external results reported by Zindi were 0.792 reproduced and",
            "20.5% on the separate unseen private dataset. Those values are not directly",
            "comparable to this challenge's public score or our local validation.",
            "",
            "The comparison supports structural routing and interval-aware features, but",
            "does not estimate how Gopher's serialized XGBoost component would score here.",
            "",
        ]
    )
    (reports / "gopher_reference_benchmark.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
