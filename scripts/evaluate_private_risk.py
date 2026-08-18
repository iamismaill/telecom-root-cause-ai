"""Offline private-leaderboard risk assessment for frozen Candidates F and J.

This is an uncertainty model, not an attempt to infer hidden labels. Measured
quantities and explicit scenario assumptions are kept separate in the report.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.general_knowledge import categorize_general_question  # noqa: E402
from telecom_rca.markdown_diagnosis import diagnose_markdown  # noqa: E402
from telecom_rca.markdown_features import evidence_hypothesis, extract_markdown_evidence  # noqa: E402
from telecom_rca.parser import parse_question  # noqa: E402
from telecom_rca.qwen import extract_boxed_answer  # noqa: E402
from telecom_rca.routing import Route, route_question  # noqa: E402


SEED = 20260723
SCENARIOS = ("pessimistic", "central", "optimistic")
STANDARD_PROBABILITY = {
    "F": {"pessimistic": 0.960, "central": 0.983796, "optimistic": 0.993},
    "J": {"pessimistic": 0.970, "central": 0.990741, "optimistic": 0.996},
}
MARKDOWN_PROBABILITY = {
    "strong": {"pessimistic": 0.82, "central": 0.94, "optimistic": 0.985},
    "moderate": {"pessimistic": 0.65, "central": 0.82, "optimistic": 0.94},
    "fragile": {"pessimistic": 0.45, "central": 0.65, "optimistic": 0.85},
}
GK_PROBABILITY = {
    "verified_exact": {"pessimistic": 0.97, "central": 0.995, "optimistic": 0.999},
    "unanimous": {"pessimistic": 0.65, "central": 0.86, "optimistic": 0.95},
    "majority": {"pessimistic": 0.50, "central": 0.72, "optimistic": 0.88},
    "disputed": {"pessimistic": 0.30, "central": 0.50, "optimistic": 0.72},
}


def base_answers(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    first = frame[frame["ID"].astype(str).str.endswith("_1")].copy()
    first["base_id"] = first["ID"].astype(str).str.rsplit("_", n=1).str[0]
    return dict(zip(first["base_id"], first["Target"].astype(str)))


def answer_label(value: str, question: str) -> str:
    allowed = {option.label for option in route_question(question).options}
    return extract_boxed_answer(value, allowed)


def markdown_tier(question: str) -> tuple[str, dict[str, object]]:
    parsed = parse_question(question)
    features = extract_markdown_evidence(parsed)
    hypothesis, score, margin = evidence_hypothesis(features)
    diagnosis = diagnose_markdown(question)
    concordant = hypothesis == diagnosis.semantic_cause
    if concordant and score >= 2.5 and margin >= 1.0:
        tier = "strong"
    elif concordant and score >= 1.5 and margin >= 0.5:
        tier = "moderate"
    else:
        tier = "fragile"
    return tier, {
        "diagnosis": diagnosis.semantic_cause,
        "evidence_hypothesis": hypothesis,
        "evidence_score": score,
        "evidence_margin": margin,
        "concordant": concordant,
    }


def gk_tier(
    identifier: str,
    question: str,
    answers: dict[str, dict[str, str]],
    verified_ids: set[str],
) -> tuple[str, dict[str, object]]:
    labels = {
        candidate: answer_label(candidate_answers[identifier], question)
        for candidate, candidate_answers in answers.items()
    }
    counts = pd.Series(list(labels.values())).value_counts()
    agreement = int(counts.iloc[0])
    if identifier in verified_ids:
        tier = "verified_exact"
    elif agreement == len(labels):
        tier = "unanimous"
    elif agreement >= 3:
        tier = "majority"
    else:
        tier = "disputed"
    return tier, {
        "category": categorize_general_question(question),
        "candidate_answers": labels,
        "agreement": agreement,
        "distinct_answers": int(counts.size),
    }


def simulate_private(
    probabilities: np.ndarray,
    private_size: int = 604,
    iterations: int = 50_000,
) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    totals = np.empty(iterations, dtype=np.int16)
    population = len(probabilities)
    for iteration in range(iterations):
        selected = rng.choice(population, size=private_size, replace=False)
        totals[iteration] = rng.binomial(1, probabilities[selected]).sum()
    quantiles = np.quantile(totals, [0.05, 0.5, 0.95])
    return {
        "mean_correct": float(totals.mean()),
        "mean_accuracy": float(totals.mean() / private_size),
        "p05_correct": int(quantiles[0]),
        "median_correct": int(quantiles[1]),
        "p95_correct": int(quantiles[2]),
        "probability_at_least_95_percent": float(np.mean(totals >= np.ceil(0.95 * private_size))),
        "probability_at_least_97_percent": float(np.mean(totals >= np.ceil(0.97 * private_size))),
    }


def main() -> None:
    test = load_current_csv("test.csv")
    answer_files = {
        "C": ROOT / "outputs/submissions/candidate_c_markdown_decoder.csv",
        "D": ROOT / "outputs/submissions/candidate_d_gk_two_pass.csv",
        "E": ROOT / "outputs/submissions/candidate_e_gk_option_text.csv",
        "F": ROOT / "outputs/submissions/candidate_f_verified_math.csv",
        "J": ROOT / "outputs/private_candidates/candidate_j_unanimous_all_cause.csv",
    }
    answers = {name: base_answers(path) for name, path in answer_files.items()}
    f_manifest = json.loads(
        (ROOT / "outputs/submissions/candidate_f_manifest.json").read_text(encoding="utf-8")
    )
    verified_ids = {proof["ID"] for proof in f_manifest["proofs"]}

    records: list[dict[str, object]] = []
    for row in test.itertuples(index=False):
        decision = route_question(row.question)
        record: dict[str, object] = {"ID": row.ID, "route": decision.route.value}
        if decision.route == Route.STANDARD_TELECOM:
            record.update({"risk_tier": "labelled_standard", "evidence_detail": "official validation"})
        elif decision.route == Route.MARKDOWN_TELECOM:
            tier, detail = markdown_tier(row.question)
            record.update({"risk_tier": tier, **detail})
        else:
            tier, detail = gk_tier(row.ID, row.question, {k: answers[k] for k in ("C", "D", "E", "F")}, verified_ids)
            record.update({"risk_tier": tier, **detail})
        records.append(record)
    audit = pd.DataFrame(records)

    route_counts = audit["route"].value_counts().to_dict()
    tier_counts = (
        audit.groupby(["route", "risk_tier"]).size().rename("questions").reset_index().to_dict("records")
    )
    simulations: dict[str, dict[str, dict[str, float]]] = {"F": {}, "J": {}}
    route_expectations: dict[str, dict[str, dict[str, float]]] = {"F": {}, "J": {}}
    for candidate in ("F", "J"):
        for scenario in SCENARIOS:
            probabilities = []
            per_route: dict[str, list[float]] = {}
            for record in records:
                route = str(record["route"])
                tier = str(record["risk_tier"])
                if route == Route.STANDARD_TELECOM.value:
                    probability = STANDARD_PROBABILITY[candidate][scenario]
                elif route == Route.MARKDOWN_TELECOM.value:
                    probability = MARKDOWN_PROBABILITY[tier][scenario]
                else:
                    probability = GK_PROBABILITY[tier][scenario]
                probabilities.append(probability)
                per_route.setdefault(route, []).append(probability)
            vector = np.asarray(probabilities)
            simulations[candidate][scenario] = simulate_private(vector)
            route_expectations[candidate][scenario] = {
                route: float(np.mean(values)) for route, values in per_route.items()
            }

    public = {
        "questions": 259,
        "F": {"score": 0.945945945, "correct": 245, "incorrect": 14},
        "J": {"score": 0.949806949, "correct": 246, "incorrect": 13},
        "interpretation": "measured aggregate only; route membership and labels are hidden",
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "test_questions": len(test),
        "assumed_private_questions": 604,
        "route_counts": route_counts,
        "risk_tier_counts": tier_counts,
        "measured_public": public,
        "scenario_probabilities": {
            "standard": STANDARD_PROBABILITY,
            "markdown": MARKDOWN_PROBABILITY,
            "general_knowledge": GK_PROBABILITY,
        },
        "route_expected_accuracy": route_expectations,
        "private_simulations": simulations,
        "limitations": [
            "Only standard telecom has supplied local labels.",
            "Markdown and general-knowledge probabilities are explicit sensitivity assumptions, not measurements.",
            "The public/private split may not be a random sample.",
            "Competitor private performance is unknowable; this report cannot estimate win probability reliably.",
        ],
        "frozen_candidates_modified": False,
    }
    reports = ROOT / "reports"
    audit.to_csv(reports / "private_risk_question_audit.csv", index=False)
    (reports / "private_risk_assessment.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = [
        "# Offline private-risk assessment",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "Candidate F and Candidate J were not modified.",
        "",
        "## Route census",
        "",
        "| Route | Questions |",
        "|---|---:|",
    ]
    for route, count in route_counts.items():
        lines.append(f"| `{route}` | {count} |")
    lines.extend(
        [
            "",
            "## Risk tiers",
            "",
            "| Route | Tier | Questions |",
            "|---|---|---:|",
        ]
    )
    for item in tier_counts:
        lines.append(f"| `{item['route']}` | `{item['risk_tier']}` | {item['questions']} |")
    lines.extend(
        [
            "",
            "## Simulated private outcomes (604 questions)",
            "",
            "| Candidate | Scenario | Mean accuracy | Mean correct | 5th–95th percentile correct | P(≥95%) | P(≥97%) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in ("F", "J"):
        for scenario in SCENARIOS:
            value = simulations[candidate][scenario]
            lines.append(
                f"| {candidate} | {scenario} | {value['mean_accuracy']:.2%} | "
                f"{value['mean_correct']:.1f} | {value['p05_correct']}–{value['p95_correct']} | "
                f"{value['probability_at_least_95_percent']:.1%} | "
                f"{value['probability_at_least_97_percent']:.1%} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Standard telecom is the strongest route because it has labelled validation and grouped CV.",
            "- Markdown and general knowledge dominate uncertainty because no local labels are provided for them.",
            "- Scenario results are sensitivity analysis, not hidden-label predictions.",
            "- Public accuracy is a measured aggregate constraint: F is 245/259 and J is 246/259.",
            "- The private split may differ from a random sample, so scenario ranges are more useful than a single forecast.",
            "",
        ]
    )
    (reports / "private_risk_assessment.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
