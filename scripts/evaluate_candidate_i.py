"""Evaluate Candidate I components without generating or changing a submission."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.candidate_i import (  # noqa: E402
    TrainingCaseRetriever,
    assess_standard_evidence,
    offered_semantics,
)
from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.evaluation import validation_truth  # noqa: E402
from telecom_rca.pipeline import UnifiedHybrid, load_c13_resolver  # noqa: E402


class RejectChoiceBackend:
    def answer(self, question: str, allowed: set[str]) -> str:
        raise RuntimeError("Standard validation must not invoke Qwen")


def main() -> None:
    train = load_current_csv("train.csv")
    validation = validation_truth(
        load_current_csv("validation_questions.csv"), load_current_csv("validation_target.csv")
    )
    retriever = TrainingCaseRetriever(k=7).fit(train["question"], train["answer"])
    baseline = UnifiedHybrid(
        load_c13_resolver(ROOT / "outputs/models/c13_resolver.joblib"), RejectChoiceBackend()
    )
    rows = []
    for row in validation.itertuples(index=False):
        base = baseline.predict(row.question).semantic_label
        quality = assess_standard_evidence(row.question)
        retrieved = retriever.predict(row.question, offered_semantics(row.question))
        rows.append(
            {
                "ID": row.ID,
                "truth": row.truth,
                "baseline": base,
                "retrieval": retrieved.label,
                "retrieval_confidence": retrieved.confidence,
                "evidence_complete": quality.complete,
                "evidence_score": quality.score,
            }
        )

    total = len(rows)
    baseline_correct = sum(row["baseline"] == row["truth"] for row in rows)
    retrieval_correct = sum(row["retrieval"] == row["truth"] for row in rows)
    thresholds = {}
    for threshold in (0.60, 0.70, 0.80, 0.90, 1.00):
        selected = [row for row in rows if row["retrieval_confidence"] >= threshold]
        changed = [row for row in selected if row["retrieval"] != row["baseline"]]
        corrected = sum(row["retrieval"] == row["truth"] for row in changed)
        harmed = sum(row["baseline"] == row["truth"] and row["retrieval"] != row["truth"] for row in changed)
        hybrid_correct = baseline_correct + corrected - sum(
            row["baseline"] == row["truth"] for row in changed
        )
        thresholds[str(threshold)] = {
            "selected": len(selected),
            "changed": len(changed),
            "corrected_changes": corrected,
            "harmful_changes": harmed,
            "hybrid_accuracy": hybrid_correct / total,
        }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": total,
        "baseline_accuracy": baseline_correct / total,
        "retrieval_accuracy": retrieval_correct / total,
        "complete_evidence_rows": sum(row["evidence_complete"] for row in rows),
        "retrieval_thresholds": thresholds,
        "decision": "Do not generate Candidate I until an independently justified threshold improves validation without harmful changes.",
    }
    output = ROOT / "reports/candidate_i_experiments.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
