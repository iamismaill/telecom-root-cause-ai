"""Evaluate the independent deterministic baseline on official validation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.diagnosis import diagnose_standard  # noqa: E402
from telecom_rca.evaluation import classification_metrics, validation_truth  # noqa: E402
from telecom_rca.features import extract_diagnostic_features  # noqa: E402
from telecom_rca.options import map_standard_cause  # noqa: E402
from telecom_rca.parser import parse_question  # noqa: E402


def main() -> None:
    joined = validation_truth(
        load_current_csv("validation_questions.csv"),
        load_current_csv("validation_target.csv"),
    )
    rows: list[dict[str, str]] = []
    for row in joined.itertuples(index=False):
        parsed = parse_question(row.question)
        features = extract_diagnostic_features(row.question, parsed)
        diagnosis = diagnose_standard(features)
        displayed = map_standard_cause(row.question, diagnosis.label)
        rows.append(
            {
                "ID": row.ID,
                "truth": row.truth,
                "prediction": diagnosis.label,
                "displayed": displayed,
                "tier": diagnosis.confidence_tier,
            }
        )
    predictions = pd.DataFrame(rows)
    metrics = classification_metrics(predictions["truth"], predictions["prediction"])
    mapping_accuracy = float((predictions["displayed"] == predictions["truth"]).mean())

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(predictions),
        "semantic_accuracy": metrics.accuracy,
        "displayed_option_accuracy": mapping_accuracy,
        "per_class": metrics.per_class.reset_index().to_dict(orient="records"),
        "confusion": metrics.confusion.to_dict(orient="index"),
        "confidence_tiers": predictions["tier"].value_counts().to_dict(),
    }
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "deterministic_baseline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Deterministic validation baseline",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"- Questions: {len(predictions)}",
        f"- Semantic C1-C8 accuracy: **{metrics.accuracy:.4%}**",
        f"- Displayed-option mapping accuracy: **{mapping_accuracy:.4%}**",
        "",
        "## Per-class results",
        "",
        "| Label | Correct | Support | Recall |",
        "|---|---:|---:|---:|",
    ]
    for label, values in metrics.per_class.iterrows():
        lines.append(f"| {label} | {int(values.correct)} | {int(values.support)} | {values.recall:.4%} |")
    labels = list(metrics.confusion.columns)
    lines.extend(
        [
            "",
            "## Confusion matrix",
            "",
            "| Truth \\ Prediction | " + " | ".join(labels) + " |",
            "|---|" + "---:|" * len(labels),
        ]
    )
    for label, values in metrics.confusion.iterrows():
        lines.append(f"| {label} | " + " | ".join(str(int(values[column])) for column in labels) + " |")
    lines.append("")
    output = "\n".join(lines)
    (report_dir / "deterministic_baseline.md").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
