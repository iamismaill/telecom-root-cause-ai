"""Verify routing counts and standard-route validation without Qwen inference."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.evaluation import classification_metrics, validation_truth  # noqa: E402
from telecom_rca.features import extract_diagnostic_features  # noqa: E402
from telecom_rca.options import map_standard_cause  # noqa: E402
from telecom_rca.pipeline import load_c13_resolver, sha256_file  # noqa: E402
from telecom_rca.routing import Route, route_question  # noqa: E402


def main() -> None:
    artifact_path = ROOT / "outputs" / "models" / "c13_resolver.joblib"
    artifact_hash = sha256_file(artifact_path)
    resolver = load_c13_resolver(artifact_path, expected_sha256=artifact_hash)

    test = load_current_csv("test.csv")
    routes = [route_question(question).route.value for question in test["question"].astype(str)]
    route_counts = dict(sorted(Counter(routes).items()))

    joined = validation_truth(
        load_current_csv("validation_questions.csv"),
        load_current_csv("validation_target.csv"),
    )
    predictions = []
    for row in joined.itertuples(index=False):
        decision = route_question(row.question)
        if decision.route != Route.STANDARD_TELECOM:
            raise RuntimeError(f"Validation question misrouted: {row.ID} -> {decision.route}")
        features = extract_diagnostic_features(row.question, decision.parsed)
        diagnosis = resolver.predict_one(features)
        displayed = map_standard_cause(row.question, diagnosis.label)
        predictions.append(displayed)
    metrics = classification_metrics(joined["truth"], pd.Series(predictions))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_rows": len(test),
        "test_route_counts": route_counts,
        "validation_rows": len(joined),
        "validation_standard_route_count": len(joined),
        "validation_accuracy": metrics.accuracy,
        "resolver_sha256": artifact_hash,
    }
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "stage6_router.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Stage 6 routed hybrid audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Test routing",
        "",
        "| Route | Count |",
        "|---|---:|",
    ]
    for route, count in route_counts.items():
        lines.append(f"| `{route}` | {count} |")
    lines.extend(
        [
            "",
            "## Labelled validation",
            "",
            f"- Routed to standard telecom: {len(joined)}/{len(joined)}",
            f"- Unified standard-route accuracy: **{metrics.accuracy:.4%}**",
            f"- Resolver SHA-256: `{artifact_hash}`",
            "",
            "No Qwen inference or submission generation was performed by this audit.",
            "",
        ]
    )
    output = "\n".join(lines)
    (report_dir / "stage6_router.md").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

