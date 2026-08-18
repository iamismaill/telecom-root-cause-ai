"""Create aggregate class-conditional evidence for independent features."""

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
from telecom_rca.features import extract_diagnostic_features  # noqa: E402
from telecom_rca.parser import parse_question  # noqa: E402


DISPLAY_FEATURES = [
    "low_speed_max",
    "low_rbs_mean",
    "low_distance_max_km",
    "low_serving_total_tilt_max",
    "low_tilt_excess_max_deg",
    "low_neighbor_margin_max_db",
    "low_neighbor_stronger_fraction",
    "low_neighbor_throughput_gain_max",
    "low_strong_noncolocated_fraction",
    "low_close_noncolocated_fraction",
    "low_mod30_conflict_fraction",
    "handover_count",
    "handover_rate",
    "ping_pong_count",
]


def eta_squared(values: pd.Series, labels: pd.Series) -> float:
    valid = values.notna() & labels.notna()
    x, y = values[valid], labels[valid]
    if len(x) < 2:
        return 0.0
    grand = x.mean()
    total = float(((x - grand) ** 2).sum())
    if total == 0:
        return 0.0
    between = sum(len(group) * float(group.mean() - grand) ** 2 for _, group in x.groupby(y))
    return between / total


def main() -> None:
    train = load_current_csv("train.csv")
    records: list[dict[str, object]] = []
    failures: list[str] = []
    for row in train.itertuples(index=False):
        try:
            parsed = parse_question(row.question)
            records.append({"ID": row.ID, "label": row.answer, **extract_diagnostic_features(row.question, parsed)})
        except Exception as exc:
            failures.append(f"{row.ID}: {type(exc).__name__}: {exc}")

    features = pd.DataFrame(records)
    numeric = [c for c in features.columns if c not in {"ID", "label"}]
    medians = features.groupby("label")[numeric].median().round(4)
    missing = features[numeric].isna().mean().sort_values(ascending=False).round(6)
    separation = pd.Series(
        {column: eta_squared(features[column], features["label"]) for column in numeric}
    ).sort_values(ascending=False)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(train),
        "features_extracted": len(features),
        "failures": failures,
        "missing_fraction": missing.to_dict(),
        "eta_squared": separation.round(6).to_dict(),
        "class_medians": medians.to_dict(orient="index"),
    }
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "feature_evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Stage 2 diagnostic feature evidence",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Features extracted: {len(features)}/{len(train)}; failures: {len(failures)}.",
        "",
        "## Strongest univariate class separation",
        "",
        "Eta-squared measures the fraction of feature variance explained by the eight labels; it is descriptive, not validation accuracy.",
        "",
        "| Feature | Eta-squared | Missing fraction |",
        "|---|---:|---:|",
    ]
    for feature, score in separation.head(15).items():
        lines.append(f"| `{feature}` | {score:.4f} | {missing[feature]:.4f} |")
    lines.extend(["", "## Class medians for diagnostic features", ""])
    shown = [feature for feature in DISPLAY_FEATURES if feature in medians]
    lines.append("| Label | " + " | ".join(shown) + " |")
    lines.append("|---|" + "---:|" * len(shown))
    for label, row in medians[shown].iterrows():
        rendered = ["NA" if pd.isna(value) else f"{value:.3f}" for value in row]
        lines.append(f"| {label} | " + " | ".join(rendered) + " |")
    lines.extend(["", "## Important cautions", ""])
    lines.extend(
        [
            "- These statistics use training labels only and do not estimate generalization.",
            "- Correlated features can appear individually strong without being causal.",
            "- Thresholds must be selected using training folds and evaluated once on official validation.",
            "- No winner predictions, model, or submission data were used.",
        ]
    )
    output = "\n".join(lines) + "\n"
    (report_dir / "feature_evidence.md").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
