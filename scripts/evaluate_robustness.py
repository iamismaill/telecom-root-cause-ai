"""Run full metamorphic robustness evaluation on official validation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.evaluation import validation_truth  # noqa: E402
from telecom_rca.options import map_standard_cause  # noqa: E402
from telecom_rca.pipeline import UnifiedHybrid, load_c13_resolver  # noqa: E402
from telecom_rca.robustness import (  # noqa: E402
    anonymize_gnodeb_ids,
    combined_stress,
    drop_irrelevant_engineering_columns,
    normalize_pipe_spacing,
    rename_supported_columns,
    shift_dates,
    shuffle_and_relabel_options,
    swap_first_two_pipe_tables,
)
from telecom_rca.routing import Route  # noqa: E402


class RejectChoiceBackend:
    def answer(self, question: str, allowed: set[str]) -> str:
        raise RuntimeError("Robustness validation must never call Qwen")


def stable_seed(identifier: str) -> int:
    return int(hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:8], 16)


def main() -> None:
    resolver = load_c13_resolver(ROOT / "outputs" / "models" / "c13_resolver.joblib")
    pipeline = UnifiedHybrid(resolver, RejectChoiceBackend())
    joined = validation_truth(
        load_current_csv("validation_questions.csv"),
        load_current_csv("validation_target.csv"),
    )
    baseline = {
        row.ID: pipeline.predict(row.question)
        for row in joined.itertuples(index=False)
    }
    transforms = {
        "option_shuffle_relabel": lambda q, seed: shuffle_and_relabel_options(q, seed),
        "table_order_swap": lambda q, seed: swap_first_two_pipe_tables(q),
        "pipe_spacing": lambda q, seed: normalize_pipe_spacing(q),
        "date_shift": lambda q, seed: shift_dates(q),
        "site_id_anonymization": lambda q, seed: anonymize_gnodeb_ids(q),
        "supported_column_renaming": lambda q, seed: rename_supported_columns(q),
        "irrelevant_column_removal": lambda q, seed: drop_irrelevant_engineering_columns(q),
        "combined": combined_stress,
    }
    metrics: dict[str, dict[str, object]] = {}
    for name, transform in transforms.items():
        correct = agreement = displayed_correct = route_correct = 0
        failures: list[str] = []
        for row in joined.itertuples(index=False):
            try:
                changed = transform(row.question, stable_seed(row.ID))
                prediction = pipeline.predict(changed)
                expected_display = map_standard_cause(changed, row.truth)
                correct += int(prediction.semantic_label == row.truth)
                agreement += int(prediction.semantic_label == baseline[row.ID].semantic_label)
                displayed_correct += int(prediction.answer == expected_display)
                route_correct += int(prediction.route == Route.STANDARD_TELECOM)
            except Exception as exc:
                failures.append(f"{row.ID}: {type(exc).__name__}: {exc}")
        total = len(joined)
        metrics[name] = {
            "rows": total,
            "semantic_accuracy": correct / total,
            "baseline_agreement": agreement / total,
            "displayed_option_accuracy": displayed_correct / total,
            "route_accuracy": route_correct / total,
            "failure_count": len(failures),
            "failure_examples": failures[:10],
        }
        print(f"completed={name} metrics={metrics[name]}", flush=True)

    baseline_accuracy = sum(
        prediction.semantic_label == truth
        for prediction, truth in zip(baseline.values(), joined["truth"])
    ) / len(joined)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(joined),
        "baseline_accuracy": baseline_accuracy,
        "transformations": metrics,
    }
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "stage7_robustness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Stage 7 robustness evaluation",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Baseline accuracy: **{baseline_accuracy:.4%}** on {len(joined)} questions.",
        "",
        "| Transformation | Semantic accuracy | Baseline agreement | Displayed-option accuracy | Route accuracy | Failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        lines.append(
            f"| `{name}` | {values['semantic_accuracy']:.4%} | {values['baseline_agreement']:.4%} | "
            f"{values['displayed_option_accuracy']:.4%} | {values['route_accuracy']:.4%} | "
            f"{values['failure_count']} |"
        )
    lines.extend(
        [
            "",
            "All transformations are validation-only and preserve the intended answer.",
            "No transformed data was used for training or submission generation.",
            "",
        ]
    )
    output = "\n".join(lines)
    (report_dir / "stage7_robustness.md").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
