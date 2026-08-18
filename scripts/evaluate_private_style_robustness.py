"""Cross-route metamorphic tests approximating unseen private-format shifts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.markdown_diagnosis import diagnose_markdown  # noqa: E402
from telecom_rca.robustness import (  # noqa: E402
    anonymize_gnodeb_ids,
    normalize_pipe_spacing,
    reverse_table_data_rows,
    rotate_table_columns,
    shift_dates,
    shuffle_and_relabel_options,
    swap_first_two_pipe_tables,
)
from telecom_rca.routing import Route, route_question  # noqa: E402
from telecom_rca.verified_math import solve_verified_math  # noqa: E402


def seed(identifier: str) -> int:
    return int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16)


def main() -> None:
    test = load_current_csv("test.csv")
    markdown = []
    verified_math = []
    for row in test.itertuples(index=False):
        route = route_question(row.question).route
        if route == Route.MARKDOWN_TELECOM:
            markdown.append(row)
        elif route == Route.GENERAL_KNOWLEDGE and solve_verified_math(row.question) is not None:
            verified_math.append(row)

    markdown_transforms = {
        "option_shuffle_relabel": lambda q, s: shuffle_and_relabel_options(q, s),
        "pipe_spacing": lambda q, s: normalize_pipe_spacing(q),
        "date_shift": lambda q, s: shift_dates(q),
        "site_anonymization": lambda q, s: anonymize_gnodeb_ids(q),
        "table_row_reversal": lambda q, s: reverse_table_data_rows(q),
        "table_column_rotation": lambda q, s: rotate_table_columns(q),
        "first_table_swap": lambda q, s: swap_first_two_pipe_tables(q),
    }
    markdown_metrics = {}
    for name, transform in markdown_transforms.items():
        agreement = route_ok = answer_valid = 0
        failures = []
        for row in markdown:
            baseline = diagnose_markdown(row.question)
            try:
                changed = transform(row.question, seed(row.ID))
                decision = route_question(changed)
                predicted = diagnose_markdown(changed)
                agreement += predicted.semantic_cause == baseline.semantic_cause
                route_ok += decision.route == Route.MARKDOWN_TELECOM
                answer_valid += predicted.displayed_answer in {o.label for o in decision.options}
            except Exception as exc:
                failures.append(f"{row.ID}: {type(exc).__name__}: {exc}")
        markdown_metrics[name] = {
            "rows": len(markdown),
            "semantic_agreement": agreement / len(markdown),
            "route_accuracy": route_ok / len(markdown),
            "answer_domain_accuracy": answer_valid / len(markdown),
            "failures": len(failures),
            "failure_examples": failures[:5],
        }

    math_agreement = math_route = math_answer_valid = 0
    math_failures = []
    for row in verified_math:
        baseline = solve_verified_math(row.question)
        try:
            changed = shuffle_and_relabel_options(row.question, seed(row.ID))
            decision = route_question(changed)
            predicted = solve_verified_math(changed)
            if baseline is None or predicted is None:
                raise ValueError("Verified solver disappeared after option perturbation")
            math_agreement += predicted.value == baseline.value and predicted.solver == baseline.solver
            math_route += decision.route == Route.GENERAL_KNOWLEDGE
            math_answer_valid += predicted.label in {o.label for o in decision.options}
        except Exception as exc:
            math_failures.append(f"{row.ID}: {type(exc).__name__}: {exc}")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "private-style metamorphic robustness; no transformed data used for training or submission",
        "markdown_rows": len(markdown),
        "verified_math_rows": len(verified_math),
        "markdown_transformations": markdown_metrics,
        "verified_math_option_perturbation": {
            "semantic_agreement": math_agreement / len(verified_math),
            "route_accuracy": math_route / len(verified_math),
            "answer_domain_accuracy": math_answer_valid / len(verified_math),
            "failures": len(math_failures),
            "failure_examples": math_failures[:5],
        },
    }
    path = ROOT / "reports/private_style_robustness.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Private-style cross-route robustness",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "| Markdown transformation | Semantic agreement | Route accuracy | Answer-domain accuracy | Failures |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in markdown_metrics.items():
        lines.append(
            f"| `{name}` | {values['semantic_agreement']:.2%} | {values['route_accuracy']:.2%} | "
            f"{values['answer_domain_accuracy']:.2%} | {values['failures']} |"
        )
    math_values = report["verified_math_option_perturbation"]
    lines.extend(
        [
            "",
            "## Verified mathematics",
            "",
            f"Option shuffle/relabel semantic agreement: **{math_values['semantic_agreement']:.2%}**; "
            f"route accuracy: **{math_values['route_accuracy']:.2%}**; failures: **{math_values['failures']}**.",
            "",
            "These transformations are evaluation-only and use current challenge questions.",
            "",
        ]
    )
    markdown_path = ROOT / "reports/private_style_robustness.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
