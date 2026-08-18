"""Aggregate Stage 8 audit; extracts evidence but generates no answers."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.general_knowledge import categorize_general_question  # noqa: E402
from telecom_rca.markdown_features import (  # noqa: E402
    compact_evidence_summary,
    evidence_flags,
    extract_markdown_evidence,
    markdown_option_map,
)
from telecom_rca.routing import Route, route_question  # noqa: E402


def main() -> None:
    test = load_current_csv("test.csv")
    markdown_records = []
    gk_records = []
    failures = []
    for row in test.itertuples(index=False):
        decision = route_question(row.question)
        try:
            if decision.route == Route.MARKDOWN_TELECOM:
                features = extract_markdown_evidence(decision.parsed)
                markdown_option_map(row.question)
                markdown_records.append({"ID": row.ID, **features, **{f"flag_{k}": v for k, v in evidence_flags(features).items()}})
            elif decision.route == Route.GENERAL_KNOWLEDGE:
                gk_records.append(
                    {
                        "ID": row.ID,
                        "category": categorize_general_question(row.question),
                        "characters": len(row.question),
                        "option_count": len(decision.options),
                        "question": row.question,
                    }
                )
        except Exception as exc:
            failures.append(f"{row.ID}: {type(exc).__name__}: {exc}")

    markdown = pd.DataFrame(markdown_records)
    gk = pd.DataFrame(gk_records)
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "models" / "Qwen2.5-1.5B-Instruct", local_files_only=True
    )
    gk["tokens"] = [len(tokenizer.encode(question)) for question in gk.pop("question")]
    flag_columns = [column for column in markdown if column.startswith("flag_")]
    numeric = [column for column in markdown if column != "ID" and not column.startswith("flag_")]
    quantiles = {
        column: {
            "min": float(markdown[column].min()),
            "median": float(markdown[column].median()),
            "max": float(markdown[column].max()),
        }
        for column in numeric
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "markdown_rows": len(markdown),
        "markdown_failures": failures,
        "markdown_flag_counts": {column.removeprefix("flag_"): int(markdown[column].sum()) for column in flag_columns},
        "markdown_feature_ranges": quantiles,
        "gk_rows": len(gk),
        "gk_categories": dict(sorted(Counter(gk["category"]).items())),
        "gk_option_counts": {str(k): int(v) for k, v in sorted(Counter(gk["option_count"]).items())},
        "gk_token_lengths": {
            "min": int(gk["tokens"].min()),
            "median": float(gk["tokens"].median()),
            "max": int(gk["tokens"].max()),
        },
    }
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "stage8_evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Stage 8 evidence audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Markdown telecom",
        "",
        f"- Records: {len(markdown)}",
        f"- Evidence extraction failures: {len(failures)}",
        "- All nine semantic choices mapped independently of displayed labels.",
        "",
        "| Evidence flag | Records |",
        "|---|---:|",
    ]
    for flag, count in report["markdown_flag_counts"].items():
        lines.append(f"| `{flag}` | {count} |")
    lines.extend(
        [
            "",
            "## General knowledge",
            "",
            f"- Records: {len(gk)}",
            f"- Token lengths: min={report['gk_token_lengths']['min']}, "
            f"median={report['gk_token_lengths']['median']:.0f}, max={report['gk_token_lengths']['max']}",
            "",
            "| Category | Records |",
            "|---|---:|",
        ]
    )
    for category, count in report["gk_categories"].items():
        lines.append(f"| `{category}` | {count} |")
    lines.extend(
        [
            "",
            "This audit does not generate or infer test answers. The Markdown set has no supplied labels, so evidence flags are not accuracy claims.",
            "",
        ]
    )
    output = "\n".join(lines)
    (report_dir / "stage8_evidence.md").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

