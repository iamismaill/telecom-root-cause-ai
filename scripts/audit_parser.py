"""Generate an aggregate parser-coverage report without exposing dataset rows."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.parser import parse_question  # noqa: E402


FILES = ("train.csv", "validation_questions.csv", "test.csv")


def audit_file(filename: str) -> dict[str, object]:
    frame = load_current_csv(filename)
    formats: Counter[str] = Counter()
    table_kinds: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    telecom = clean = malformed = 0

    for question in frame["question"].astype(str):
        result = parse_question(question)
        formats[result.format_name] += 1
        table_kinds.update(result.tables.keys())
        table_kinds["unknown"] += len(result.unknown_tables)
        telecom += int(result.is_telecom)
        clean += int(result.parsed_cleanly)
        malformed += sum(len(t.malformed_rows) for t in result.tables.values())
        for error in result.errors:
            error_types[error.split(":", 1)[0]] += 1

    return {
        "rows": len(frame),
        "telecom_rows": telecom,
        "non_telecom_rows": len(frame) - telecom,
        "clean_telecom_rows": clean,
        "formats": dict(sorted(formats.items())),
        "tables": dict(sorted(table_kinds.items())),
        "malformed_table_rows": malformed,
        "error_types": dict(sorted(error_types.items())),
    }


def markdown(report: dict[str, object]) -> str:
    lines = [
        "# Parser coverage report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "This report contains aggregate counts only. Original datasets were not modified.",
        "",
        "| File | Rows | Telecom | Non-telecom | Clean telecom | Malformed rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for filename, data in report["files"].items():
        lines.append(
            f"| `{filename}` | {data['rows']} | {data['telecom_rows']} | "
            f"{data['non_telecom_rows']} | {data['clean_telecom_rows']} | "
            f"{data['malformed_table_rows']} |"
        )
    lines.extend(["", "## Format counts", ""])
    for filename, data in report["files"].items():
        values = ", ".join(f"{k}={v}" for k, v in data["formats"].items())
        lines.append(f"- `{filename}`: {values}")
    lines.extend(["", "## Parser errors", ""])
    for filename, data in report["files"].items():
        values = data["error_types"] or {"none": 0}
        rendered = ", ".join(f"{k}={v}" for k, v in values.items())
        lines.append(f"- `{filename}`: {rendered}")
    return "\n".join(lines) + "\n"


def main() -> None:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": {filename: audit_file(filename) for filename in FILES},
    }
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "parser_coverage.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (report_dir / "parser_coverage.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report), end="")


if __name__ == "__main__":
    main()
