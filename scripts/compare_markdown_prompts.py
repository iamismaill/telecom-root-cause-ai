"""Fixed unlabelled raw-vs-evidence Qwen comparison for Markdown telecom."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.markdown_features import (  # noqa: E402
    evidence_guided_messages,
    evidence_hypothesis,
    extract_markdown_evidence,
    markdown_option_map,
)
from telecom_rca.qwen import LocalQwen, extract_boxed_answer  # noqa: E402
from telecom_rca.routing import Route, route_question  # noqa: E402


def select_distinct(records: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    """Select strongest-margin examples while diversifying hypotheses."""
    ordered = sorted(records, key=lambda row: (-float(row["margin"]), -float(row["score"]), str(row["ID"])))
    selected = []
    used_causes: set[str] = set()
    for row in ordered:
        cause = str(row["hypothesis"])
        if cause not in used_causes and float(row["score"]) > 0:
            selected.append(row)
            used_causes.add(cause)
            if len(selected) == count:
                return selected
    for row in ordered:
        if row not in selected:
            selected.append(row)
            if len(selected) == count:
                break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()
    test = load_current_csv("test.csv")
    candidates = []
    for row in test.itertuples(index=False):
        decision = route_question(row.question)
        if decision.route != Route.MARKDOWN_TELECOM:
            continue
        features = extract_markdown_evidence(decision.parsed)
        hypothesis, score, margin = evidence_hypothesis(features)
        candidates.append(
            {
                "ID": row.ID,
                "question": row.question,
                "features": features,
                "hypothesis": hypothesis,
                "score": score,
                "margin": margin,
            }
        )
    selected = select_distinct(candidates, args.samples)
    runtime = LocalQwen(ROOT / "models" / "Qwen2.5-1.5B-Instruct")
    results = []
    for index, row in enumerate(selected, start=1):
        question = str(row["question"])
        option_map = markdown_option_map(question)
        allowed = set(option_map.values())
        expected_evidence_label = option_map[str(row["hypothesis"])]

        raw = runtime.generate(question, example_choice=sorted(allowed)[0])
        guided = runtime.generate_messages(
            evidence_guided_messages(question, row["features"]), max_new_tokens=16  # type: ignore[arg-type]
        )
        try:
            raw_answer = extract_boxed_answer(raw.text, allowed)
            raw_valid = True
        except ValueError:
            raw_answer = None
            raw_valid = False
        try:
            guided_answer = extract_boxed_answer(guided.text, allowed)
            guided_valid = True
        except ValueError:
            guided_answer = None
            guided_valid = False
        results.append(
            {
                "sample": index,
                "hypothesis": row["hypothesis"],
                "evidence_score": row["score"],
                "evidence_margin": row["margin"],
                "raw_answer": raw_answer,
                "guided_answer": guided_answer,
                "raw_valid": raw_valid,
                "guided_valid": guided_valid,
                "raw_evidence_concordant": raw_answer == expected_evidence_label,
                "guided_evidence_concordant": guided_answer == expected_evidence_label,
                "raw_guided_agree": raw_answer is not None and raw_answer == guided_answer,
                "raw_input_tokens": raw.input_tokens,
                "guided_input_tokens": guided.input_tokens,
                "raw_seconds": raw.elapsed_seconds,
                "guided_seconds": guided.elapsed_seconds,
            }
        )
        print(
            f"completed={index}/{len(selected)} hypothesis={row['hypothesis']} "
            f"raw={raw_answer} guided={guided_answer}",
            flush=True,
        )
    frame = pd.DataFrame(results)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "samples": len(frame),
        "selection": "highest evidence margin with distinct hypotheses",
        "raw_format_success": float(frame["raw_valid"].mean()),
        "guided_format_success": float(frame["guided_valid"].mean()),
        "raw_evidence_concordance": float(frame["raw_evidence_concordant"].mean()),
        "guided_evidence_concordance": float(frame["guided_evidence_concordant"].mean()),
        "raw_guided_agreement": float(frame["raw_guided_agree"].mean()),
        "raw_mean_input_tokens": float(frame["raw_input_tokens"].mean()),
        "guided_mean_input_tokens": float(frame["guided_input_tokens"].mean()),
        "raw_mean_seconds": float(frame["raw_seconds"].mean()),
        "guided_mean_seconds": float(frame["guided_seconds"].mean()),
        "records": results,
    }
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "stage9_markdown_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Stage 9 Markdown raw-vs-evidence comparison",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Fixed samples: {len(frame)} ({report['selection']}).",
        "",
        "| Metric | Raw logs | Evidence guided |",
        "|---|---:|---:|",
        f"| Valid boxed choice | {report['raw_format_success']:.2%} | {report['guided_format_success']:.2%} |",
        f"| Evidence concordance | {report['raw_evidence_concordance']:.2%} | {report['guided_evidence_concordance']:.2%} |",
        f"| Mean input tokens | {report['raw_mean_input_tokens']:.1f} | {report['guided_mean_input_tokens']:.1f} |",
        f"| Mean inference seconds | {report['raw_mean_seconds']:.3f} | {report['guided_mean_seconds']:.3f} |",
        "",
        f"Raw/guided answer agreement: **{report['raw_guided_agreement']:.2%}**.",
        "",
        "Evidence concordance is not accuracy because Markdown labels are unavailable.",
        "No submission file was generated.",
        "",
    ]
    output = "\n".join(lines)
    (report_dir / "stage9_markdown_comparison.md").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

