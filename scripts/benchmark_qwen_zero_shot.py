"""Balanced, bounded zero-shot benchmark on official validation labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

import pandas as pd
import psutil
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.evaluation import validation_truth  # noqa: E402
from telecom_rca.qwen import LocalQwen, extract_boxed_answer  # noqa: E402


SEED = 42


def balanced_sample(frame: pd.DataFrame, per_class: int) -> pd.DataFrame:
    pieces = []
    for label, group in frame.groupby("truth", sort=True):
        pieces.append(group.sample(n=min(per_class, len(group)), random_state=SEED))
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-class", type=int, default=2)
    args = parser.parse_args()
    random.seed(SEED)

    joined = validation_truth(
        load_current_csv("validation_questions.csv"),
        load_current_csv("validation_target.csv"),
    )
    sample = balanced_sample(joined, args.samples_per_class)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    runtime = LocalQwen(ROOT / "models" / "Qwen2.5-1.5B-Instruct")
    rss_loaded = process.memory_info().rss
    mps_driver_loaded = int(torch.mps.driver_allocated_memory()) if runtime.device.type == "mps" else None

    records = []
    allowed = {f"C{i}" for i in range(1, 9)}
    for row in sample.itertuples(index=False):
        result = runtime.generate(row.question)
        try:
            raw_answer = extract_boxed_answer(result.text)
            boxed_syntax = True
            if raw_answer in allowed:
                prediction = raw_answer
                exact_target_format = True
                parse_error = None
            elif raw_answer in {str(i) for i in range(1, 9)}:
                prediction = f"C{raw_answer}"
                exact_target_format = False
                parse_error = "Numeric choice normalized to semantic C-label"
            else:
                prediction = None
                exact_target_format = False
                parse_error = f"Unsupported boxed answer: {raw_answer!r}"
        except ValueError as exc:
            raw_answer = None
            prediction = None
            boxed_syntax = False
            exact_target_format = False
            parse_error = str(exc)
        records.append(
            {
                "ID": row.ID,
                "truth": row.truth,
                "raw_answer": raw_answer,
                "prediction": prediction,
                "correct": prediction == row.truth,
                "boxed_syntax": boxed_syntax,
                "exact_target_format": exact_target_format,
                "parse_error": parse_error,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "elapsed_seconds": result.elapsed_seconds,
            }
        )
        print(
            f"completed={len(records)}/{len(sample)} truth={row.truth} "
            f"prediction={prediction} seconds={result.elapsed_seconds:.3f}",
            flush=True,
        )

    results = pd.DataFrame(records)
    total_seconds = float(results["elapsed_seconds"].sum())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "device": runtime.device.type,
        "seed": SEED,
        "sampling": f"balanced {args.samples_per_class} per class",
        "rows": len(results),
        "semantic_accuracy": float(results["correct"].mean()),
        "boxed_syntax_success": float(results["boxed_syntax"].mean()),
        "exact_target_format_success": float(results["exact_target_format"].mean()),
        "mean_seconds": float(results["elapsed_seconds"].mean()),
        "total_seconds": total_seconds,
        "input_tokens": {
            "min": int(results["input_tokens"].min()),
            "median": float(results["input_tokens"].median()),
            "max": int(results["input_tokens"].max()),
        },
        "generated_tokens_per_second": float(results["output_tokens"].sum() / total_seconds),
        "rss_before_bytes": rss_before,
        "rss_after_load_bytes": rss_loaded,
        "rss_load_delta_bytes": rss_loaded - rss_before,
        "mps_driver_allocated_after_load_bytes": mps_driver_loaded,
        "per_class": results.groupby("truth")["correct"].agg(["sum", "count", "mean"]).to_dict(orient="index"),
    }
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "qwen_zero_shot_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Qwen2.5-1.5B-Instruct zero-shot benchmark",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"- Device: `{report['device']}`",
        f"- Sampling: {report['sampling']} ({report['rows']} questions)",
        f"- Semantic accuracy after safe numeric normalization: **{report['semantic_accuracy']:.4%}**",
        f"- Boxed-syntax success: **{report['boxed_syntax_success']:.4%}**",
        f"- Exact validation target-format success: **{report['exact_target_format_success']:.4%}**",
        f"- Mean inference time: **{report['mean_seconds']:.3f} seconds/question**",
        f"- Generated-token throughput: **{report['generated_tokens_per_second']:.3f} tokens/second**",
        f"- MPS driver memory after model load: **{report['mps_driver_allocated_after_load_bytes'] / 1024**3:.3f} GiB**",
        "",
        "This is a bounded smoke benchmark, not the full validation score.",
        "",
        "| Label | Correct | Count | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for label, values in report["per_class"].items():
        lines.append(f"| {label} | {int(values['sum'])} | {int(values['count'])} | {values['mean']:.4%} |")
    output = "\n".join(lines) + "\n"
    (report_dir / "qwen_zero_shot_benchmark.md").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
