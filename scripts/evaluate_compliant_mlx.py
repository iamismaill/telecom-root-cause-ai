"""Evaluate model-only MLX inference on labelled validation questions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.evaluation import validation_truth  # noqa: E402
from telecom_rca.options import map_standard_cause  # noqa: E402
from telecom_rca.qwen import extract_boxed_answer  # noqa: E402
from scripts.prepare_compliant_sft import SYSTEM  # noqa: E402


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter")
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--rubric", action="store_true")
    parser.add_argument("--conditions", action="store_true")
    parser.add_argument("--enhanced-conditions", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=12)
    args = parser.parse_args()

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model_path = ROOT / "models/Qwen2.5-1.5B-Instruct-mlx-4bit"
    adapter_path = ROOT / args.adapter if args.adapter else None
    model, tokenizer = load(
        str(model_path),
        adapter_path=str(adapter_path) if adapter_path else None,
    )
    joined = validation_truth(
        load_current_csv("validation_questions.csv"),
        load_current_csv("validation_target.csv"),
    )
    selected = (
        joined.groupby("truth", group_keys=False)
        .head(args.per_class)
        .reset_index(drop=True)
    )
    sampler = make_sampler(temp=0.0)
    records = []
    if args.compact or args.rubric or args.conditions or args.enhanced_conditions:
        from telecom_rca.compliant_prompt import (
            compact_model_prompt,
            condition_model_prompt,
            enhanced_condition_model_prompt,
            rubric_model_prompt,
        )
        from scripts.prepare_compliant_sft_v2 import SYSTEM as compact_system

    for index, row in enumerate(selected.itertuples(index=False), 1):
        if args.enhanced_conditions:
            user_content = enhanced_condition_model_prompt(row.question)
        elif args.conditions:
            user_content = condition_model_prompt(row.question)
        elif args.rubric:
            user_content = rubric_model_prompt(row.question)
        elif args.compact:
            user_content = compact_model_prompt(row.question)
        else:
            user_content = row.question
        messages = [
            {
                "role": "system",
                "content": (
                    compact_system
                    if (
                        args.compact
                        or args.rubric
                        or args.conditions
                        or args.enhanced_conditions
                    )
                    else SYSTEM
                ),
            },
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        start = time.perf_counter()
        raw = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        ).strip()
        elapsed = time.perf_counter() - start
        allowed = {f"C{i}" for i in range(1, 9)}
        try:
            answer = extract_boxed_answer(raw, allowed)
            valid = True
        except ValueError:
            answer = ""
            valid = False
        records.append({
            "ID": row.ID,
            "truth": row.truth,
            "model_answer": answer,
            "raw_generation": raw,
            "valid_boxed_answer": valid,
            "correct": answer == row.truth,
            "elapsed_seconds": elapsed,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        })
        print(
            f"{index}/{len(selected)} id={row.ID} truth={row.truth} "
            f"answer={answer!r} raw={raw!r}",
            flush=True,
        )
    frame = pd.DataFrame(records)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "base_parameters": 1_500_000_000,
        "adapter": str(adapter_path.relative_to(ROOT)) if adapter_path else None,
        "adapter_sha256": (
            file_hash(adapter_path / "adapters.safetensors") if adapter_path else None
        ),
        "questions": len(frame),
        "accuracy": float(frame["correct"].mean()),
        "format_success": float(frame["valid_boxed_answer"].mean()),
        "mean_seconds": float(frame["elapsed_seconds"].mean()),
        "final_answers_generated_by_model": True,
        "rule_overrides": False,
        "retrieval": False,
        "compact_current_question_evidence": args.compact,
        "model_applied_engineering_rubric": args.rubric,
        "model_applied_condition_results": args.conditions,
        "model_applied_enhanced_conditions": args.enhanced_conditions,
        "records": records,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
