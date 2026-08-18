"""Evaluate auditable, model-only two-pass Qwen decisions."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.compliant_prompt import condition_model_prompt  # noqa: E402
from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.evaluation import validation_truth  # noqa: E402
from telecom_rca.options import parse_options  # noqa: E402
from telecom_rca.qwen import extract_boxed_answer  # noqa: E402
from telecom_rca.routing import route_question  # noqa: E402


DIAGNOSIS_SYSTEM = (
    "You are a careful 5G root-cause diagnostician. Use only the evidence in the "
    "current prompt. Your entire response must be exactly one final answer in the "
    "form \\boxed{CHOICE}. Never retrieve or match another question."
)
MAPPING_SYSTEM = (
    "You are a multiple-choice language model. Use the diagnosis generated in the "
    "current prompt and map its meaning to one offered option. Generate exactly one "
    "offered label in \\boxed{LABEL}; output nothing else."
)
REPAIR_SYSTEM = (
    "You repair only the format of another language-model response. Preserve its "
    "intended answer and generate exactly one allowed label in \\boxed{LABEL}. "
    "Do not independently solve or replace the diagnosis."
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def chat_prompt(tokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_once(model, tokenizer, sampler, system: str, user: str) -> tuple[str, str]:
    from mlx_lm import generate

    prompt = chat_prompt(tokenizer, system, user)
    raw = generate(
        model, tokenizer, prompt, max_tokens=24, sampler=sampler, verbose=False
    ).strip()
    return raw, sha(prompt)


def mapping_user(question: str, diagnosis_raw: str) -> tuple[str, set[str]]:
    options = parse_options(question)
    allowed = {item.label for item in options}
    lines = "\n".join(f"- {item.label}: {item.description}" for item in options)
    return (
        f"Model-generated semantic diagnosis:\n{diagnosis_raw}\n\n"
        f"Offered choices:\n{lines}\n\n"
        "Map the diagnosis by meaning and generate its exact displayed label.",
        allowed,
    )


def extract_generated_label(raw: str, allowed: set[str]) -> str:
    """Parse only an explicit model-generated label; never infer a replacement."""
    try:
        return extract_boxed_answer(raw, allowed)
    except ValueError:
        exact = raw.strip()
        if exact in allowed:
            return exact
        raise


def select_rows(per_class: int | None) -> pd.DataFrame:
    frame = validation_truth(
        load_current_csv("validation_questions.csv"),
        load_current_csv("validation_target.csv"),
    )
    if per_class is not None:
        frame = frame.groupby("truth", group_keys=False).head(per_class)
    return frame.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    model_path = ROOT / "models/Qwen2.5-1.5B-Instruct-mlx-4bit"
    model, tokenizer = load(str(model_path))
    sampler = make_sampler(temp=0.0)
    selected = select_rows(args.per_class)
    records = []
    started = time.perf_counter()

    for index, row in enumerate(selected.itertuples(index=False), 1):
        diagnosis_raw, diagnosis_prompt_hash = generate_once(
            model,
            tokenizer,
            sampler,
            DIAGNOSIS_SYSTEM,
            condition_model_prompt(row.question),
        )
        map_user, allowed = mapping_user(row.question, diagnosis_raw)
        mapped_raw, mapping_prompt_hash = generate_once(
            model, tokenizer, sampler, MAPPING_SYSTEM, map_user
        )
        repaired_raw = None
        repair_prompt_hash = None
        try:
            answer = extract_generated_label(mapped_raw, allowed)
            valid = True
        except ValueError:
            repair_user = (
                f"Original model response:\n{mapped_raw}\n\n"
                f"Allowed labels: {', '.join(sorted(allowed))}"
            )
            repaired_raw, repair_prompt_hash = generate_once(
                model, tokenizer, sampler, REPAIR_SYSTEM, repair_user
            )
            try:
                answer = extract_generated_label(repaired_raw, allowed)
                valid = True
            except ValueError:
                answer = ""
                valid = False

        diagnosis_match = re.search(r"\bC[1-8]\b", diagnosis_raw.upper())
        semantic_diagnosis = diagnosis_match.group(0) if diagnosis_match else ""
        route = route_question(row.question).route.value
        records.append(
            {
                "ID": row.ID,
                "truth": row.truth,
                "route": route,
                "semantic_diagnosis": semantic_diagnosis,
                "diagnosis_raw": diagnosis_raw,
                "mapped_raw": mapped_raw,
                "repair_raw": repaired_raw,
                "answer": answer,
                "valid": valid,
                "correct": answer == row.truth,
                "diagnosis_prompt_sha256": diagnosis_prompt_hash,
                "mapping_prompt_sha256": mapping_prompt_hash,
                "repair_prompt_sha256": repair_prompt_hash,
            }
        )
        if len(selected) <= 32 or index % 25 == 0:
            print(
                f"{index}/{len(selected)} {row.ID} truth={row.truth} "
                f"diagnosis={diagnosis_raw!r} answer={answer!r}",
                flush=True,
            )

    frame = pd.DataFrame(records)
    by_class = {
        label: {
            "questions": int(len(group)),
            "correct": int(group["correct"].sum()),
            "accuracy": float(group["correct"].mean()),
        }
        for label, group in frame.groupby("truth")
    }
    by_route = {
        route: {
            "questions": int(len(group)),
            "correct": int(group["correct"].sum()),
            "accuracy": float(group["correct"].mean()),
        }
        for route, group in frame.groupby("route")
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "base_parameters": 1_500_000_000,
        "model_quantization": "MLX 4-bit",
        "questions": len(frame),
        "correct": int(frame["correct"].sum()),
        "accuracy": float(frame["correct"].mean()),
        "format_success": float(frame["valid"].mean()),
        "diagnosis_distribution": dict(Counter(frame["semantic_diagnosis"])),
        "answer_distribution": dict(Counter(frame["answer"])),
        "by_class": by_class,
        "by_route": by_route,
        "elapsed_seconds": time.perf_counter() - started,
        "model_generated_all_answers": True,
        "model_generated_format_repairs": True,
        "rule_answer_overrides": False,
        "retrieval": False,
        "cross_dataset_matching": False,
        "systems": {
            "diagnosis": DIAGNOSIS_SYSTEM,
            "mapping": MAPPING_SYSTEM,
            "repair": REPAIR_SYSTEM,
        },
        "records": records,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
