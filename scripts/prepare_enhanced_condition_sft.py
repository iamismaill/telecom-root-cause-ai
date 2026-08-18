"""Prepare enhanced C1/C3 evidence SFT data without validation leakage."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.compliant_prompt import enhanced_condition_model_prompt  # noqa: E402
from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.evaluation import validation_truth  # noqa: E402
from telecom_rca.options import map_standard_cause  # noqa: E402
from telecom_rca.robustness import shuffle_and_relabel_options  # noqa: E402


SYSTEM = (
    "You are a careful 5G root-cause diagnostician. Use only the evidence in the "
    "current prompt. Your entire response must be exactly one final answer in the "
    "form \\boxed{CHOICE}. Never retrieve or match another question."
)


def seed(identifier: str, salt: str) -> int:
    return int(
        hashlib.sha256(f"{salt}:{identifier}".encode()).hexdigest()[:8], 16
    )


def record(question: str, semantic: str) -> dict:
    displayed = map_standard_cause(question, semantic)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": enhanced_condition_model_prompt(question)},
            {"role": "assistant", "content": f"\\boxed{{{displayed}}}"},
        ]
    }


def build(frame, augment: bool) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        semantic = str(row.answer).upper()
        grouped[semantic].append(record(row.question, semantic))
        if augment:
            changed = shuffle_and_relabel_options(
                row.question, seed(str(row.ID), "enhanced-condition")
            )
            grouped[semantic].append(record(changed, semantic))
            if semantic in {"C1", "C3"}:
                focused = shuffle_and_relabel_options(
                    row.question, seed(str(row.ID), "enhanced-c13-focus")
                )
                grouped[semantic].append(record(focused, semantic))
    labels = sorted(grouped)
    rows = []
    for index in range(max(map(len, grouped.values()))):
        for label in labels:
            if index < len(grouped[label]):
                rows.append(grouped[label][index])
    return rows


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    train = load_current_csv("train.csv")[["ID", "question", "answer"]]
    validation = validation_truth(
        load_current_csv("validation_questions.csv"),
        load_current_csv("validation_target.csv"),
    ).rename(columns={"truth": "answer"})
    output = ROOT / "outputs/compliant_sft/qwen25_15b_enhanced_conditions_v1"
    train_rows = build(train, augment=True)
    valid_rows = build(validation, augment=False)
    write(output / "train.jsonl", train_rows)
    write(output / "valid.jsonl", valid_rows)
    manifest = {
        "train_examples": len(train_rows),
        "validation_examples": len(valid_rows),
        "validation_used_for_training": False,
        "focused_causes": ["C1", "C3"],
        "continuous_evidence": [
            "tilt excess",
            "total/mechanical tilt",
            "serving RSRP",
            "neighbor margin and fractions",
        ],
        "retrieval": False,
        "cross_dataset_matching": False,
        "rule_answer_override": False,
        "model_generated_final_answer": True,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
