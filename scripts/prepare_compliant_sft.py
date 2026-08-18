"""Prepare rule-compliant model fine-tuning data from official labelled files.

Training examples are used for supervised fine-tuning only. No test question
is matched against, augmented from, or included with a label.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.evaluation import validation_truth  # noqa: E402
from telecom_rca.options import map_standard_cause  # noqa: E402
from telecom_rca.robustness import shuffle_and_relabel_options  # noqa: E402


SYSTEM = (
    "You are a careful multiple-choice 5G troubleshooting assistant. Analyze only "
    "the content of the current question. Select one offered choice. Your entire "
    "response must be exactly one answer in the form \\boxed{CHOICE}, with no "
    "explanation or additional text."
)


def stable_seed(identifier: str, salt: str) -> int:
    value = hashlib.sha256(f"{salt}:{identifier}".encode()).hexdigest()
    return int(value[:8], 16)


def record(question: str, answer: str) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": rf"\boxed{{{answer}}}"},
        ]
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_rows(frame: pd.DataFrame, augment: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in frame.itertuples(index=False):
        identifier, question, semantic = str(item.ID), str(item.question), str(item.answer)
        rows.append(record(question, map_standard_cause(question, semantic)))
        if augment:
            changed = shuffle_and_relabel_options(
                question, stable_seed(identifier, "option-relabel")
            )
            rows.append(record(changed, map_standard_cause(changed, semantic)))
    return rows


def main() -> None:
    train = load_current_csv("train.csv")[["ID", "question", "answer"]]
    validation = validation_truth(
        load_current_csv("validation_questions.csv"),
        load_current_csv("validation_target.csv"),
    ).rename(columns={"truth": "answer"})
    train_rows = build_rows(train, augment=True)
    valid_rows = build_rows(validation, augment=True)
    output = ROOT / "outputs/compliant_sft/qwen25_15b"
    write_jsonl(output / "train.jsonl", train_rows)
    write_jsonl(output / "valid.jsonl", valid_rows)
    test_path = output / "test.jsonl"
    if test_path.exists():
        test_path.unlink()
    manifest = {
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "parameter_count": 1_500_000_000,
        "training_source": "current_challenge_data/train.csv only",
        "validation_source": "validation_questions.csv + validation_target.csv",
        "test_labels_used": False,
        "retrieval_used": False,
        "train_examples": len(train_rows),
        "validation_examples": len(valid_rows),
        "augmentation": "one deterministic option-shuffle/relabel variant per labelled question",
        "system_prompt": SYSTEM,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
