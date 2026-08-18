"""Prepare compact reasoning SFT data for rule-compliant Qwen inference."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from collections import defaultdict
import hashlib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.compliant_prompt import (  # noqa: E402
    compact_model_prompt,
    supervised_rationale,
)
from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.evaluation import validation_truth  # noqa: E402
from telecom_rca.options import map_standard_cause  # noqa: E402
from telecom_rca.robustness import shuffle_and_relabel_options  # noqa: E402


SYSTEM = (
    "You are a careful 5G root-cause diagnostician. Use only the evidence in the "
    "current prompt. Your entire response must be exactly one final answer in the "
    "form \\boxed{CHOICE}. Never retrieve or match another question."
)


def stable_seed(identifier: str, salt: str) -> int:
    value = hashlib.sha256(f"{salt}:{identifier}".encode()).hexdigest()
    return int(value[:8], 16)


def _record(question: str, semantic: str):
    displayed = map_standard_cause(question, semantic)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": compact_model_prompt(question)},
            {
                "role": "assistant",
                "content": supervised_rationale(semantic, displayed),
            },
        ]
    }


def build(frame, augment: bool):
    grouped = defaultdict(list)
    for item in frame.itertuples(index=False):
        semantic = str(item.answer).upper()
        question = str(item.question)
        grouped[semantic].append(_record(question, semantic))
        if augment:
            transformed = shuffle_and_relabel_options(
                question, stable_seed(str(item.ID), "compliant-v3")
            )
            grouped[semantic].append(_record(transformed, semantic))
    # Interleave causes so every early checkpoint has balanced exposure.
    rows = []
    labels = sorted(grouped)
    for index in range(max(len(grouped[label]) for label in labels)):
        for label in labels:
            if index < len(grouped[label]):
                rows.append(grouped[label][index])
    return rows


def write(path: Path, rows) -> None:
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
    output = ROOT / "outputs/compliant_sft/qwen25_15b_v4"
    train_rows = build(train, augment=True)
    valid_rows = build(validation, augment=False)
    write(output / "train.jsonl", train_rows)
    write(output / "valid.jsonl", valid_rows)
    manifest = {
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "training_source": "current_challenge_data/train.csv",
        "validation_source": "official validation files",
        "examples": {"train": len(train_rows), "validation": len(valid_rows)},
        "input": "compact evidence derived independently from each current question",
        "augmentation": "one shuffled and relabelled option variant per training question",
        "ordering": "class-interleaved to balance early checkpoints",
        "model_generates_final_answer": True,
        "retrieval": False,
        "cross_dataset_matching": False,
        "rule_answer_override": False,
        "system_prompt": SYSTEM,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
