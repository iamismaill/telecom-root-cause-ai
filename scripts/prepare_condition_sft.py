"""Prepare balanced condition-prompt SFT data for compliant model decisions."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.compliant_prompt import condition_model_prompt  # noqa: E402
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
    digest = hashlib.sha256(f"{salt}:{identifier}".encode()).hexdigest()
    return int(digest[:8], 16)


def record(question: str, semantic: str) -> dict:
    displayed = map_standard_cause(question, semantic)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": condition_model_prompt(question)},
            {"role": "assistant", "content": f"\\boxed{{{displayed}}}"},
        ]
    }


def interleave(frame, augment: bool) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        semantic = str(row.answer).upper()
        grouped[semantic].append(record(row.question, semantic))
        if augment:
            changed = shuffle_and_relabel_options(
                row.question, seed(str(row.ID), "condition-shuffle-1")
            )
            grouped[semantic].append(record(changed, semantic))
            if semantic in {"C3", "C7"}:
                focused = shuffle_and_relabel_options(
                    row.question, seed(str(row.ID), "condition-shuffle-focus")
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
    train_rows = interleave(train, augment=True)
    validation_rows = interleave(validation, augment=False)
    output = ROOT / "outputs/compliant_sft/qwen25_15b_conditions_v1"
    write(output / "train.jsonl", train_rows)
    write(output / "valid.jsonl", validation_rows)
    manifest = {
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "train_source": "train.csv only",
        "validation_used_for_training": False,
        "input": "condition evidence derived only from each current question",
        "augmentation": "shuffled/relabelled choices; extra C3/C7 exposure",
        "model_generated_final_answer": True,
        "retrieval": False,
        "cross_dataset_matching": False,
        "rule_answer_override": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
