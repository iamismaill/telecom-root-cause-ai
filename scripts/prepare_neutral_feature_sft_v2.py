"""Prepare balanced, label-neutral feature prompts from official challenge data."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.evaluation import validation_truth  # noqa: E402
from telecom_rca.features import extract_diagnostic_features  # noqa: E402
from telecom_rca.options import parse_options, semantic_cause  # noqa: E402
from telecom_rca.parser import parse_question  # noqa: E402
from telecom_rca.routing import Route, route_question  # noqa: E402


SYSTEM = (
    "You are an open-source 5G diagnostic language model. Use only the current "
    "question's offered choices and numeric evidence summary. Infer the most "
    "likely cause yourself. Generate exactly one offered label and nothing else. "
    "Never retrieve or match another question."
)

# Descriptive values only. No cause flags, TRUE/FALSE decisions, ordered rules,
# or precomputed label is placed in the user prompt.
FEATURES = (
    ("row_count", "drive-test row count"),
    ("throughput_threshold", "stated throughput threshold Mbps"),
    ("low_row_count", "rows below the stated threshold"),
    ("low_fraction", "fraction of rows below the stated threshold"),
    ("throughput_min", "throughput minimum Mbps"),
    ("throughput_mean", "throughput mean Mbps"),
    ("low_throughput_mean", "degraded-row throughput mean Mbps"),
    ("speed_max", "speed maximum km/h"),
    ("low_speed_max", "degraded-row speed maximum km/h"),
    ("low_speed_mean", "degraded-row speed mean km/h"),
    ("rbs_mean", "scheduled RB mean"),
    ("low_rbs_mean", "degraded-row scheduled RB mean"),
    ("serving_rsrp_mean", "serving RSRP mean dBm"),
    ("low_serving_rsrp_mean", "degraded-row serving RSRP mean dBm"),
    ("low_serving_rsrp_min", "degraded-row serving RSRP minimum dBm"),
    ("serving_sinr_mean", "serving SINR mean dB"),
    ("low_serving_sinr_mean", "degraded-row serving SINR mean dB"),
    ("serving_cell_count", "distinct serving-cell count"),
    ("handover_count", "serving-cell transition count"),
    ("handover_rate", "serving-cell transition rate"),
    ("ping_pong_count", "A-B-A serving-cell sequence count"),
    ("low_handover_count", "degraded-row serving-cell transition count"),
    ("distance_max_km", "serving distance maximum km"),
    ("low_distance_max_km", "degraded-row serving distance maximum km"),
    ("low_distance_mean_km", "degraded-row serving distance mean km"),
    ("serving_total_tilt_max", "serving total tilt maximum degrees"),
    ("low_serving_total_tilt_max", "degraded-row total tilt maximum degrees"),
    ("low_serving_total_tilt_mean", "degraded-row total tilt mean degrees"),
    ("low_tilt_excess_max_deg", "degraded-row beam-edge tilt excess maximum degrees"),
    ("low_tilt_excess_mean_deg", "degraded-row beam-edge tilt excess mean degrees"),
    ("low_mechanical_tilt_max", "degraded-row mechanical tilt maximum degrees"),
    ("low_antenna_height_mean", "degraded-row antenna height mean metres"),
    ("neighbor_margin_max_db", "best-neighbor RSRP margin maximum dB"),
    ("low_neighbor_margin_max_db", "degraded-row best-neighbor RSRP margin maximum dB"),
    ("low_neighbor_stronger_fraction", "degraded rows with a stronger measured neighbor fraction"),
    ("low_neighbor_throughput_gain_max", "degraded-row neighbor throughput gain maximum Mbps"),
    (
        "low_neighbor_throughput_gain_positive_fraction",
        "degraded rows with positive neighbor throughput gain fraction",
    ),
    (
        "low_mod30_conflict_fraction",
        "degraded serving-neighbor PCI remainder-equality fraction modulo 30",
    ),
    (
        "low_strong_noncolocated_fraction",
        "degraded rows with strong non-colocated neighbor fraction",
    ),
    (
        "low_close_noncolocated_fraction",
        "degraded rows with close non-colocated neighbor fraction",
    ),
    ("engineering_match_fraction", "rows matched to engineering records fraction"),
)

LABEL_FAMILIES = (
    tuple(f"C{i}" for i in range(1, 9)),
    tuple(str(i) for i in range(1, 9)),
    tuple(f"M{i}" for i in range(1, 9)),
    tuple("ABCDEFGH"),
)


def display(value: float) -> str:
    return "unavailable" if math.isnan(value) else f"{value:.6g}"


def stable_seed(identifier: str, salt: str) -> int:
    return int(
        hashlib.sha256(f"{salt}:{identifier}".encode()).hexdigest()[:8],
        16,
    )


def option_meanings(question: str) -> list[tuple[str, str, str]]:
    rows = []
    for option in parse_options(question):
        semantic = semantic_cause(option.description)
        if semantic:
            rows.append((semantic, option.label, option.description))
    if not 2 <= len(rows) <= 8 or len({row[0] for row in rows}) != len(rows):
        raise ValueError("Expected two to eight unique standard cause descriptions")
    return rows


def relabel_options(
    rows: list[tuple[str, str, str]],
    identifier: str,
) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    rng = random.Random(stable_seed(identifier, "neutral-v2-labels"))
    family = list(LABEL_FAMILIES[stable_seed(identifier, "family") % len(LABEL_FAMILIES)])
    rng.shuffle(family)
    mapping = {
        semantic: family[index]
        for index, (semantic, _, _) in enumerate(sorted(rows))
    }
    changed = [
        (semantic, mapping[semantic], description)
        for semantic, _, description in rows
    ]
    rng.shuffle(changed)
    return changed, mapping


def neutral_prompt(
    question: str,
    identifier: str,
    relabel: bool,
) -> tuple[str, dict[str, str]]:
    parsed = parse_question(question)
    values = extract_diagnostic_features(question, parsed)
    options = option_meanings(question)

    if relabel:
        options, mapping = relabel_options(options, identifier)
    else:
        mapping = {semantic: label for semantic, label, _ in options}

    option_text = "\n".join(
        f"- {label}: {description}"
        for _, label, description in options
    )
    evidence_text = "\n".join(
        f"- {description}: {display(float(values[name]))}"
        for name, description in FEATURES
    )
    prompt = (
        "Diagnose this current 5G throughput degradation case.\n\n"
        f"Offered choices:\n{option_text}\n\n"
        f"Label-neutral numeric evidence summary:\n{evidence_text}\n\n"
        "Compare the evidence with every offered cause. Generate exactly the "
        "single offered label you select, with no explanation."
    )
    return prompt, mapping


def record(row, relabel: bool) -> dict:
    semantic = str(row.answer).upper()
    prompt, mapping = neutral_prompt(
        str(row.question),
        f"{row.ID}-{'shift' if relabel else 'original'}",
        relabel,
    )
    return {
        "ID": str(row.ID),
        "semantic": semantic,
        "displayed_answer": mapping[semantic],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": mapping[semantic]},
        ],
    }


def balanced_training(frame) -> list[dict]:
    grouped: dict[str, list] = defaultdict(list)
    for row in frame.itertuples(index=False):
        grouped[str(row.answer).upper()].append(row)

    target = max(len(rows) for rows in grouped.values())
    output = []
    for semantic in sorted(grouped):
        rows = grouped[semantic]
        for index in range(target):
            row = rows[index % len(rows)]
            output.append(record(row, relabel=(index % 2 == 1)))

    random.Random(20260727).shuffle(output)
    return output


def validation_rows(frame) -> list[dict]:
    return [record(row, relabel=False) for row in frame.itertuples(index=False)]


def standard_test_rows(frame) -> list[dict]:
    """Build answer-free prompts from each standard test question independently."""
    output = []
    for row in frame.itertuples(index=False):
        decision = route_question(str(row.question))
        if decision.route is not Route.STANDARD_TELECOM:
            continue
        prompt, _ = neutral_prompt(
            str(row.question),
            f"{row.ID}-test",
            relabel=False,
        )
        labels = [option.label.upper() for option in decision.options]
        output.append(
            {
                "ID": str(row.ID),
                "route": "standard",
                "offered_labels": labels,
                "question_sha256": hashlib.sha256(
                    str(row.question).encode()
                ).hexdigest(),
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            }
        )
    return output


def write_jsonl(path: Path, rows: list[dict]) -> None:
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
    test = load_current_csv("test.csv")[["ID", "question"]]

    train_rows = balanced_training(train)
    valid_rows = validation_rows(validation)
    test_rows = standard_test_rows(test)
    output = ROOT / "outputs/neutral_feature_sft_v2"
    write_jsonl(output / "train.jsonl", train_rows)
    write_jsonl(output / "validation.jsonl", valid_rows)
    write_jsonl(output / "test_standard.jsonl", test_rows)

    manifest = {
        "train_rows": len(train_rows),
        "validation_rows": len(valid_rows),
        "standard_test_rows": len(test_rows),
        "train_class_counts": {
            label: sum(row["semantic"] == label for row in train_rows)
            for label in sorted({row["semantic"] for row in train_rows})
        },
        "prompt_contains_true_false_flags": False,
        "prompt_contains_ordered_answer_rules": False,
        "prompt_contains_precomputed_label": False,
        "features_are_descriptive_current_question_statistics": True,
        "retrieval": False,
        "cross_dataset_matching": False,
        "external_data": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
