"""Build the balanced neutral-feature Qwen3B V2 notebook."""

from pathlib import Path
import hashlib
import json
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/Cassava_H200_RawQuestion_Qwen3B_V1.ipynb"
OUTPUT = ROOT / "notebooks/Cassava_H200_NeutralFeature_Qwen3B_V2.ipynb"


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).lstrip().splitlines(keepends=True),
    }


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]

cells[0]["source"] = [
    "# Cassava H200 — balanced neutral-feature Qwen2.5-3B V2\n\n",
    "Uses only descriptive statistics calculated from each current official ",
    "question. Prompts contain no TRUE/FALSE cause flags, ordered answer rules, ",
    "retrieved examples, cross-dataset matches, or precomputed labels. Qwen ",
    "generates the offered answer label.\n",
]

config = "".join(cells[2]["source"]).replace(
    'OUTPUT_DIR = Path("outputs/qwen25_3b_raw_v1")',
    'OUTPUT_DIR = Path("outputs/qwen25_3b_neutral_v2")',
).replace(
    "MAX_LENGTH = 3072",
    "MAX_LENGTH = 1536",
)
cells[2]["source"] = config.splitlines(keepends=True)

cells[3] = code(
    r"""
    FEATURE_DATA_DIR = Path("neutral_feature_sft_v2")
    TRAIN_JSONL = FEATURE_DATA_DIR / "train.jsonl"
    VALID_JSONL = FEATURE_DATA_DIR / "validation.jsonl"
    FEATURE_MANIFEST = FEATURE_DATA_DIR / "manifest.json"

    for path in [TRAIN_JSONL, VALID_JSONL, FEATURE_MANIFEST]:
        assert path.is_file(), f"Upload missing V2 input: {path}"

    def read_jsonl(path):
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    train_rows = read_jsonl(TRAIN_JSONL)
    validation_rows = read_jsonl(VALID_JSONL)
    feature_manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))

    assert len(train_rows) == 2816
    assert len(validation_rows) == 864
    assert feature_manifest["prompt_contains_true_false_flags"] is False
    assert feature_manifest["prompt_contains_ordered_answer_rules"] is False
    assert feature_manifest["prompt_contains_precomputed_label"] is False
    assert set(feature_manifest["train_class_counts"].values()) == {352}

    validation_df = pd.DataFrame({
        "ID": [row["ID"] for row in validation_rows],
        "question": [row["messages"][1]["content"] for row in validation_rows],
        "answer": [row["displayed_answer"] for row in validation_rows],
        "semantic": [row["semantic"] for row in validation_rows],
    })
    assert validation_df["ID"].is_unique
    print("Balanced V2 train rows:", len(train_rows))
    print("Untouched V2 validation rows:", len(validation_rows))
    print("Training semantic distribution:")
    print(pd.Series([row["semantic"] for row in train_rows]).value_counts().sort_index())
    print("Feature manifest:")
    print(json.dumps(feature_manifest, indent=2))
    """
)

cells[4] = code(
    r"""
    train_records = [
        {"messages": row["messages"]}
        for row in train_rows
    ]
    validation_records = [
        {"messages": row["messages"]}
        for row in validation_rows
    ]
    assert len(train_records) == 2816
    assert len(validation_records) == 864
    print("V2 conversations ready:", len(train_records), len(validation_records))
    print("Assistant answer examples:", sorted({
        row["messages"][-1]["content"] for row in train_records
    })[:30])
    """
)

training = "".join(cells[6]["source"])
training = training.replace(
    "num_train_epochs=1.0",
    "num_train_epochs=2.0",
).replace(
    "learning_rate=5e-5",
    "learning_rate=1e-4",
).replace(
    "eval_steps=100",
    "eval_steps=200",
).replace(
    "save_steps=100",
    "save_steps=200",
).replace(
    'print("Starting clean raw-question QLoRA training...")',
    'print("Starting balanced neutral-feature QLoRA V2 training...")',
)
cells[6]["source"] = training.splitlines(keepends=True)

reload_source = "".join(cells[7]["source"]).replace(
    "qwen25_3b_raw_v1",
    "qwen25_3b_neutral_v2",
)
cells[7]["source"] = reload_source.splitlines(keepends=True)

cells[8] = code(
    r"""
    import re

    model.eval()
    model.config.use_cache = True
    BOXED_RE = re.compile(r"\\boxed\{\s*([A-Za-z0-9]+)\s*\}", re.I)
    OFFER_RE = re.compile(r"^\s*-\s*([A-Za-z][0-9]?|[1-9])\s*:\s*", re.I)

    def offered_labels(prompt):
        labels = {
            match.group(1).upper()
            for line in str(prompt).splitlines()
            if (match := OFFER_RE.match(line))
        }
        assert len(labels) == 8, labels
        return labels

    @torch.inference_mode()
    def generate_v2(prompt, max_new_tokens=8):
        messages = [
            {"role": "system", "content": train_rows[0]["messages"][0]["content"]},
            {"role": "user", "content": prompt},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        raw = tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        match = BOXED_RE.search(raw)
        candidate = match.group(1).upper() if match else raw.upper()
        allowed = offered_labels(prompt)
        answer = candidate if candidate in allowed else ""
        return raw, answer, hashlib.sha256(rendered.encode()).hexdigest()

    balanced_parts = []
    for semantic in [f"C{i}" for i in range(1, 9)]:
        part = validation_df[validation_df["semantic"] == semantic].sample(
            n=10, random_state=20260727
        )
        balanced_parts.append(part)
    gate_df = pd.concat(balanced_parts, ignore_index=True)
    assert len(gate_df) == 80

    gate_outputs = []
    gate_started = time.perf_counter()
    for index, row in enumerate(gate_df.itertuples(index=False), 1):
        raw, answer, prompt_hash = generate_v2(row.question)
        gate_outputs.append({
            "ID": row.ID,
            "semantic": row.semantic,
            "truth": row.answer,
            "answer": answer,
            "raw_generation": raw,
            "prompt_sha256": prompt_hash,
            "valid": bool(answer),
            "correct": answer == row.answer,
        })
        if index % 10 == 0:
            current = pd.DataFrame(gate_outputs)
            print(
                f"Gate {index}/80 accuracy={current.correct.mean():.4f} "
                f"format={current.valid.mean():.4f}",
                flush=True,
            )

    gate_result = pd.DataFrame(gate_outputs)
    gate_by_class = gate_result.groupby("semantic").agg(
        questions=("ID", "size"),
        correct=("correct", "sum"),
        accuracy=("correct", "mean"),
    )
    gate_summary = {
        "experiment": "neutral_feature_qwen3b_v2_reloaded_gate",
        "questions": 80,
        "correct": int(gate_result.correct.sum()),
        "accuracy": float(gate_result.correct.mean()),
        "format_success": float(gate_result.valid.mean()),
        "adapter_sha256": adapter_sha256,
        "model_reloaded_before_validation": True,
        "elapsed_seconds": time.perf_counter() - gate_started,
        "model_generated_all_answers": True,
        "true_false_flags": False,
        "ordered_answer_rules": False,
        "retrieval": False,
        "cross_dataset_matching": False,
        "rule_answer_overrides": False,
    }
    print("\nV2 RELOADED BALANCED GATE")
    print(json.dumps(gate_summary, indent=2))
    print(gate_by_class)
    print("\nAnswer distribution:")
    print(gate_result.answer.replace("", "INVALID").value_counts())
    (REPORT_DIR / "qwen25_3b_neutral_v2_gate_80.json").write_text(
        json.dumps(
            {
                **gate_summary,
                "by_class": gate_by_class.to_dict(orient="index"),
                "records": gate_outputs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert gate_summary["accuracy"] >= 0.60, (
        "V2 EARLY GATE FAILED. Stop here; do not run full validation or test inference."
    )
    assert gate_summary["format_success"] == 1.0
    print("\nPASS: V2 may proceed to full validation.")
    """
)

cells[9] = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Stop and report the 80-question gate\n\n",
        "Do not run full validation or test inference until Codex reviews the ",
        "saved/reloaded balanced gate.\n",
    ],
}
cells[10] = code(
    r"""
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "neutral_feature_qwen3b_v2",
        "base_model": MODEL_ID,
        "base_model_parameters_under_4b": True,
        "adapter_sha256": adapter_sha256,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "balanced_classes": feature_manifest["train_class_counts"],
        "model_reloaded_before_validation": True,
        "features_from_current_question_only": True,
        "true_false_flags": False,
        "ordered_answer_rules": False,
        "precomputed_label_in_prompt": False,
        "retrieval": False,
        "cross_dataset_matching": False,
        "external_data": False,
        "rule_answer_overrides": False,
        "all_final_answers_generated_by_model": True,
    }
    (REPORT_DIR / "qwen25_3b_neutral_v2_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    """
)

notebook["metadata"]["cassava"] = {
    "experiment": "neutral_feature_qwen3b_v2",
    "base_model": "Qwen/Qwen2.5-3B-Instruct",
    "parameters_under_4b": True,
    "early_gate": 80,
    "test_inference_included": False,
}

OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
print("sha256", hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
