"""Build the final restart-safe, rules-compliant H200 submission notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/Cassava_H200_Unified_Compliant_Submission.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Cassava AI — unified compliant submission

This notebook performs inference only. It loads four frozen Qwen2.5-1.5B
adapters, uses only each current question, reuses the separately audited
Qwen2.5-3B general-knowledge generations, records every raw response, and
creates one 3,452-row submission. It performs no retrieval, cross-dataset
matching, deterministic diagnosis, or rule-based answer replacement."""
        ),
        code(
            r"""
%pip install -q "transformers==4.49.0" "peft==0.14.0" \
  "accelerate==1.3.0" "bitsandbytes>=0.49.2" \
  "safetensors>=0.5.2" "pandas>=2.2.0"
print("If packages changed, restart the kernel once, then continue from Cell 2.")
"""
        ),
        code(
            r"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import os
import re
import time

import pandas as pd
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = Path("/home/jovyan/data")
REPORT_DIR = Path("/home/jovyan/reports")
RUN_DIR = Path("/home/jovyan/outputs/unified_compliant_20260726_v1")

STANDARD_DIAGNOSTIC = Path("/home/jovyan/outputs/qwen25_15b_converted_from_mlx")
STANDARD_MAPPER = Path("/home/jovyan/outputs/qwen25_15b_mapper_v1/final_adapter/mapper")
NINE_DIAGNOSTIC = Path(
    "/home/jovyan/outputs/qwen25_15b_nine_cause_v3/final_adapter/nine_cause"
)
NINE_MAPPER = Path(
    "/home/jovyan/outputs/qwen25_15b_exact_nine_mapper_v2/"
    "final_adapter/exact_mapper_v2"
)
GENERAL_FROZEN = REPORT_DIR / "qwen25_3b_general_knowledge_frozen_82.csv"
GENERAL_RAW = REPORT_DIR / "qwen25_3b_general_knowledge_frozen_82.jsonl"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)

required_data = [
    DATA_DIR / "test.csv",
    DATA_DIR / "SampleSubmission.csv",
]
required_models = [
    STANDARD_DIAGNOSTIC / "adapter_model.safetensors",
    STANDARD_MAPPER / "adapter_model.safetensors",
    NINE_DIAGNOSTIC / "adapter_model.safetensors",
    NINE_MAPPER / "adapter_model.safetensors",
    GENERAL_FROZEN,
    GENERAL_RAW,
]
for path in required_data + required_models:
    assert path.is_file(), f"Missing required frozen input: {path}"

print("All frozen inputs verified.")
"""
        ),
        code(
            r"""
STANDARD_DESCRIPTIONS = {
    "C1": "The serving cell's downtilt angle is too large, causing weak coverage at the far end.",
    "C2": "The serving cell's coverage distance exceeds 1km, resulting in over-shooting.",
    "C3": "A neighboring cell provides higher throughput.",
    "C4": "Non-colocated co-frequency neighboring cells cause severe overlapping coverage.",
    "C5": "Frequent handovers degrade performance.",
    "C6": "Neighbor cell and serving cell have the same PCI mod 30, leading to interference.",
    "C7": "Test vehicle speed exceeds 40km/h, impacting user throughput.",
    "C8": "Average scheduled RBs are below 160, affecting throughput.",
}
NINE_DESCRIPTIONS = {
    "N1": "RF or power parameters cause severe overlap coverage",
    "N2": "Inter-frequency handover threshold configuration unreasonable",
    "N3": "Network capacity insufficient or load imbalance between cells",
    "N4": "Test server or transport anomaly causes insufficient upstream traffic",
    "N5": "Missing neighbor cell configuration",
    "N6": "RF, power parameters or site construction cause weak coverage",
    "N7": "Intra-frequency handover threshold too high",
    "N8": "Intra-frequency handover threshold too low",
    "N9": "PDCCH resource management parameters unreasonable",
}
STANDARD_BY_DESCRIPTION = {
    value.casefold(): key for key, value in STANDARD_DESCRIPTIONS.items()
}
NINE_BY_DESCRIPTION = {
    value.casefold(): key for key, value in NINE_DESCRIPTIONS.items()
}

OPTION_RE = re.compile(
    r"^\s*([A-Z](?:[1-9])?|[1-9])\s*[:.)]\s*(.+?)\s*$",
    re.IGNORECASE,
)
BOX_RE = re.compile(r"\\boxed\{\s*([A-Za-z0-9]+)\s*\}", re.IGNORECASE)

def parsed_lines(question):
    rows = []
    for line_index, line in enumerate(str(question).splitlines()):
        if "|" in line:
            continue
        match = OPTION_RE.match(line)
        if match:
            rows.append({
                "line_index": line_index,
                "label": match.group(1).upper(),
                "description": match.group(2).strip(),
            })
    return rows

def recognized_options(question, description_map):
    return [
        {
            **row,
            "semantic": description_map[row["description"].casefold()],
        }
        for row in parsed_lines(question)
        if row["description"].casefold() in description_map
    ]

def route_question(question):
    standard = recognized_options(question, STANDARD_BY_DESCRIPTION)
    nine = recognized_options(question, NINE_BY_DESCRIPTION)
    if len(nine) == 9 and {x["semantic"] for x in nine} == set(NINE_DESCRIPTIONS):
        return "nine_cause", nine
    if 5 <= len(standard) <= 8 and len({x["semantic"] for x in standard}) == len(standard):
        return "standard", standard
    if "drive test" not in str(question).casefold() and "engineering" not in str(question).casefold():
        final_four = parsed_lines(question)[-4:]
        assert tuple(x["label"] for x in final_four) == ("1", "2", "3", "4")
        return "general", final_four
    raise ValueError("Question could not be routed safely")

test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_submission = pd.read_csv(DATA_DIR / "SampleSubmission.csv")
assert set(test_df.columns) >= {"ID", "question"}
assert list(sample_submission.columns) == ["ID", "Target"]
assert len(test_df) == 863
assert len(sample_submission) == 3452
assert test_df["ID"].is_unique
sample_submission["QuestionID"] = sample_submission["ID"].str.replace(
    r"_[1-4]$", "", regex=True
)

route_records = []
for row in test_df.itertuples(index=False):
    route, options = route_question(row.question)
    route_records.append({
        "ID": row.ID,
        "question": row.question,
        "route": route,
        "options": options,
    })
route_df = pd.DataFrame(route_records)
route_counts = route_df["route"].value_counts().to_dict()
assert route_counts == {"standard": 681, "nine_cause": 100, "general": 82}
assert sample_submission.groupby("QuestionID").size().eq(4).all()
assert set(sample_submission["QuestionID"]) == set(test_df["ID"])

print("Route counts:", route_counts)
print("Submission structure verified:", len(sample_submission))
"""
        ),
        code(
            r"""
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=False)
config.use_sliding_window = False
config.sliding_window = None
config.use_cache = True

quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    quantization_config=quantization,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
    trust_remote_code=False,
)
model = PeftModel.from_pretrained(
    base,
    str(STANDARD_DIAGNOSTIC),
    adapter_name="standard_diagnostic",
    is_trainable=False,
)
model.load_adapter(
    str(STANDARD_MAPPER), adapter_name="standard_mapper", is_trainable=False
)
model.load_adapter(
    str(NINE_DIAGNOSTIC), adapter_name="nine_diagnostic", is_trainable=False
)
model.load_adapter(
    str(NINE_MAPPER), adapter_name="nine_mapper", is_trainable=False
)
model.eval()
model.config.use_cache = True
model.generation_config.do_sample = False
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.top_k = None

print("Loaded frozen adapters:", list(model.peft_config))
print("GPU memory GiB:", round(torch.cuda.memory_allocated() / 1024**3, 2))
"""
        ),
        code(
            r"""
STANDARD_SYSTEM = (
    "Use only the current question to identify the semantic 5G root cause. "
    "Select one of the canonical causes offered in this prompt. Your entire "
    "response must be exactly one label in the form \\boxed{C1} through \\boxed{C8}."
)
NINE_SYSTEM = (
    "You are a 5G root-cause diagnostician. Use only the current question's "
    "drive-test and engineering evidence. Select one canonical nine-cause "
    "diagnosis offered in this prompt. Your entire response must be exactly "
    "one of \\boxed{N1} through \\boxed{N9}."
)
STANDARD_MAPPER_SYSTEM = (
    "Map the supplied model-generated semantic diagnosis to the offered choice "
    "whose description has the same meaning. Use only this prompt. Your entire "
    "response must be exactly one displayed label in the form \\boxed{LABEL}."
)
NINE_MAPPER_SYSTEM = (
    "Map the supplied model-generated semantic diagnosis to the offered choice "
    "whose description exactly matches the canonical meaning. Use only this "
    "prompt. Your entire response must be exactly one displayed label in the "
    "form \\boxed{LABEL}."
)

def boxed(text):
    match = BOX_RE.search(str(text).strip())
    return match.group(1).upper() if match else None

def canonical_question(question, options, descriptions):
    remove_indexes = {item["line_index"] for item in options}
    kept = [
        line for index, line in enumerate(str(question).splitlines())
        if index not in remove_indexes
    ]
    canonical = "\n".join(
        f"{item['semantic']}: {descriptions[item['semantic']]}"
        for item in options
    )
    return (
        "\n".join(kept).rstrip()
        + "\n\nCanonical offered causes:\n"
        + canonical
        + "\n\nGenerate only the canonical semantic label."
    )

def mapper_messages(semantic, options, descriptions, system):
    offered = "\n".join(
        f"- {item['label']}: {item['description']}" for item in options
    )
    content = (
        f"Model-generated semantic diagnosis: {semantic}\n"
        f"Canonical meaning: {descriptions[semantic]}\n\n"
        f"Offered choices from the current question:\n{offered}\n\n"
        "Generate the displayed label whose description matches the diagnosis."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]

@torch.inference_mode()
def generate(adapter, messages, max_length=4096):
    model.set_adapter(adapter)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    output = model.generate(
        **encoded,
        max_new_tokens=12,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    raw = tokenizer.decode(
        output[0, encoded["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return raw, boxed(raw)

def load_journal(path):
    if not path.is_file():
        return {}
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                rows[item["ID"]] = item
    return rows

def append_journal(path, item):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()

print("Inference helpers ready.")
"""
        ),
        code(
            r"""
STANDARD_JOURNAL = RUN_DIR / "standard_predictions.jsonl"
standard_done = load_journal(STANDARD_JOURNAL)
standard_rows = route_df[route_df["route"] == "standard"]
started = time.perf_counter()

for index, row in enumerate(standard_rows.itertuples(index=False), 1):
    if row.ID in standard_done:
        continue
    semantic_user = canonical_question(
        row.question, row.options, STANDARD_DESCRIPTIONS
    )
    semantic_messages = [
        {"role": "system", "content": STANDARD_SYSTEM},
        {"role": "user", "content": semantic_user},
    ]
    semantic_raw, semantic = generate(
        "standard_diagnostic", semantic_messages
    )
    offered_semantics = {item["semantic"] for item in row.options}
    if semantic not in offered_semantics:
        raise RuntimeError(
            f"Standard semantic output invalid for {row.ID}: {semantic_raw!r}"
        )
    map_messages = mapper_messages(
        semantic, row.options, STANDARD_DESCRIPTIONS, STANDARD_MAPPER_SYSTEM
    )
    mapper_raw, answer = generate("standard_mapper", map_messages, max_length=1536)
    offered_labels = {item["label"] for item in row.options}
    if answer not in offered_labels:
        raise RuntimeError(
            f"Standard mapper output invalid for {row.ID}: {mapper_raw!r}"
        )
    item = {
        "ID": row.ID,
        "route": "standard",
        "semantic": semantic,
        "answer": answer,
        "semantic_raw": semantic_raw,
        "mapper_raw": mapper_raw,
        "question_sha256": hashlib.sha256(row.question.encode()).hexdigest(),
        "model_generated_semantic": True,
        "model_generated_final_answer": True,
    }
    append_journal(STANDARD_JOURNAL, item)
    standard_done[row.ID] = item
    if len(standard_done) % 25 == 0 or len(standard_done) == len(standard_rows):
        print(
            f"Standard {len(standard_done)}/{len(standard_rows)} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

assert len(standard_done) == 681
print("Standard route complete:", len(standard_done))
"""
        ),
        code(
            r"""
NINE_JOURNAL = RUN_DIR / "nine_cause_predictions.jsonl"
nine_done = load_journal(NINE_JOURNAL)
nine_rows = route_df[route_df["route"] == "nine_cause"]
started = time.perf_counter()

for index, row in enumerate(nine_rows.itertuples(index=False), 1):
    if row.ID in nine_done:
        continue
    semantic_user = canonical_question(
        row.question, row.options, NINE_DESCRIPTIONS
    )
    semantic_messages = [
        {"role": "system", "content": NINE_SYSTEM},
        {"role": "user", "content": semantic_user},
    ]
    semantic_raw, semantic = generate("nine_diagnostic", semantic_messages)
    if semantic not in set(NINE_DESCRIPTIONS):
        raise RuntimeError(
            f"Nine-cause semantic output invalid for {row.ID}: {semantic_raw!r}"
        )
    map_messages = mapper_messages(
        semantic, row.options, NINE_DESCRIPTIONS, NINE_MAPPER_SYSTEM
    )
    mapper_raw, answer = generate("nine_mapper", map_messages, max_length=1536)
    offered_labels = {item["label"] for item in row.options}
    if answer not in offered_labels:
        raise RuntimeError(
            f"Nine-cause mapper output invalid for {row.ID}: {mapper_raw!r}"
        )
    item = {
        "ID": row.ID,
        "route": "nine_cause",
        "semantic": semantic,
        "answer": answer,
        "semantic_raw": semantic_raw,
        "mapper_raw": mapper_raw,
        "question_sha256": hashlib.sha256(row.question.encode()).hexdigest(),
        "model_generated_semantic": True,
        "model_generated_final_answer": True,
    }
    append_journal(NINE_JOURNAL, item)
    nine_done[row.ID] = item
    if len(nine_done) % 20 == 0 or len(nine_done) == len(nine_rows):
        print(
            f"Nine-cause {len(nine_done)}/{len(nine_rows)} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

assert len(nine_done) == 100
print("Nine-cause route complete:", len(nine_done))
"""
        ),
        code(
            r"""
general_df = pd.read_csv(GENERAL_FROZEN, dtype={"answer": str})
assert set(general_df.columns) >= {"ID", "answer", "source"}
assert len(general_df) == 82
assert general_df["ID"].is_unique
general_route_ids = set(route_df.loc[route_df["route"] == "general", "ID"])
assert set(general_df["ID"]) == general_route_ids
assert general_df["answer"].isin(["1", "2", "3", "4"]).all()

prediction_records = []
prediction_records.extend(standard_done.values())
prediction_records.extend(nine_done.values())
prediction_records.extend({
    "ID": row.ID,
    "route": "general",
    "semantic": None,
    "answer": str(row.answer),
    "semantic_raw": None,
    "mapper_raw": None,
    "general_source": row.source,
    "model_generated_final_answer": True,
} for row in general_df.itertuples(index=False))

predictions = pd.DataFrame(prediction_records)
assert len(predictions) == 863
assert predictions["ID"].is_unique
assert set(predictions["ID"]) == set(test_df["ID"])

route_lookup = route_df.set_index("ID")
for row in predictions.itertuples(index=False):
    route_row = route_lookup.loc[row.ID]
    offered = {item["label"] for item in route_row["options"]}
    assert str(row.answer) in offered, (
        f"Answer is not offered for {row.ID}: {row.answer}, offered={offered}"
    )

answer_map = predictions.set_index("ID")["answer"].astype(str).to_dict()
submission = sample_submission.copy()
submission["Target"] = submission["QuestionID"].map(
    lambda identifier: f"\\boxed{{{answer_map[identifier]}}}"
)
assert submission["Target"].notna().all()
assert len(submission) == 3452
assert submission.groupby("QuestionID").size().eq(4).all()
assert submission.groupby("QuestionID")["Target"].nunique().eq(1).all()
assert submission["Target"].str.fullmatch(
    r"\\boxed\{[A-Za-z0-9]+\}"
).all()
submission = submission[["ID", "Target"]]

submission_path = RUN_DIR / "Cassava_Compliant_Unified_Submission.csv"
audit_path = RUN_DIR / "Cassava_Compliant_Unified_Predictions.jsonl"
summary_path = RUN_DIR / "Cassava_Compliant_Unified_Summary.json"

submission.to_csv(submission_path, index=False)
predictions.to_json(
    audit_path, orient="records", lines=True, force_ascii=False
)

summary = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "base_models": [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    ],
    "all_models_under_4B": True,
    "questions": 863,
    "submission_rows": 3452,
    "route_counts": predictions["route"].value_counts().to_dict(),
    "all_answers_offered": True,
    "all_final_answers_model_generated": True,
    "retrieval": False,
    "cross_dataset_matching": False,
    "rule_answer_overrides": False,
    "submission_file": str(submission_path),
}
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print("\nSubmission:", submission_path)
print("Audit:", audit_path)
"""
        ),
        code(
            r"""
def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

hash_targets = {
    "test_csv": DATA_DIR / "test.csv",
    "sample_submission": DATA_DIR / "SampleSubmission.csv",
    "standard_diagnostic": STANDARD_DIAGNOSTIC / "adapter_model.safetensors",
    "standard_mapper": STANDARD_MAPPER / "adapter_model.safetensors",
    "nine_diagnostic": NINE_DIAGNOSTIC / "adapter_model.safetensors",
    "nine_mapper": NINE_MAPPER / "adapter_model.safetensors",
    "general_frozen_csv": GENERAL_FROZEN,
    "general_raw_jsonl": GENERAL_RAW,
    "final_submission": RUN_DIR / "Cassava_Compliant_Unified_Submission.csv",
    "final_audit": RUN_DIR / "Cassava_Compliant_Unified_Predictions.jsonl",
}
hashes = {name: file_sha256(path) for name, path in hash_targets.items()}
(RUN_DIR / "sha256_manifest.json").write_text(
    json.dumps(hashes, indent=2), encoding="utf-8"
)

print("FINAL VALIDATION PASSED")
print("Questions:", len(predictions))
print("Rows:", len(submission))
print("Routes:")
print(predictions["route"].value_counts())
print("Answer distribution by route:")
print(pd.crosstab(predictions["route"], predictions["answer"]))
print("Submission SHA256:", hashes["final_submission"])
"""
        ),
    ]
    nbf.write(nb, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
