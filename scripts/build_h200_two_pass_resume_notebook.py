"""Build a restart-safe notebook for the frozen diagnostic and mapper adapters."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/Cassava_H200_TwoPass_Resume_Robustness.ipynb"


def cell(source: str):
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
            """# Cassava H200 — restart-safe two-pass robustness

Loads the frozen diagnostic and trained mapper adapters from disk. It does not
train or overwrite either adapter. It evaluates numeric, alphabetic, and unseen
prefixed displayed labels using model-generated answers only."""
        ),
        cell(
            r"""
%pip install -q "transformers==4.49.0" "peft==0.14.0" \
  "accelerate==1.3.0" "bitsandbytes>=0.49.2" \
  "safetensors>=0.5.2" "pandas>=2.2.0"
print("Restart the kernel once if packages changed, then continue.")
"""
        ),
        cell(
            r"""
from pathlib import Path
import copy, json, random, re, time
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PACKAGE = Path("/home/jovyan/frozen_local_15b_upload")
VALID_PATH = PACKAGE / "sft_data/valid.jsonl"
DIAGNOSTIC = Path("/home/jovyan/outputs/qwen25_15b_converted_from_mlx")
MAPPER = Path("/home/jovyan/outputs/qwen25_15b_mapper_v1/final_adapter/mapper")
REPORT_DIR = Path("/home/jovyan/reports")
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42

for path in [
    VALID_PATH,
    DIAGNOSTIC / "adapter_model.safetensors",
    MAPPER / "adapter_model.safetensors",
]:
    assert path.is_file(), f"Missing restored file: {path}"

with VALID_PATH.open(encoding="utf-8") as handle:
    valid_records = [json.loads(line) for line in handle if line.strip()]
assert len(valid_records) == 864
REPORT_DIR.mkdir(parents=True, exist_ok=True)
print("Restored inputs verified:", len(valid_records))
"""
        ),
        cell(
            r"""
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(
    base, str(DIAGNOSTIC), adapter_name="diagnostic", is_trainable=False
)
model.load_adapter(str(MAPPER), adapter_name="mapper", is_trainable=False)
model.eval()
model.config.use_cache = True
model.generation_config.do_sample = False
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.top_k = None
print("Loaded adapters:", list(model.peft_config))
"""
        ),
        cell(
            r"""
CANONICAL = {
    "C1": "Excessive antenna downtilt causing weak coverage at the far cell edge.",
    "C2": "Overshooting coverage or serving distance greater than 1 kilometre.",
    "C3": "A neighboring cell offers better radio or throughput performance.",
    "C4": "Severe overlapping coverage from close non-colocated co-frequency cells.",
    "C5": "Frequent handovers or ping-pong mobility.",
    "C6": "PCI modulo-30 interference.",
    "C7": "High vehicle speed, typically above 40 kilometres per hour.",
    "C8": "Insufficient scheduled resource blocks, typically below 160 RBs.",
}
MAPPER_SYSTEM = (
    "Map the supplied semantic diagnosis to the offered choice whose description "
    "has the same meaning. Use only this prompt. Your entire response must be "
    "exactly one displayed label in the form \\boxed{LABEL}."
)
SEMANTIC_SYSTEM = (
    "Use only the current prompt to identify the semantic 5G root cause. "
    "Respond exactly with one of \\boxed{C1} through \\boxed{C8}."
)
BLOCK_RE = re.compile(r"(Offered choices:\n)(.*?)(\n\nCondition results:)", re.S)
LINE_RE = re.compile(r"^-\s*(C[1-8])\s*:\s*(.+)$")
BOX_RE = re.compile(r"\\+boxed\{\s*([A-Za-z0-9]+)\s*\}", re.I)
CANONICAL_LINES = "\n".join(f"- {k}: {v}" for k, v in CANONICAL.items())

def boxed(text):
    match = BOX_RE.search(str(text).strip())
    return match.group(1).upper() if match else ""

def generate(adapter, messages):
    model.set_adapter(adapter)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **encoded, max_new_tokens=8, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(
        output[0, encoded["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return raw, boxed(raw)

rows = []
for record in valid_records:
    rows.append({
        "messages": record["messages"][:-1],
        "truth": boxed(record["messages"][-1]["content"]),
    })
frame = pd.DataFrame(rows)
balanced = (
    frame.groupby("truth", group_keys=False)
    .sample(n=10, random_state=SEED)
    .reset_index(drop=True)
)
assert len(balanced) == 80

def original_choices(messages):
    text = messages[-1]["content"]
    match = BLOCK_RE.search(text)
    assert match
    options = []
    for line in match.group(2).splitlines():
        item = LINE_RE.match(line.strip())
        if item:
            options.append((item.group(1), item.group(2)))
    assert len(options) == 8
    return match, options

def semantic_messages(messages):
    changed = copy.deepcopy(messages)
    text = changed[-1]["content"]
    match, _ = original_choices(changed)
    replacement = (
        "Canonical semantic causes:\n" + CANONICAL_LINES + match.group(3)
    )
    changed[-1]["content"] = (
        text[:match.start()] + replacement + text[match.end():]
        + "\n\nIgnore displayed labels. Generate only the semantic cause C1-C8."
    )
    changed[0] = {"role": "system", "content": SEMANTIC_SYSTEM}
    return changed

def relabel(messages, labels, seed):
    _, options = original_choices(messages)
    rng = random.Random(seed)
    displayed = list(labels)
    rng.shuffle(displayed)
    mapping = dict(zip([x[0] for x in options], displayed))
    choices = [(mapping[semantic], desc) for semantic, desc in options]
    rng.shuffle(choices)
    return mapping, choices

def mapping_messages(semantic, choices):
    offered = "\n".join(f"- {label}: {desc}" for label, desc in choices)
    text = (
        f"Semantic diagnosis: {semantic}\n"
        f"Meaning: {CANONICAL.get(semantic, 'Unknown')}\n\n"
        f"Offered choices:\n{offered}\n\n"
        "Generate the displayed label whose description matches the diagnosis."
    )
    return [
        {"role": "system", "content": MAPPER_SYSTEM},
        {"role": "user", "content": text},
    ]

print(balanced["truth"].value_counts().sort_index())
"""
        ),
        cell(
            r"""
# Compute semantic diagnoses once; displayed labels do not affect this pass.
semantic_outputs = []
for index, row in enumerate(balanced.itertuples(index=False), 1):
    raw, answer = generate("diagnostic", semantic_messages(row.messages))
    semantic_outputs.append({
        "index": index, "truth": row.truth,
        "answer": answer, "raw": raw,
        "correct": answer == row.truth,
    })
    if index % 20 == 0:
        print(f"Semantic {index}/80", flush=True)

families = {
    "numeric_1_8": [str(i) for i in range(1, 9)],
    "alphabetic_A_H": list("ABCDEFGH"),
    "unseen_Q1_Q8": [f"Q{i}" for i in range(1, 9)],
    "unseen_V1_V8": [f"V{i}" for i in range(1, 9)],
}
records = []
summaries = []
started = time.perf_counter()

for family, labels in families.items():
    family_rows = []
    print("\nStarting", family, flush=True)
    for index, row in enumerate(balanced.itertuples(index=False), 1):
        mapping, choices = relabel(row.messages, labels, SEED + index)
        semantic = semantic_outputs[index - 1]["answer"]
        raw, answer = generate(
            "mapper", mapping_messages(semantic, choices)
        )
        item = {
            "family": family,
            "index": index,
            "semantic_truth": row.truth,
            "semantic_answer": semantic,
            "displayed_truth": mapping[row.truth],
            "displayed_answer": answer,
            "raw_generation": raw,
            "valid": answer in set(labels),
            "correct": answer == mapping[row.truth],
        }
        records.append(item)
        family_rows.append(item)
        if index % 20 == 0:
            accuracy = sum(x["correct"] for x in family_rows) / len(family_rows)
            print(f"{family}: {index}/80 accuracy={accuracy:.4f}", flush=True)
    part = pd.DataFrame(family_rows)
    summaries.append({
        "family": family,
        "questions": len(part),
        "accuracy": float(part["correct"].mean()),
        "format_success": float(part["valid"].mean()),
    })

summary_frame = pd.DataFrame(summaries)
print("\nLABEL-FAMILY ROBUSTNESS")
print(summary_frame.to_string(index=False))
report = {
    "semantic_accuracy": sum(x["correct"] for x in semantic_outputs) / 80,
    "families": summaries,
    "model_generated_all_answers": True,
    "retrieval": False,
    "cross_dataset_matching": False,
    "rule_answer_overrides": False,
    "records": records,
}
(REPORT_DIR / "qwen25_15b_label_family_robustness.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print("\nSaved report.")
"""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
