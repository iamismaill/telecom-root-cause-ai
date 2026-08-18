"""Build an H200 notebook for the isolated nine-cause semantic adapter."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/Cassava_H200_Nine_Cause_Synthetic.ipynb"


def c(source: str):
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
            """# Cassava H200 — isolated nine-cause semantic adapter

Trains a separate Qwen2.5-1.5B LoRA on balanced synthetic engineering
summaries. It does not use test labels, retrieve another question, or modify
the frozen eight-cause diagnostic and mapper adapters. Stop after synthetic
holdout evaluation; test inference is not authorized by this notebook."""
        ),
        c(
            r"""
from pathlib import Path
import json, random
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = Path("/home/jovyan/outputs/qwen25_15b_nine_cause_v1")
REPORT_DIR = Path("/home/jovyan/reports")
SEED = 42
MAX_LENGTH = 768
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CAUSES = {
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
SYSTEM = (
    "You are a 5G root-cause diagnostician. Use only the supplied engineering "
    "evidence. Select one canonical nine-cause diagnosis. Your entire response "
    "must be exactly one of \\boxed{N1} through \\boxed{N9}."
)
CAUSE_LINES = "\n".join(f"- {k}: {v}" for k, v in CAUSES.items())
print("Nine-cause taxonomy loaded.")
"""
        ),
        c(
            r"""
def normal(rng):
    return {
        "serving_rsrp_dbm": rng.uniform(-91, -76),
        "serving_sinr_db": rng.uniform(10, 22),
        "strongest_neighbor_margin_db": rng.uniform(-10, -3),
        "strong_neighbor_count": rng.choice([0, 1]),
        "cce_fail_rate": rng.uniform(0.00, 0.06),
        "average_rank": rng.uniform(2.0, 3.5),
        "average_mcs": rng.uniform(14, 25),
        "rb_utilization": rng.uniform(0.45, 0.75),
        "initial_bler_percent": rng.uniform(2, 10),
        "residual_bler_percent": rng.uniform(0, 2),
        "a3_offset_db": rng.uniform(2, 4),
        "a3_ttt_ms": rng.choice([160, 320]),
        "inter_a2_threshold_dbm": rng.uniform(-110, -100),
        "inter_a5_threshold1_dbm": rng.uniform(-110, -100),
        "inter_a5_threshold2_dbm": rng.uniform(-100, -90),
        "observed_strong_neighbor_configured": "TRUE",
        "handover_pattern": "stable",
        "pdcch_symbols": rng.choice([2, 3]),
        "radio_load_balance": "balanced",
        "server_transport_evidence": "normal",
        "site_coverage_evidence": "normal",
    }

def scenario(cause, rng):
    e = normal(rng)
    if cause == "N1":
        e.update(
            serving_rsrp_dbm=rng.uniform(-92, -78),
            serving_sinr_db=rng.uniform(-6, 4),
            strongest_neighbor_margin_db=rng.uniform(-1, 5),
            strong_neighbor_count=rng.randint(2, 4),
            site_coverage_evidence="multiple strong overlapping co-frequency cells",
        )
    elif cause == "N2":
        e.update(
            inter_a2_threshold_dbm=rng.uniform(-126, -116),
            inter_a5_threshold1_dbm=rng.uniform(-125, -116),
            inter_a5_threshold2_dbm=rng.uniform(-115, -106),
            handover_pattern="inter-frequency trigger delayed or absent",
        )
    elif cause == "N3":
        e.update(
            rb_utilization=rng.uniform(0.93, 1.0),
            average_mcs=rng.uniform(15, 25),
            radio_load_balance="serving cell congested while neighbor load is low",
            server_transport_evidence="normal",
        )
    elif cause == "N4":
        e.update(
            rb_utilization=rng.uniform(0.35, 0.70),
            average_mcs=rng.uniform(15, 26),
            initial_bler_percent=rng.uniform(1, 7),
            residual_bler_percent=rng.uniform(0, 1),
            server_transport_evidence="insufficient upstream traffic despite healthy radio",
        )
    elif cause == "N5":
        e.update(
            strongest_neighbor_margin_db=rng.uniform(5, 15),
            strong_neighbor_count=rng.randint(1, 3),
            observed_strong_neighbor_configured="FALSE",
            handover_pattern="strong neighbor observed but no handover",
        )
    elif cause == "N6":
        e.update(
            serving_rsrp_dbm=rng.uniform(-122, -108),
            serving_sinr_db=rng.uniform(-8, 2),
            strongest_neighbor_margin_db=rng.uniform(-12, -3),
            strong_neighbor_count=0,
            site_coverage_evidence="serving and all neighbors weak; coverage hole",
        )
    elif cause == "N7":
        e.update(
            a3_offset_db=rng.uniform(7, 12),
            a3_ttt_ms=rng.choice([320, 480, 640]),
            strongest_neighbor_margin_db=rng.uniform(6, 14),
            handover_pattern="handover delayed while neighbor is much stronger",
        )
    elif cause == "N8":
        e.update(
            a3_offset_db=rng.uniform(-1, 1),
            a3_ttt_ms=rng.choice([40, 64, 80]),
            strongest_neighbor_margin_db=rng.uniform(-1, 2),
            handover_pattern="premature repeated ping-pong handovers",
        )
    elif cause == "N9":
        e.update(
            cce_fail_rate=rng.uniform(0.22, 0.55),
            pdcch_symbols=1,
            average_rank=rng.uniform(1.8, 3.0),
            average_mcs=rng.uniform(12, 23),
            server_transport_evidence="normal",
        )
    return e

def prompt(e):
    evidence = "\n".join(f"- {k}: {v:.4g}" if isinstance(v, float)
                         else f"- {k}: {v}" for k, v in e.items())
    return (
        f"Canonical causes:\n{CAUSE_LINES}\n\n"
        f"Current-question engineering summary:\n{evidence}\n\n"
        "Compare all nine causes and generate only the best canonical N-label."
    )

def build(count, salt):
    rng = random.Random(f"{SEED}:{salt}")
    labels = list(CAUSES)
    rows = []
    for i in range(count):
        label = labels[i % 9]
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt(scenario(label, rng))},
                {"role": "assistant", "content": f"\\boxed{{{label}}}"},
            ]
        })
    rng.shuffle(rows)
    return rows

train_rows = build(3600, "train")
valid_rows = build(450, "holdout")
print("Synthetic rows:", len(train_rows), len(valid_rows))
"""
        ),
        c(
            r"""
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

def encode(row):
    messages = row["messages"]
    p = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    a = messages[-1]["content"] + tokenizer.eos_token
    pids = tokenizer(p, add_special_tokens=False).input_ids
    aids = tokenizer(a, add_special_tokens=False).input_ids
    assert len(pids) + len(aids) <= MAX_LENGTH
    return {
        "input_ids": pids + aids,
        "attention_mask": [1] * (len(pids) + len(aids)),
        "labels": [-100] * len(pids) + aids,
    }

train_ds = Dataset.from_list(train_rows).map(
    encode, remove_columns=["messages"], desc="Tokenizing nine-cause train"
)
valid_ds = Dataset.from_list(valid_rows).map(
    encode, remove_columns=["messages"], desc="Tokenizing nine-cause holdout"
)

class Collator:
    def __call__(self, features):
        width = max(len(x["input_ids"]) for x in features)
        ids, masks, labels = [], [], []
        for x in features:
            pad = width - len(x["input_ids"])
            ids.append(x["input_ids"] + [tokenizer.pad_token_id] * pad)
            masks.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(masks),
            "labels": torch.tensor(labels),
        }
print("Tokenized:", len(train_ds), len(valid_ds))
"""
        ),
        c(
            r"""
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=quant, device_map={"": 0},
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
)
base.config.use_cache = False
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(base, config, adapter_name="nine_cause")
model.print_trainable_parameters()
args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=1.0,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    optim="paged_adamw_8bit",
    bf16=True, tf32=True,
    logging_steps=10,
    eval_strategy="steps", eval_steps=50,
    save_strategy="steps", save_steps=50, save_total_limit=2,
    report_to="none", remove_unused_columns=False,
    label_names=["labels"], seed=SEED, data_seed=SEED,
)
trainer = Trainer(
    model=model, args=args, train_dataset=train_ds,
    eval_dataset=valid_ds, data_collator=Collator(),
)
print("Starting isolated nine-cause training...", flush=True)
trainer.train()
final_path = OUTPUT_DIR / "final_adapter"
model.save_pretrained(str(final_path), selected_adapters=["nine_cause"])
tokenizer.save_pretrained(str(final_path))
print("Saved:", final_path)
"""
        ),
        nbf.v4.new_markdown_cell(
            """Stop here and send Codex the training and validation losses. The
synthetic generated-answer holdout and real-question summary extractor are
separate acceptance gates."""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
