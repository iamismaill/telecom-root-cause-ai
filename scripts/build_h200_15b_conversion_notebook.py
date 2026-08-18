"""Build the H200 notebook for validating the frozen MLX LoRA conversion."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/Cassava_H200_Frozen_15B_Conversion.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
    notebook["cells"] = [
        markdown(
            """
# Cassava H200 — frozen Qwen2.5-1.5B MLX adapter conversion

This notebook does **not** retrain or overwrite the frozen local model. It:

1. validates the uploaded package;
2. converts the MLX rank-8 LoRA tensors into an in-memory PEFT adapter;
3. evaluates exactly 80 balanced validation prompts;
4. saves a conversion report and converted adapter only if tensor loading succeeds.

Do not run the full 864-question evaluation unless the balanced result is strong.
"""
        ),
        code(
            r"""
%pip install -q "transformers==4.49.0" "peft==0.14.0" \
  "accelerate==1.3.0" "safetensors>=0.5.2" \
  "bitsandbytes>=0.49.2" "pandas>=2.2.0"

print("If packages changed, restart the kernel once, then continue with Cell 2.")
"""
        ),
        code(
            r"""
from pathlib import Path
import gc
import json
import random
import re
import time

import pandas as pd
import torch
from safetensors import safe_open
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

PACKAGE = Path("/home/jovyan/frozen_local_15b_upload")
MODEL_DIR = PACKAGE / "model"
VALID_JSONL = PACKAGE / "sft_data" / "valid.jsonl"
OUTPUT_DIR = Path("/home/jovyan/outputs/qwen25_15b_converted_from_mlx")
REPORT_DIR = Path("/home/jovyan/reports")
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42
MAX_LENGTH = 2048

required = [
    MODEL_DIR / "adapters.safetensors",
    MODEL_DIR / "adapter_config.json",
    VALID_JSONL,
    PACKAGE / "reports/compliant_enhanced_conditions_v1_validation_full.json",
]
missing = [str(path) for path in required if not path.is_file()]
assert not missing, f"Missing package files: {missing}"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

with VALID_JSONL.open(encoding="utf-8") as handle:
    validation_records = [json.loads(line) for line in handle if line.strip()]

assert len(validation_records) == 864
print("Package verified.")
print("Validation conversations:", len(validation_records))
print("GPU:", torch.cuda.get_device_name(0))
print("GPU GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
"""
        ),
        code(
            r"""
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=False,
)
base_model.config.use_cache = True

# MLX used scale=20 directly. PEFT scale is alpha/r, hence alpha=20*8=160.
lora_config = LoraConfig(
    r=8,
    lora_alpha=160,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    layers_to_transform=list(range(12, 28)),
    layers_pattern="layers",
    inference_mode=True,
)

model = get_peft_model(base_model, lora_config)
model.eval()
print("PEFT adapter structure created.")
model.print_trainable_parameters()
"""
        ),
        code(
            r"""
mlx_path = MODEL_DIR / "adapters.safetensors"
named_parameters = dict(model.named_parameters())
converted = 0
unmatched = []

with safe_open(str(mlx_path), framework="pt", device="cpu") as handle:
    mlx_keys = list(handle.keys())
    assert len(mlx_keys) == 224, f"Expected 224 MLX tensors, found {len(mlx_keys)}"

    with torch.no_grad():
        for mlx_key in mlx_keys:
            tensor = handle.get_tensor(mlx_key)
            if mlx_key.endswith(".lora_a"):
                suffix = mlx_key[:-7] + ".lora_A.default.weight"
            elif mlx_key.endswith(".lora_b"):
                suffix = mlx_key[:-7] + ".lora_B.default.weight"
            else:
                unmatched.append((mlx_key, "unsupported MLX key"))
                continue

            candidates = [
                name for name in named_parameters
                if name.endswith(suffix)
            ]
            if len(candidates) != 1:
                unmatched.append((mlx_key, candidates))
                continue

            target_name = candidates[0]
            target = named_parameters[target_name]
            converted_tensor = tensor.T.contiguous()
            if tuple(converted_tensor.shape) != tuple(target.shape):
                unmatched.append(
                    (mlx_key, tuple(converted_tensor.shape), target_name, tuple(target.shape))
                )
                continue

            target.copy_(
                converted_tensor.to(
                    device=target.device,
                    dtype=target.dtype,
                )
            )
            converted += 1

assert not unmatched, f"Unmatched conversion entries: {unmatched[:5]}"
assert converted == 224, f"Converted {converted}/224 tensors"

model.save_pretrained(str(OUTPUT_DIR), safe_serialization=True)
tokenizer.save_pretrained(str(OUTPUT_DIR))
print("Converted tensors:", converted)
print("Converted PEFT adapter saved to:", OUTPUT_DIR)
"""
        ),
        code(
            r"""
BOXED_RE = re.compile(
    r"\\+boxed\{\s*([A-Za-z0-9]+)\s*\}",
    re.IGNORECASE,
)

def boxed_value(text):
    match = BOXED_RE.search(str(text).strip())
    return match.group(1).upper() if match else ""

rows = []
for record in validation_records:
    messages = record["messages"]
    truth = boxed_value(messages[-1]["content"])
    assert truth in {f"C{i}" for i in range(1, 9)}
    rows.append({
        "messages": messages[:-1],
        "truth": truth,
    })

validation_frame = pd.DataFrame(rows)
balanced = (
    validation_frame
    .groupby("truth", group_keys=False)
    .sample(n=10, random_state=SEED)
    .reset_index(drop=True)
)
assert len(balanced) == 80
print(balanced["truth"].value_counts().sort_index())

def generate_answer(messages, max_new_tokens=8):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, encoded["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return raw, boxed_value(raw)

outputs = []
started = time.perf_counter()
for index, row in enumerate(balanced.itertuples(index=False), start=1):
    raw, answer = generate_answer(row.messages)
    outputs.append({
        "index": index,
        "truth": row.truth,
        "answer": answer,
        "raw_generation": raw,
        "valid": answer in {f"C{i}" for i in range(1, 9)},
        "correct": answer == row.truth,
    })
    if index % 10 == 0:
        running = sum(item["correct"] for item in outputs) / len(outputs)
        print(f"{index}/80 running_accuracy={running:.4f}", flush=True)

result = pd.DataFrame(outputs)
by_class = result.groupby("truth").agg(
    questions=("truth", "size"),
    correct=("correct", "sum"),
    accuracy=("correct", "mean"),
)
summary = {
    "experiment": "frozen_mlx_to_peft_conversion",
    "base_model": MODEL_ID,
    "questions": len(result),
    "correct": int(result["correct"].sum()),
    "accuracy": float(result["correct"].mean()),
    "format_success": float(result["valid"].mean()),
    "elapsed_seconds": time.perf_counter() - started,
    "mlx_scale": 20.0,
    "peft_rank": 8,
    "peft_alpha": 160,
    "full_validation_authorized": bool(result["correct"].mean() >= 0.80),
}
print("\nBALANCED CONVERSION RESULT")
print(json.dumps(summary, indent=2))
print(by_class)

(REPORT_DIR / "qwen25_15b_conversion_balanced.json").write_text(
    json.dumps(
        {
            **summary,
            "by_class": by_class.to_dict(orient="index"),
            "records": outputs,
        },
        indent=2,
    ),
    encoding="utf-8",
)

if not summary["full_validation_authorized"]:
    print("\nSTOP: conversion did not meet the 80% balanced gate.")
    print("Do not run a full validation or test submission from this conversion.")
else:
    print("\nPASS: conversion met the balanced gate.")
    print("Ask Codex for the full 864-question evaluation cell.")
"""
        ),
    ]
    notebook["metadata"]["cassava"] = {
        "purpose": "MLX-to-PEFT frozen adapter conversion validation",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_parameters_under_4b": True,
        "retrieval": False,
        "cross_dataset_matching": False,
        "rule_answer_override": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(json.dumps({"notebook": str(OUTPUT), "cells": len(notebook["cells"])}, indent=2))


if __name__ == "__main__":
    main()
