from pathlib import Path

import nbformat as nbf


SOURCE = Path("notebooks/Cassava_H200_Frozen_15B_Conversion.ipynb")
OUTPUT = Path("notebooks/Cassava_H200_Standard_Recovery_SaveReload.ipynb")


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def main():
    notebook = nbf.read(SOURCE, as_version=4)

    # Never overwrite the rejected conversion while diagnosing it.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = cell.source.replace(
                'OUTPUT_DIR = Path("/home/jovyan/outputs/qwen25_15b_converted_from_mlx")',
                'OUTPUT_DIR = Path("/home/jovyan/outputs/qwen25_15b_recovery_v2")',
            )

    notebook.cells.extend(
        [
            nbf.v4.new_markdown_cell(
                """## Mandatory save/delete/reload gate

The in-memory conversion result is not accepted by itself. This section records
the in-memory predictions, deletes the converted model, reloads the saved PEFT
adapter into a fresh base model, and repeats the same balanced evaluation."""
            ),
            code(
                r"""
in_memory_summary = dict(summary)
in_memory_records = [dict(item) for item in outputs]
in_memory_predictions = {
    (item["truth"], item["index"]): item["answer"]
    for item in in_memory_records
}

adapter_file = OUTPUT_DIR / "adapter_model.safetensors"
adapter_config_file = OUTPUT_DIR / "adapter_config.json"
assert adapter_file.is_file()
assert adapter_config_file.is_file()

adapter_sha256 = hashlib.sha256(adapter_file.read_bytes()).hexdigest()
print("Saved adapter:", adapter_file)
print("Adapter SHA256:", adapter_sha256)
print("In-memory balanced accuracy:", in_memory_summary["accuracy"])
"""
            ),
            code(
                r"""
import gc

del model
del base_model
gc.collect()
torch.cuda.empty_cache()

reload_config = AutoConfig.from_pretrained(
    MODEL_ID,
    trust_remote_code=False,
)
reload_config.use_sliding_window = False
reload_config.sliding_window = None

reloaded_base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=reload_config,
    quantization_config=quantization_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
    trust_remote_code=False,
)
reloaded_base.config.use_cache = True

reloaded_model = PeftModel.from_pretrained(
    reloaded_base,
    str(OUTPUT_DIR),
    is_trainable=False,
)
reloaded_model.eval()
reloaded_model.config.use_cache = True
reloaded_model.generation_config.do_sample = False
reloaded_model.generation_config.temperature = None
reloaded_model.generation_config.top_p = None
reloaded_model.generation_config.top_k = None

print("Reloaded adapters:", list(reloaded_model.peft_config.keys()))
print("Reloaded adapter configuration:", reloaded_model.peft_config["default"])
print("Reloaded GPU memory GiB:", round(torch.cuda.memory_allocated() / 1024**3, 2))
"""
            ),
            code(
                r"""
def generate_reloaded(messages, max_new_tokens=8):
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
    ).to(reloaded_model.device)
    with torch.inference_mode():
        generated = reloaded_model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, encoded["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return raw, boxed_value(raw)

reloaded_outputs = []
reloaded_started = time.perf_counter()
for index, row in enumerate(balanced.itertuples(index=False), start=1):
    raw, answer = generate_reloaded(row.messages)
    reloaded_outputs.append({
        "index": index,
        "truth": row.truth,
        "answer": answer,
        "raw_generation": raw,
        "valid": answer in {f"C{i}" for i in range(1, 9)},
        "correct": answer == row.truth,
    })
    if index % 10 == 0:
        running = sum(x["correct"] for x in reloaded_outputs) / len(reloaded_outputs)
        print(f"Reloaded {index}/80 accuracy={running:.4f}", flush=True)

reloaded_result = pd.DataFrame(reloaded_outputs)
reloaded_by_class = reloaded_result.groupby("truth").agg(
    questions=("truth", "size"),
    correct=("correct", "sum"),
    accuracy=("correct", "mean"),
)
reloaded_accuracy = float(reloaded_result["correct"].mean())
prediction_agreement = float(
    (pd.DataFrame(in_memory_records)["answer"].reset_index(drop=True)
     == reloaded_result["answer"].reset_index(drop=True)).mean()
)

reload_summary = {
    "experiment": "standard_mlx_conversion_save_reload_gate",
    "questions": len(reloaded_result),
    "in_memory_accuracy": float(in_memory_summary["accuracy"]),
    "reloaded_accuracy": reloaded_accuracy,
    "in_memory_reload_prediction_agreement": prediction_agreement,
    "format_success": float(reloaded_result["valid"].mean()),
    "adapter_sha256": adapter_sha256,
    "elapsed_seconds": time.perf_counter() - reloaded_started,
    "accepted": bool(
        in_memory_summary["accuracy"] >= 0.90
        and reloaded_accuracy >= 0.90
        and prediction_agreement >= 0.99
    ),
}

print("\nMANDATORY SAVE/RELOAD RESULT")
print(json.dumps(reload_summary, indent=2))
print("\nRELOADED PER-CLASS RESULT")
print(reloaded_by_class)

(REPORT_DIR / "qwen25_15b_recovery_save_reload_gate.json").write_text(
    json.dumps(
        {
            **reload_summary,
            "by_class": reloaded_by_class.to_dict(orient="index"),
            "in_memory_records": in_memory_records,
            "reloaded_records": reloaded_outputs,
        },
        indent=2,
    ),
    encoding="utf-8",
)

assert reload_summary["accepted"], (
    "SAVE/RELOAD GATE FAILED. Do not use this adapter and do not run test inference."
)
print("\nPASS: the saved adapter reproduces the in-memory model.")
"""
            ),
            nbf.v4.new_markdown_cell(
                """## Full validation (runs only after the reload gate passes)"""
            ),
            code(
                r"""
full_outputs = []
full_started = time.perf_counter()
for index, row in enumerate(validation_frame.itertuples(index=False), start=1):
    raw, answer = generate_reloaded(row.messages)
    full_outputs.append({
        "index": index,
        "truth": row.truth,
        "answer": answer,
        "raw_generation": raw,
        "valid": answer in {f"C{i}" for i in range(1, 9)},
        "correct": answer == row.truth,
    })
    if index % 25 == 0 or index == len(validation_frame):
        running = sum(x["correct"] for x in full_outputs) / len(full_outputs)
        print(f"Full validation {index}/{len(validation_frame)} accuracy={running:.4f}", flush=True)

full_result = pd.DataFrame(full_outputs)
full_by_class = full_result.groupby("truth").agg(
    questions=("truth", "size"),
    correct=("correct", "sum"),
    accuracy=("correct", "mean"),
)
full_summary = {
    "experiment": "standard_recovery_reloaded_full_validation",
    "questions": len(full_result),
    "correct": int(full_result["correct"].sum()),
    "accuracy": float(full_result["correct"].mean()),
    "format_success": float(full_result["valid"].mean()),
    "adapter_sha256": adapter_sha256,
    "model_reloaded_before_validation": True,
    "model_generated_all_answers": True,
    "retrieval": False,
    "cross_dataset_matching": False,
    "rule_answer_overrides": False,
    "elapsed_seconds": time.perf_counter() - full_started,
}
print("\nRELOADED FULL VALIDATION")
print(json.dumps(full_summary, indent=2))
print(full_by_class)

(REPORT_DIR / "qwen25_15b_recovery_reloaded_full_validation.json").write_text(
    json.dumps(
        {
            **full_summary,
            "by_class": full_by_class.to_dict(orient="index"),
            "records": full_outputs,
        },
        indent=2,
    ),
    encoding="utf-8",
)
assert full_summary["accuracy"] >= 0.90
assert full_summary["format_success"] == 1.0
print("\nACCEPTED: clean, saved, reloaded standard adapter passed full validation.")
"""
            ),
        ]
    )

    notebook.metadata["cassava"]["purpose"] = (
        "Clean standard adapter recovery with mandatory save/delete/reload validation"
    )
    notebook.metadata["cassava"]["mandatory_reload_gate"] = True
    notebook.metadata["cassava"]["test_inference_included"] = False

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)
    print("cells:", len(notebook.cells))


if __name__ == "__main__":
    main()
