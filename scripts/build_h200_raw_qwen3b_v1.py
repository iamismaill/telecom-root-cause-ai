"""Build the clean raw-question Qwen2.5-3B QLoRA notebook."""

from pathlib import Path
import json
import runpy
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/build_h200_notebook.py"
OUTPUT = ROOT / "notebooks/Cassava_H200_RawQuestion_Qwen3B_V1.ipynb"


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).lstrip().splitlines(keepends=True),
    }


namespace = runpy.run_path(str(SOURCE))
cells = namespace["cells"]

# Include the 4-bit backend explicitly for a clean kernel.
install_source = "".join(cells[1]["source"]).replace(
    '"safetensors>=0.5.2" "pandas>=2.2.0"',
    '"safetensors>=0.5.2" "pandas>=2.2.0" "bitsandbytes>=0.49.2"',
)
cells[1]["source"] = install_source.splitlines(keepends=True)

# Make this a new experiment with raw questions and a practical sequence length.
config_source = "".join(cells[2]["source"])
config_source = config_source.replace(
    "MAX_LENGTH = 16384",
    "MAX_LENGTH = 3072",
).replace(
    'OUTPUT_DIR = Path("outputs/qwen25_3b_h200_v1")',
    'OUTPUT_DIR = Path("outputs/qwen25_3b_raw_v1")',
)
cells[2]["source"] = config_source.splitlines(keepends=True)

# No synthetic relabeling, computed conditions, retrieval, or answer rules.
cells[4] = code(
    r"""
    SYSTEM = (
        "You are a careful multiple-choice 5G root-cause diagnostician. "
        "Analyze only the raw current question, including its drive-test table, "
        "engineering parameters, and offered choices. Select the most likely "
        "offered choice. Your entire response must be exactly one answer in "
        "the form \\boxed{CHOICE}, with no explanation or additional text. "
        "Never retrieve or match another question."
    )

    def chat_record(question: str, answer: str):
        return {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": str(question)},
                {"role": "assistant", "content": f"\\boxed{{{answer}}}"},
            ]
        }

    train_records = [
        chat_record(row.question, row.answer)
        for row in train_df.itertuples(index=False)
    ]
    validation_records = [
        chat_record(row.question, row.answer)
        for row in validation_df.itertuples(index=False)
    ]

    assert len(train_records) == 2400
    assert len(validation_records) == 864
    assert all(
        record["messages"][-1]["content"].startswith("\\boxed{")
        for record in train_records
    )

    print("Raw training conversations:", len(train_records))
    print("Untouched raw validation conversations:", len(validation_records))
    print("No synthetic relabeling or computed-condition prompts were added.")
    """
)

# Memory-safe raw QLoRA training.
cells[6] = code(
    r"""
    import gc
    from transformers import AutoConfig, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    gc.collect()
    torch.cuda.empty_cache()

    model_config = AutoConfig.from_pretrained(
        MODEL_ID,
        trust_remote_code=False,
    )
    model_config.use_sliding_window = False
    model_config.sliding_window = None

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=model_config,
        quantization_config=quantization_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    monitoring_validation_dataset = validation_dataset.select(
        range(min(128, len(validation_dataset)))
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        optim="paged_adamw_8bit",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=4,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=monitoring_validation_dataset,
        data_collator=CausalCollator(),
    )

    print("Starting clean raw-question QLoRA training...")
    train_result = trainer.train()

    final_adapter_path = OUTPUT_DIR / "final_adapter"
    trainer.save_model(str(final_adapter_path))
    tokenizer.save_pretrained(str(final_adapter_path))

    print(train_result)
    print("Saved raw adapter:", final_adapter_path)
    """
)

# Mandatory delete and reload before any generated-answer evaluation.
reload_cell = code(
    r"""
    import gc

    final_adapter_path = OUTPUT_DIR / "final_adapter"
    assert (final_adapter_path / "adapter_model.safetensors").is_file()
    adapter_sha256 = hashlib.sha256(
        (final_adapter_path / "adapter_model.safetensors").read_bytes()
    ).hexdigest()

    del trainer
    del model
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
    model = PeftModel.from_pretrained(
        reloaded_base,
        str(final_adapter_path),
        is_trainable=False,
    )
    model.eval()
    model.config.use_cache = True
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    print("Freshly reloaded adapter:", list(model.peft_config.keys()))
    print("Adapter SHA256:", adapter_sha256)
    print("GPU memory GiB:", round(torch.cuda.memory_allocated() / 1024**3, 2))
    """
)
cells.insert(7, reload_cell)

# Rename the full raw validation report and record the mandatory reload.
evaluation_source = "".join(cells[8]["source"])
evaluation_source = evaluation_source.replace(
    'REPORT_DIR / "h200_validation_report.json"',
    'REPORT_DIR / "qwen25_3b_raw_v1_reloaded_validation.json"',
).replace(
    '"adapter": str(OUTPUT_DIR / "final_adapter"),',
    '"adapter": str(final_adapter_path),\n'
    '            "adapter_sha256": adapter_sha256,\n'
    '            "model_reloaded_before_validation": True,',
)
cells[8]["source"] = evaluation_source.splitlines(keepends=True)

# Remove the inherited synthetic relabel test. A later notebook may add carefully
# validated transformations only after raw validation succeeds.
cells[9] = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Stop after raw validation\n\n",
        "Do not generate test predictions or add label transformations until the ",
        "freshly reloaded model's 864-question raw validation report is reviewed.\n",
    ],
}

manifest_source = "".join(cells[10]["source"]).replace(
    'REPORT_DIR / "h200_compliance_manifest.json"',
    'REPORT_DIR / "qwen25_3b_raw_v1_compliance_manifest.json"',
)
cells[10]["source"] = manifest_source.splitlines(keepends=True)

cells[0]["source"] = [
    "# Cassava H200 — clean raw-question Qwen2.5-3B QLoRA V1\n\n",
    "Trains only on the 2,400 official raw training questions. No computed ",
    "TRUE/FALSE conditions, retrieval, cross-dataset matching, synthetic ",
    "relabeling, or deterministic answer override is used. The adapter is ",
    "deleted and reloaded from disk before raw validation.\n",
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "cassava": {
            "raw_questions_only": True,
            "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "parameters_under_4b": True,
            "mandatory_save_reload": True,
            "test_inference_included": False,
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
