"""Build the clean, standalone Cassava H200 training notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/Cassava_H200_Compliant_Qwen3B.ipynb"


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).lstrip().splitlines(keepends=True),
    }


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).lstrip().splitlines(keepends=True),
    }


cells = [
    markdown(
        r"""
        # Cassava AI Root Cause Detective — compliant H200 training

        This notebook trains and evaluates an open-source language model under 4B
        parameters. It uses only the official challenge files. Every evaluated and
        submitted answer is generated through Qwen inference.

        Compliance boundaries:

        - no test-to-train or test-to-validation matching;
        - no retrieval of an existing label;
        - no classical model or deterministic answer override;
        - validation remains completely outside training;
        - raw generations, prompts, settings and hashes are saved.

        Expected upload layout:

        ```
        data/
          train.csv
          validation_questions.csv
          validation_target.csv
          test.csv
          SampleSubmission.csv
        ```
        """
    ),
    code(
        """
        # Run once in the Cassava notebook environment.
        %pip install -q "transformers==4.49.0" "peft==0.14.0" \\
          "accelerate==1.3.0" "datasets==3.2.0" "sentencepiece>=0.2.0" \\
          "safetensors>=0.5.2" "pandas>=2.2.0"
        """
    ),
    code(
        """
        import hashlib
        import json
        import os
        import platform
        import random
        import re
        import time
        from collections import Counter
        from datetime import datetime, timezone
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )

        SEED = 42
        MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
        MAX_LENGTH = 16384
        DATA_DIR = Path("data")
        OUTPUT_DIR = Path("outputs/qwen25_3b_h200_v1")
        REPORT_DIR = Path("reports")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        set_seed(SEED)

        assert torch.cuda.is_available(), "GPU is not visible"
        properties = torch.cuda.get_device_properties(0)
        print("GPU:", properties.name)
        print("GPU memory GiB:", round(properties.total_memory / 2**30, 2))
        print("PyTorch:", torch.__version__)
        print("CUDA:", torch.version.cuda)
        """
    ),
    code(
        r"""
        required = {
            "train.csv",
            "validation_questions.csv",
            "validation_target.csv",
            "test.csv",
            "SampleSubmission.csv",
        }
        missing = sorted(name for name in required if not (DATA_DIR / name).is_file())
        assert not missing, f"Upload these files into data/: {missing}"

        train_df = pd.read_csv(DATA_DIR / "train.csv")
        validation_questions = pd.read_csv(DATA_DIR / "validation_questions.csv")
        validation_targets = pd.read_csv(DATA_DIR / "validation_target.csv")

        target_column = next(
            column for column in validation_targets.columns if column != "ID"
        )
        validation_targets = validation_targets.assign(
            base_ID=validation_targets["ID"].str.replace(
                r"_[1-4]$", "", regex=True
            )
        )
        target_consistency = validation_targets.groupby("base_ID")[target_column].nunique()
        assert target_consistency.eq(1).all(), "Repeated validation targets disagree"
        validation_targets = (
            validation_targets.groupby("base_ID", as_index=False)[target_column]
            .first()
            .rename(columns={"base_ID": "ID"})
        )
        validation_df = validation_questions.merge(
            validation_targets[["ID", target_column]], on="ID", validate="one_to_one"
        ).rename(columns={target_column: "answer"})

        assert set(train_df.columns) >= {"ID", "question", "answer"}
        assert len(validation_df) == 864
        assert set(train_df.answer) == {f"C{i}" for i in range(1, 9)}
        assert set(validation_df.answer) == {f"C{i}" for i in range(1, 9)}
        print("Train:", train_df.shape)
        print("Validation:", validation_df.shape)
        print(train_df.answer.value_counts().sort_index())
        """
    ),
    code(
        r"""
        SYSTEM = (
            "You are a careful multiple-choice 5G root-cause diagnostician. "
            "Analyze only the current question. Select one offered choice. "
            "Your entire response must be exactly one answer in the form "
            "\\boxed{CHOICE}, with no explanation or additional text."
        )
        OPTION_RE = re.compile(
            r"^\s*(C[1-8]|[1-8])\s*[:.)]\s*(.+?)\s*$", re.IGNORECASE
        )

        def stable_seed(identifier: str, salt: str) -> int:
            digest = hashlib.sha256(f"{salt}:{identifier}".encode()).hexdigest()
            return int(digest[:8], 16)

        def option_lines(question: str):
            found = []
            for index, line in enumerate(question.splitlines()):
                if "|" in line:
                    continue
                match = OPTION_RE.match(line)
                if match:
                    found.append((index, match.group(1), match.group(2)))
            if len(found) != 8:
                raise ValueError(f"Expected 8 options; found {len(found)}")
            return found

        def shuffled_numeric_example(identifier: str, question: str, answer: str):
            lines = question.splitlines()
            found = option_lines(question)
            correct_description = next(
                description
                for _, label, description in found
                if label.upper() == answer.upper()
            )
            descriptions = [description for _, _, description in found]
            rng = random.Random(stable_seed(identifier, "h200-option-shuffle"))
            rng.shuffle(descriptions)
            for new_label, ((line_index, _, _), description) in enumerate(
                zip(found, descriptions), start=1
            ):
                lines[line_index] = f"{new_label}: {description}"
            displayed = str(descriptions.index(correct_description) + 1)
            return "\n".join(lines), displayed

        def chat_record(question: str, displayed_answer: str):
            return {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": f"\\boxed{{{displayed_answer}}}"},
                ]
            }

        train_records = []
        for row in train_df.itertuples(index=False):
            train_records.append(chat_record(row.question, row.answer))
            changed, displayed = shuffled_numeric_example(row.ID, row.question, row.answer)
            train_records.append(chat_record(changed, displayed))
            # Focused additional exposure for the two previously weak semantic causes.
            if row.answer in {"C3", "C7"}:
                changed2, displayed2 = shuffled_numeric_example(
                    f"{row.ID}-focus", row.question, row.answer
                )
                train_records.append(chat_record(changed2, displayed2))

        random.Random(SEED).shuffle(train_records)
        validation_records = [
            chat_record(row.question, row.answer)
            for row in validation_df.itertuples(index=False)
        ]
        print("Training conversations:", len(train_records))
        print("Validation conversations:", len(validation_records))
        assert all(record["messages"][-1]["content"].startswith("\\boxed{") for record in train_records)
        """
    ),
    code(
        r"""
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        def encode_record(record):
            messages = record["messages"]
            prompt_text = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            answer_text = messages[-1]["content"] + tokenizer.eos_token
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
            budget = MAX_LENGTH - len(answer_ids)
            if len(prompt_ids) > budget:
                # Preserve the question instructions/options and engineering tail.
                head = budget // 2
                prompt_ids = prompt_ids[:head] + prompt_ids[-(budget - head):]
            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }

        token_lengths = []
        for sample in train_records[: min(500, len(train_records))]:
            rendered = tokenizer.apply_chat_template(
                sample["messages"], tokenize=False, add_generation_prompt=False
            )
            token_lengths.append(len(tokenizer(rendered, add_special_tokens=False).input_ids))
        print(
            "Token length sample percentiles:",
            {p: int(np.percentile(token_lengths, p)) for p in [50, 90, 95, 99, 100]},
        )

        train_dataset = Dataset.from_list(train_records).map(
            encode_record, remove_columns=["messages"], desc="Tokenizing train"
        )
        validation_dataset = Dataset.from_list(validation_records).map(
            encode_record, remove_columns=["messages"], desc="Tokenizing validation"
        )

        class CausalCollator:
            def __call__(self, features):
                width = max(len(item["input_ids"]) for item in features)
                input_ids, attention_mask, labels = [], [], []
                for item in features:
                    padding = width - len(item["input_ids"])
                    input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding)
                    attention_mask.append(item["attention_mask"] + [0] * padding)
                    labels.append(item["labels"] + [-100] * padding)
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
        """
    ),
    code(
        """
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=False,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
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

        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            num_train_epochs=1.0,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            bf16=True,
            tf32=True,
            gradient_checkpointing=True,
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
            eval_dataset=validation_dataset,
            data_collator=CausalCollator(),
        )
        train_result = trainer.train()
        trainer.save_model(OUTPUT_DIR / "final_adapter")
        tokenizer.save_pretrained(OUTPUT_DIR / "final_adapter")
        print(train_result)
        """
    ),
    code(
        r"""
        model.eval()
        model.config.use_cache = True
        BOXED_RE = re.compile(r"\\boxed\\{\\s*([A-Za-z0-9]+)\\s*\\}")

        def model_generate(question: str, max_new_tokens: int = 16):
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
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
            new_tokens = generated[0, encoded.input_ids.shape[1]:]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return raw, hashlib.sha256(prompt.encode()).hexdigest()

        def allowed_labels(question: str):
            return {label for _, label, _ in option_lines(question)}

        def explicit_label(raw: str, allowed: set[str]):
            match = BOXED_RE.search(raw)
            if match and match.group(1) in allowed:
                return match.group(1)
            if raw.strip() in allowed:
                return raw.strip()
            return ""

        validation_outputs = []
        started = time.perf_counter()
        for index, row in enumerate(validation_df.itertuples(index=False), start=1):
            raw, prompt_hash = model_generate(row.question)
            answer = explicit_label(raw, allowed_labels(row.question))
            validation_outputs.append(
                {
                    "ID": row.ID,
                    "truth": row.answer,
                    "answer": answer,
                    "raw_generation": raw,
                    "prompt_sha256": prompt_hash,
                    "valid": bool(answer),
                    "correct": answer == row.answer,
                }
            )
            if index % 25 == 0:
                print(f"{index}/{len(validation_df)}")

        validation_result = pd.DataFrame(validation_outputs)
        by_class = validation_result.groupby("truth").agg(
            questions=("ID", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
        )
        print(by_class)
        print("Overall accuracy:", validation_result.correct.mean())
        print("Format success:", validation_result.valid.mean())

        validation_report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_ID,
            "parameters_under_4b": True,
            "adapter": str(OUTPUT_DIR / "final_adapter"),
            "questions": len(validation_result),
            "correct": int(validation_result.correct.sum()),
            "accuracy": float(validation_result.correct.mean()),
            "format_success": float(validation_result.valid.mean()),
            "by_class": by_class.to_dict(orient="index"),
            "elapsed_seconds": time.perf_counter() - started,
            "model_generated_all_answers": True,
            "retrieval": False,
            "cross_dataset_matching": False,
            "rule_answer_overrides": False,
            "records": validation_outputs,
        }
        (REPORT_DIR / "h200_validation_report.json").write_text(
            json.dumps(validation_report, indent=2), encoding="utf-8"
        )
        """
    ),
    code(
        r"""
        # Robustness test: shuffled/relabelled validation choices.
        shifted_outputs = []
        for index, row in enumerate(validation_df.itertuples(index=False), start=1):
            changed, displayed_truth = shuffled_numeric_example(
                f"{row.ID}-validation-shift", row.question, row.answer
            )
            raw, prompt_hash = model_generate(changed)
            answer = explicit_label(raw, allowed_labels(changed))
            shifted_outputs.append(
                {
                    "ID": row.ID,
                    "truth": displayed_truth,
                    "answer": answer,
                    "raw_generation": raw,
                    "prompt_sha256": prompt_hash,
                    "valid": bool(answer),
                    "correct": answer == displayed_truth,
                }
            )
            if index % 50 == 0:
                print(f"shifted {index}/{len(validation_df)}")

        shifted_result = pd.DataFrame(shifted_outputs)
        print("Shuffled-option accuracy:", shifted_result.correct.mean())
        print("Shuffled-option format success:", shifted_result.valid.mean())
        (REPORT_DIR / "h200_shifted_validation_report.json").write_text(
            json.dumps(
                {
                    "model": MODEL_ID,
                    "transformation": "deterministic option shuffle and numeric relabel",
                    "questions": len(shifted_result),
                    "accuracy": float(shifted_result.correct.mean()),
                    "format_success": float(shifted_result.valid.mean()),
                    "records": shifted_outputs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        """
    ),
    code(
        r"""
        # Compliance manifest and reproducibility bundle.
        def file_sha256(path: Path):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "competition": "Cassava AI Root Cause Detective Hackathon",
            "base_model": MODEL_ID,
            "base_model_open_source": True,
            "base_model_parameters_under_4b": True,
            "training_files": {
                name: file_sha256(DATA_DIR / name)
                for name in ["train.csv", "validation_questions.csv", "validation_target.csv"]
            },
            "test_labels_used": False,
            "retrieval_used": False,
            "cross_dataset_matching_used": False,
            "deterministic_answer_override_used": False,
            "all_final_answers_generated_by_model": True,
            "seed": SEED,
            "max_length": MAX_LENGTH,
            "system_prompt": SYSTEM,
            "environment": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
        }
        (REPORT_DIR / "h200_compliance_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2))

        # Do not generate a test submission unless validation and robustness are accepted.
        print("Training/evaluation complete. Review both reports before test inference.")
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
