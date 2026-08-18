"""Local deterministic Qwen2.5 inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BOXED_PATTERN = re.compile(r"\\boxed\s*\{\s*([^{}]+?)\s*\}")


def extract_boxed_answer(text: str, allowed: set[str] | None = None) -> str:
    """Extract exactly one boxed answer and optionally validate its domain."""
    matches = [match.strip() for match in BOXED_PATTERN.findall(str(text))]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one boxed answer, found {len(matches)}")
    answer = matches[0]
    if allowed is not None and answer not in allowed:
        raise ValueError(f"Boxed answer {answer!r} is not in {sorted(allowed)}")
    return answer


def zero_shot_messages(question: str, example_choice: str = "2") -> list[dict[str, str]]:
    """Create a minimal prompt that preserves the official question verbatim."""
    return [
        {
            "role": "system",
            "content": (
                "You solve multiple-choice questions accurately. Follow the answer choices in the "
                "user's question. Return exactly one final answer in the form \\boxed{CHOICE}. "
                "Do not output an explanation or any other text."
            ),
        },
        {
            "role": "user",
            "content": (
                question
                + "\n\nYour entire response must be exactly one boxed choice, for example: "
                + rf"\boxed{{{example_choice}}}"
            ),
        },
    ]


@dataclass(frozen=True)
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float


class LocalQwen:
    """One-model local runtime with deterministic decoding."""

    def __init__(self, model_path: Path, device: str | None = None) -> None:
        if not model_path.is_dir():
            raise FileNotFoundError(f"Local model directory not found: {model_path}")
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        dtype = torch.float16 if self.device.type == "mps" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

    def generate(
        self,
        question: str,
        max_new_tokens: int = 16,
        example_choice: str = "2",
    ) -> GenerationResult:
        messages = zero_shot_messages(question, example_choice=example_choice)
        return self.generate_messages(messages, max_new_tokens=max_new_tokens)

    def generate_messages(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 16,
    ) -> GenerationResult:
        """Generate deterministically from an explicit local chat transcript."""
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(rendered, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if self.device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        generated = output[0, input_ids.shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            input_tokens=int(input_ids.shape[1]),
            output_tokens=int(generated.shape[0]),
            elapsed_seconds=elapsed,
        )
