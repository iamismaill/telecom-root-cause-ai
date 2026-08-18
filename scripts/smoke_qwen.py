"""Run one harmless local smoke inference without challenge predictions."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.qwen import LocalQwen, extract_boxed_answer  # noqa: E402


def main() -> None:
    runtime = LocalQwen(ROOT / "models" / "Qwen2.5-1.5B-Instruct")
    question = "Select the correct answer. What is 2 + 2?\n1: 3\n2: 4\n3: 5\n4: 6"
    result = runtime.generate(question, max_new_tokens=32)
    print(f"raw_text={result.text!r}")
    answer = extract_boxed_answer(result.text, {"1", "2", "3", "4"})
    if answer != "2":
        raise RuntimeError(f"Smoke answer was structurally valid but incorrect: {answer}")
    print(f"device={runtime.device.type}")
    print(f"text={result.text}")
    print(f"input_tokens={result.input_tokens}")
    print(f"output_tokens={result.output_tokens}")
    print(f"elapsed_seconds={result.elapsed_seconds:.4f}")


if __name__ == "__main__":
    main()
