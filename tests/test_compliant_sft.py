import json
from pathlib import Path

from scripts.prepare_compliant_sft import SYSTEM, record


def test_compliant_record_requires_model_generated_boxed_answer() -> None:
    value = record("Question\n1: A\n2: B", "2")
    messages = value["messages"]
    assert messages[0] == {"role": "system", "content": SYSTEM}
    assert messages[-1] == {"role": "assistant", "content": r"\boxed{2}"}
    assert "current question" in SYSTEM
