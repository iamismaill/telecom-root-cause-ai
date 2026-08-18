import pytest

from telecom_rca.qwen import extract_boxed_answer, zero_shot_messages


def test_extracts_one_strict_boxed_answer() -> None:
    assert extract_boxed_answer(r"\boxed{C7}", {"C7"}) == "C7"


def test_rejects_missing_multiple_or_invalid_answers() -> None:
    with pytest.raises(ValueError):
        extract_boxed_answer("C7")
    with pytest.raises(ValueError):
        extract_boxed_answer(r"\boxed{C7} and \boxed{C8}")
    with pytest.raises(ValueError):
        extract_boxed_answer(r"\boxed{C9}", {"C1", "C2"})


def test_prompt_preserves_question_and_requires_exact_output() -> None:
    question = "Question with tables and options"
    messages = zero_shot_messages(question)
    assert messages[-1]["content"].startswith(question)
    assert r"\boxed{2}" in messages[-1]["content"]
    assert "exactly one" in messages[0]["content"]
