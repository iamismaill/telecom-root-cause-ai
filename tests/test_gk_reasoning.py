import pytest

from telecom_rca.gk_reasoning import (
    deliberation_messages,
    option_text_selection_messages,
    repair_messages,
    resolve_boxed_choice,
    resolve_exact_option_text,
    verification_messages,
)


QUESTION = "What is 2 + 2?\n1: 3\n2: 4\n3: 5\n4: 6"


def test_two_pass_prompts_preserve_question_and_choice_domain():
    first = deliberation_messages(QUESTION)
    second = verification_messages(QUESTION, "2 + 2 equals 4, so choice 2.")
    assert QUESTION in first[-1]["content"]
    assert QUESTION in second[-1]["content"]
    assert "Allowed labels: 1, 2, 3, 4" in second[-1]["content"]
    assert r"\boxed{CHOICE}" in second[-1]["content"]
    repair = repair_messages(QUESTION, "I choose 2")
    assert QUESTION in repair[-1]["content"]
    assert r"\boxed{CHOICE}" in repair[-1]["content"]


def test_resolves_label_or_exact_boxed_option_value():
    assert resolve_boxed_choice(r"\boxed{2}", QUESTION) == "2"
    numeric = "Choose the result.\n1: 41\n2: 22140\n3: 19270\n4: 2870"
    assert resolve_boxed_choice(r"\boxed{22140}", numeric) == "2"
    with pytest.raises(ValueError):
        resolve_boxed_choice(r"\boxed{four}", QUESTION)


def test_exact_option_text_is_mapped_by_code_not_model_label():
    assert resolve_exact_option_text("4", QUESTION) == "2"
    assert resolve_exact_option_text('"4"', QUESTION) == "2"
    assert resolve_exact_option_text("FINAL OPTION TEXT: 4", QUESTION) == "2"
    with pytest.raises(ValueError):
        resolve_exact_option_text("choice 2", QUESTION)
    messages = option_text_selection_messages(QUESTION, "2 + 2 = 4")
    assert "Do not output its label" in messages[0]["content"]
