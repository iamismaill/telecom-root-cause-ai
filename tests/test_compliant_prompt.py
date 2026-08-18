from telecom_rca.compliant_prompt import compact_model_prompt
from telecom_rca.data import load_current_csv


def test_compact_prompt_has_choices_and_evidence_without_answer() -> None:
    row = load_current_csv("train.csv").iloc[0]
    prompt = compact_model_prompt(row["question"])
    assert "C1:" in prompt
    assert "C8:" in prompt
    assert "degraded scheduled RB mean" in prompt
    assert "\\boxed" not in prompt
    assert len(prompt) < len(row["question"])
