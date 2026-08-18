from scripts.prepare_condition_sft import interleave
from telecom_rca.data import load_current_csv


def test_condition_sft_is_balanced_early_and_has_no_answer_in_user_prompt() -> None:
    full = load_current_csv("train.csv")
    sample = (
        full.groupby("answer", group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )
    rows = interleave(sample, augment=False)
    answers = [row["messages"][-1]["content"] for row in rows]
    assert len(rows) == 8
    assert len(set(answers)) == 8
    for row in rows:
        assert all(
            f"\\boxed{{C{index}}}" not in row["messages"][1]["content"]
            for index in range(1, 9)
        )
