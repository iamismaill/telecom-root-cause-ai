import pandas as pd

from telecom_rca.data import load_current_csv
from telecom_rca.evaluation import classification_metrics, validation_truth


def test_official_validation_alignment() -> None:
    joined = validation_truth(
        load_current_csv("validation_questions.csv"),
        load_current_csv("validation_target.csv"),
    )
    assert len(joined) == 864
    assert joined["ID"].is_unique
    assert set(joined["truth"]) == {f"C{i}" for i in range(1, 9)}


def test_classification_metrics() -> None:
    metrics = classification_metrics(pd.Series(["C1", "C1", "C2"]), pd.Series(["C1", "C2", "C2"]))
    assert metrics.accuracy == 2 / 3
    assert metrics.per_class.loc["C1", "recall"] == 0.5

