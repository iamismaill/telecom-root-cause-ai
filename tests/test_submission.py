import pandas as pd
import pytest

from telecom_rca.submission import audit_submission, build_submission, sample_id_map


def fixtures():
    test = pd.DataFrame(
        {
            "ID": ["ID_A", "ID_B"],
            "question": ["Choose\n1: one\n2: two", "Choose\nA: alpha\nB: beta"],
        }
    )
    sample = pd.DataFrame(
        {
            "ID": [f"{base}_{number}" for base in ["ID_A", "ID_B"] for number in range(1, 5)],
            "Target": ["placeholder"] * 8,
        }
    )
    return test, sample


def test_build_and_audit_exact_four_rows():
    test, sample = fixtures()
    result = build_submission(test, sample, {"ID_A": r"\boxed{2}", "ID_B": r"\boxed{A}"})
    audit = audit_submission(result, test, sample)
    assert audit.rows == 8
    assert audit.questions == 2
    assert audit.invalid_answers == 0
    assert result["Target"].tolist() == [r"\boxed{2}"] * 4 + [r"\boxed{A}"] * 4


def test_rejects_missing_or_invalid_prediction():
    test, sample = fixtures()
    with pytest.raises(ValueError, match="Prediction ID mismatch"):
        build_submission(test, sample, {"ID_A": r"\boxed{2}"})
    with pytest.raises(ValueError, match="not in"):
        build_submission(test, sample, {"ID_A": r"\boxed{9}", "ID_B": r"\boxed{A}"})


def test_rejects_bad_sample_suffixes():
    _, sample = fixtures()
    sample.loc[3, "ID"] = "ID_A_3"
    with pytest.raises(ValueError):
        sample_id_map(sample)
