from pathlib import Path

import pytest

from telecom_rca.data import CURRENT_DATA_DIR, current_data_path


def test_current_data_path_is_isolated() -> None:
    path = current_data_path("train.csv")
    assert path.parent == CURRENT_DATA_DIR
    assert "Previous winner files" not in str(path)


def test_reference_or_unknown_file_is_rejected() -> None:
    with pytest.raises(ValueError):
        current_data_path("../Previous winner files/gopher_submission/train.csv")

