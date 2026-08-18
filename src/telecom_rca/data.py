"""Safe accessors for official challenge data.

The helpers in this module deliberately refuse paths outside
``current_challenge_data`` so reference artifacts cannot enter the pipeline by
accident.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_DATA_DIR = (PROJECT_ROOT / "current_challenge_data").resolve()
EXPECTED_FILES = {
    "train.csv",
    "validation_questions.csv",
    "validation_target.csv",
    "test.csv",
    "SampleSubmission.csv",
}


def current_data_path(filename: str) -> Path:
    """Return a validated path to one official challenge CSV."""
    if filename not in EXPECTED_FILES:
        raise ValueError(
            f"Unsupported challenge file {filename!r}; expected one of "
            f"{sorted(EXPECTED_FILES)}"
        )
    path = (CURRENT_DATA_DIR / filename).resolve()
    if path.parent != CURRENT_DATA_DIR:
        raise ValueError(f"Path escapes current challenge directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing current challenge file: {path}")
    return path


def load_current_csv(filename: str) -> pd.DataFrame:
    """Load an official CSV without mutating it."""
    return pd.read_csv(current_data_path(filename))

