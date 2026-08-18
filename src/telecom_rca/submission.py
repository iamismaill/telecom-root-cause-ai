"""Strict construction and validation of Zindi submission files."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

import pandas as pd

from .options import parse_options
from .qwen import extract_boxed_answer


SAMPLE_ID = re.compile(r"^(?P<base>.+)_(?P<response>[1-4])$")


@dataclass(frozen=True)
class SubmissionAudit:
    rows: int
    questions: int
    responses_per_question: int
    unique_ids: int
    invalid_answers: int


def sample_id_map(sample: pd.DataFrame) -> dict[str, list[str]]:
    """Return the exact four sample IDs belonging to every base test ID."""
    if list(sample.columns) != ["ID", "Target"]:
        raise ValueError(f"Unexpected sample columns: {list(sample.columns)}")
    if sample["ID"].isna().any() or sample["ID"].duplicated().any():
        raise ValueError("Sample submission IDs must be non-null and unique")
    mapping: dict[str, list[tuple[int, str]]] = {}
    for sample_id in sample["ID"].astype(str):
        match = SAMPLE_ID.fullmatch(sample_id)
        if match is None:
            raise ValueError(f"Malformed sample ID: {sample_id!r}")
        mapping.setdefault(match.group("base"), []).append(
            (int(match.group("response")), sample_id)
        )
    result = {}
    for base, values in mapping.items():
        ordered = sorted(values)
        if [number for number, _ in ordered] != [1, 2, 3, 4]:
            raise ValueError(f"Expected response suffixes 1-4 for {base}: {ordered}")
        result[base] = [sample_id for _, sample_id in ordered]
    return result


def build_submission(
    test: pd.DataFrame,
    sample: pd.DataFrame,
    predictions: dict[str, str],
) -> pd.DataFrame:
    """Expand one best prediction per question to the four required responses."""
    if list(test.columns) != ["ID", "question"]:
        raise ValueError(f"Unexpected test columns: {list(test.columns)}")
    test_ids = test["ID"].astype(str).tolist()
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("Test IDs must be unique")
    mapping = sample_id_map(sample)
    if set(mapping) != set(test_ids):
        raise ValueError("Sample base IDs do not exactly match test IDs")
    if set(predictions) != set(test_ids):
        missing = sorted(set(test_ids) - set(predictions))
        extra = sorted(set(predictions) - set(test_ids))
        raise ValueError(f"Prediction ID mismatch; missing={missing[:3]}, extra={extra[:3]}")

    targets = {}
    for row in test.itertuples(index=False):
        allowed = {option.label for option in parse_options(row.question)}
        answer = extract_boxed_answer(predictions[str(row.ID)], allowed)
        boxed = rf"\boxed{{{answer}}}"
        for sample_id in mapping[str(row.ID)]:
            targets[sample_id] = boxed
    result = sample[["ID"]].copy()
    result["Target"] = result["ID"].map(targets)
    return result


def audit_submission(
    submission: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
) -> SubmissionAudit:
    """Reject any row, ID, response-count, ordering, or answer-domain error."""
    if list(submission.columns) != ["ID", "Target"]:
        raise ValueError(f"Unexpected submission columns: {list(submission.columns)}")
    if len(submission) != len(sample):
        raise ValueError(f"Expected {len(sample)} rows, found {len(submission)}")
    if submission["ID"].astype(str).tolist() != sample["ID"].astype(str).tolist():
        raise ValueError("Submission IDs or order differ from SampleSubmission.csv")
    if submission["ID"].duplicated().any() or submission["Target"].isna().any():
        raise ValueError("Submission contains duplicate IDs or null targets")

    questions = {str(row.ID): str(row.question) for row in test.itertuples(index=False)}
    counts: Counter[str] = Counter()
    invalid = 0
    for row in submission.itertuples(index=False):
        match = SAMPLE_ID.fullmatch(str(row.ID))
        if match is None or match.group("base") not in questions:
            raise ValueError(f"Unknown or malformed submission ID: {row.ID!r}")
        base = match.group("base")
        counts[base] += 1
        allowed = {option.label for option in parse_options(questions[base])}
        try:
            extract_boxed_answer(str(row.Target), allowed)
        except ValueError:
            invalid += 1
    if set(counts) != set(questions) or set(counts.values()) != {4}:
        raise ValueError("Every test question must have exactly four submission rows")
    if invalid:
        raise ValueError(f"Found {invalid} targets outside their offered choice domains")
    return SubmissionAudit(
        rows=len(submission),
        questions=len(counts),
        responses_per_question=4,
        unique_ids=int(submission["ID"].nunique()),
        invalid_answers=invalid,
    )
