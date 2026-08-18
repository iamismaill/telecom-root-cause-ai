"""Evaluation utilities for semantic classification and displayed choices."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    confusion: pd.DataFrame
    per_class: pd.DataFrame


def validation_truth(questions: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Validate four-target alignment and return one semantic truth per ID."""
    expanded = targets.copy()
    parts = expanded["ID"].str.rsplit("_", n=1, expand=True)
    expanded["base_id"] = parts[0]
    expanded["response_index"] = parts[1]
    if set(expanded["response_index"]) != {"1", "2", "3", "4"}:
        raise ValueError("Validation targets must contain response suffixes 1-4")
    counts = expanded.groupby("base_id").agg(rows=("Target", "size"), labels=("Target", "nunique"))
    if not ((counts["rows"] == 4) & (counts["labels"] == 1)).all():
        raise ValueError("Each validation question must have four identical targets")
    truth = expanded.groupby("base_id", sort=False)["Target"].first().rename("truth")
    result = questions[["ID", "question"]].merge(truth, left_on="ID", right_index=True, validate="one_to_one")
    if len(result) != len(questions):
        raise ValueError("Validation question/target IDs do not align")
    return result


def classification_metrics(truth: pd.Series, prediction: pd.Series) -> ClassificationMetrics:
    labels = sorted(set(truth) | set(prediction))
    confusion = pd.crosstab(truth, prediction, rownames=["truth"], colnames=["prediction"]).reindex(
        index=labels, columns=labels, fill_value=0
    )
    rows = []
    for label in labels:
        support = int((truth == label).sum())
        correct = int(((truth == label) & (prediction == label)).sum())
        rows.append({"label": label, "support": support, "correct": correct, "recall": correct / support if support else 0.0})
    return ClassificationMetrics(
        accuracy=float((truth.to_numpy() == prediction.to_numpy()).mean()),
        confusion=confusion,
        per_class=pd.DataFrame(rows).set_index("label"),
    )

