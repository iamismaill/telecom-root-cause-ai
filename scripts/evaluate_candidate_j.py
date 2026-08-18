"""Evaluate a broad eight-cause learner; never generate a submission here."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.evaluation import validation_truth  # noqa: E402
from telecom_rca.features import extract_diagnostic_features  # noqa: E402
from telecom_rca.ml import ALL_CAUSE_FEATURES, RANDOM_SEED, all_cause_models  # noqa: E402
from telecom_rca.parser import parse_question  # noqa: E402
from telecom_rca.pipeline import UnifiedHybrid, load_c13_resolver  # noqa: E402
from telecom_rca.robustness import (  # noqa: E402
    combined_stress, drop_irrelevant_engineering_columns, normalize_pipe_spacing,
    rename_supported_columns, reverse_table_data_rows, rotate_table_columns,
    shift_dates, shuffle_and_relabel_options,
)


class RejectBackend:
    def answer(self, question: str, allowed: set[str]) -> str:
        raise AssertionError("Standard labelled evaluation must not call Qwen")


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        extract_diagnostic_features(row.question, parse_question(row.question))
        for row in frame.itertuples(index=False)
    ])


def site_group(question: str) -> str:
    """Group correlated examples by their set of engineering-site IDs."""
    parsed = parse_question(question)
    engineering = parsed.tables["engineering"].frame
    column = next((c for c in engineering.columns if re.sub(r"\W", "", c).lower() == "gnodebid"), None)
    if column is None:
        return "missing-site"
    sites = sorted(set(engineering[column].astype(str).str.strip()))
    return "|".join(sites)


def cv_scores(model, x, y, splitter, groups=None):
    acc, balanced = [], []
    for train_idx, holdout_idx in splitter.split(x, y, groups):
        fitted = clone(model).fit(x.iloc[train_idx][ALL_CAUSE_FEATURES], y.iloc[train_idx])
        prediction = fitted.predict(x.iloc[holdout_idx][ALL_CAUSE_FEATURES])
        acc.append(float(accuracy_score(y.iloc[holdout_idx], prediction)))
        balanced.append(float(balanced_accuracy_score(y.iloc[holdout_idx], prediction)))
    return {"fold_accuracy": acc, "mean_accuracy": float(np.mean(acc)),
            "std_accuracy": float(np.std(acc)), "mean_balanced_accuracy": float(np.mean(balanced))}


def main() -> None:
    train = load_current_csv("train.csv")
    x_train = feature_frame(train)
    y_train = train["answer"].astype(str)
    groups = train["question"].map(site_group)
    random_cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    grouped_cv = StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_SEED)
    evaluations = {}
    for name, model in all_cause_models().items():
        evaluations[name] = {
            "stratified_cv": cv_scores(model, x_train, y_train, random_cv),
            "site_grouped_cv": cv_scores(model, x_train, y_train, grouped_cv, groups),
        }
        print(name, evaluations[name], flush=True)
    selected_name = max(
        evaluations,
        key=lambda name: (
            evaluations[name]["site_grouped_cv"]["mean_balanced_accuracy"],
            evaluations[name]["stratified_cv"]["mean_balanced_accuracy"],
        ),
    )
    model = all_cause_models()[selected_name].fit(x_train[ALL_CAUSE_FEATURES], y_train)

    validation = validation_truth(
        load_current_csv("validation_questions.csv"), load_current_csv("validation_target.csv")
    )
    baseline_pipeline = UnifiedHybrid(
        load_c13_resolver(ROOT / "outputs/models/c13_resolver.joblib"), RejectBackend()
    )
    baseline = np.asarray([
        baseline_pipeline.predict(row.question).semantic_label
        for row in validation.itertuples(index=False)
    ])
    truth = validation["truth"].to_numpy()

    transforms = {
        "original": lambda q, i: q,
        "row_reversal": lambda q, i: reverse_table_data_rows(q),
        "column_rotation": lambda q, i: rotate_table_columns(q),
        "option_shuffle": lambda q, i: shuffle_and_relabel_options(q, 30000 + i),
        "pipe_spacing": lambda q, i: normalize_pipe_spacing(q),
        "date_shift": lambda q, i: shift_dates(q),
        "column_renaming": lambda q, i: rename_supported_columns(q),
        "irrelevant_column_removal": lambda q, i: drop_irrelevant_engineering_columns(q),
        "combined_shift": lambda q, i: combined_stress(q, 40000 + i),
    }
    shifted = {}
    original_prediction = None
    for name, transform in transforms.items():
        changed = pd.DataFrame({
            "question": [transform(q, i) for i, q in enumerate(validation["question"])]
        })
        prediction = model.predict(feature_frame(changed)[ALL_CAUSE_FEATURES])
        if original_prediction is None:
            original_prediction = prediction
        disagreements = prediction != baseline
        corrected = int(np.sum(disagreements & (prediction == truth) & (baseline != truth)))
        harmed = int(np.sum(disagreements & (prediction != truth) & (baseline == truth)))
        shifted[name] = {
            "accuracy": float(np.mean(prediction == truth)),
            "baseline_agreement": float(np.mean(prediction == baseline)),
            "model_original_agreement": float(np.mean(prediction == original_prediction)),
            "disagreements": int(np.sum(disagreements)),
            "corrected_baseline_errors": corrected,
            "harmed_baseline_correct": harmed,
        }
        print(name, shifted[name], flush=True)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(train), "site_groups": int(groups.nunique()),
        "features": ALL_CAUSE_FEATURES, "model_cv": evaluations,
        "selected_model": selected_name,
        "candidate_f_validation_accuracy": float(np.mean(baseline == truth)),
        "validation_and_shifts": shifted,
        "submission_generated": False,
    }
    path = ROOT / "reports/candidate_j_experiments.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
