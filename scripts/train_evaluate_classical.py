"""Training-only CV model selection followed by one official validation run."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.evaluation import classification_metrics, validation_truth  # noqa: E402
from telecom_rca.features import extract_diagnostic_features  # noqa: E402
from telecom_rca.ml import C13_FEATURES, C13Resolver, RANDOM_SEED, candidate_models  # noqa: E402
from telecom_rca.options import map_standard_cause  # noqa: E402
from telecom_rca.parser import parse_question  # noqa: E402


def feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in data.itertuples(index=False):
        parsed = parse_question(row.question)
        rows.append({"ID": row.ID, **extract_diagnostic_features(row.question, parsed)})
    return pd.DataFrame(rows)


def cross_validate_candidates(frame: pd.DataFrame, labels: pd.Series) -> dict[str, dict[str, object]]:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    results: dict[str, dict[str, object]] = {}
    for name, candidate in candidate_models().items():
        fold_accuracy: list[float] = []
        fold_balanced: list[float] = []
        for train_index, test_index in folds.split(frame, labels):
            model = clone(candidate)
            model.fit(frame.iloc[train_index][C13_FEATURES], labels.iloc[train_index])
            prediction = model.predict(frame.iloc[test_index][C13_FEATURES])
            fold_accuracy.append(float(accuracy_score(labels.iloc[test_index], prediction)))
            fold_balanced.append(float(balanced_accuracy_score(labels.iloc[test_index], prediction)))
        results[name] = {
            "fold_accuracy": fold_accuracy,
            "mean_accuracy": float(np.mean(fold_accuracy)),
            "std_accuracy": float(np.std(fold_accuracy)),
            "mean_balanced_accuracy": float(np.mean(fold_balanced)),
        }
    return results


def main() -> None:
    train = load_current_csv("train.csv")
    train_features = feature_frame(train)
    c13_mask = train["answer"].isin(["C1", "C3"])
    c13_frame = train_features.loc[c13_mask].reset_index(drop=True)
    c13_labels = train.loc[c13_mask, "answer"].reset_index(drop=True)
    cv = cross_validate_candidates(c13_frame, c13_labels)
    selected_name = max(cv, key=lambda name: (cv[name]["mean_balanced_accuracy"], cv[name]["mean_accuracy"]))
    selected_model = candidate_models()[selected_name]
    resolver = C13Resolver(selected_model).fit(c13_frame, c13_labels)

    joined = validation_truth(
        load_current_csv("validation_questions.csv"),
        load_current_csv("validation_target.csv"),
    )
    validation_features = feature_frame(joined[["ID", "question"]])
    records = []
    for row, features in zip(joined.itertuples(index=False), validation_features.to_dict(orient="records")):
        diagnosis = resolver.predict_one(features)
        displayed = map_standard_cause(row.question, diagnosis.label)
        records.append(
            {
                "ID": row.ID,
                "truth": row.truth,
                "prediction": diagnosis.label,
                "displayed": displayed,
                "tier": diagnosis.confidence_tier,
            }
        )
    predictions = pd.DataFrame(records)
    metrics = classification_metrics(predictions["truth"], predictions["prediction"])

    output_dir = ROOT / "outputs" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "resolver": resolver,
            "selected_model": selected_name,
            "features": C13_FEATURES,
            "seed": RANDOM_SEED,
        },
        output_dir / "c13_resolver.joblib",
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": RANDOM_SEED,
        "training_rows_c1_c3": len(c13_frame),
        "cv": cv,
        "selected_model": selected_name,
        "validation_accuracy": metrics.accuracy,
        "displayed_option_accuracy": float((predictions["displayed"] == predictions["truth"]).mean()),
        "per_class": metrics.per_class.reset_index().to_dict(orient="records"),
        "confusion": metrics.confusion.to_dict(orient="index"),
        "confidence_tiers": predictions["tier"].value_counts().to_dict(),
    }
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "classical_hybrid.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Stage 4 classical hybrid evaluation",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Training-only five-fold cross-validation (C1 versus C3)",
        "",
        "| Candidate | Mean accuracy | Std | Balanced accuracy |",
        "|---|---:|---:|---:|",
    ]
    for name, values in cv.items():
        lines.append(
            f"| `{name}` | {values['mean_accuracy']:.4%} | {values['std_accuracy']:.4%} | "
            f"{values['mean_balanced_accuracy']:.4%} |"
        )
    lines.extend(
        [
            "",
            f"Selected from training CV: **`{selected_name}`**",
            "",
            "## Official validation",
            "",
            f"- Overall semantic accuracy: **{metrics.accuracy:.4%}**",
            f"- Displayed-option accuracy: **{report['displayed_option_accuracy']:.4%}**",
            "",
            "| Label | Correct | Support | Recall |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, values in metrics.per_class.iterrows():
        lines.append(f"| {label} | {int(values.correct)} | {int(values.support)} | {values.recall:.4%} |")
    lines.extend(
        [
            "",
            "Model selection used training folds only. Official validation was not used for hyperparameter selection.",
            "",
        ]
    )
    output = "\n".join(lines)
    (report_dir / "classical_hybrid.md").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

