"""Small, reproducible classical models for the ambiguous C1/C3 boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .diagnosis import Diagnosis, diagnose_high_confidence


RANDOM_SEED = 42

# Broad cause model features.  These are derived measurements only: no IDs,
# targets, option labels, parser-format flags, or copied threshold constants.
ALL_CAUSE_FEATURES = [
    "throughput_min", "throughput_mean", "low_throughput_mean",
    "speed_max", "low_speed_max", "low_speed_mean",
    "rbs_mean", "low_rbs_mean", "low_rbs_below_160_fraction",
    "serving_rsrp_mean", "low_serving_rsrp_mean", "low_serving_rsrp_min",
    "serving_sinr_mean", "low_serving_sinr_mean",
    "serving_cell_count", "handover_count", "handover_rate", "ping_pong_count",
    "low_handover_count", "low_handover_rate", "low_ping_pong_count",
    "distance_max_km", "low_distance_max_km", "low_distance_mean_km",
    "serving_total_tilt_max", "low_serving_total_tilt_max",
    "low_serving_total_tilt_mean", "low_tilt_excess_max_deg",
    "low_tilt_excess_mean_deg", "low_mechanical_tilt_max",
    "low_antenna_height_mean", "neighbor_margin_max_db",
    "low_neighbor_margin_max_db", "low_neighbor_stronger_fraction",
    "low_neighbor_throughput_gain_max",
    "low_neighbor_throughput_gain_positive_fraction",
    "mod30_conflict_fraction", "low_mod30_conflict_fraction",
    "strong_noncolocated_fraction", "low_strong_noncolocated_fraction",
    "low_close_noncolocated_fraction", "engineering_match_fraction",
]


def all_cause_models() -> dict[str, Pipeline]:
    """Conservative multiclass candidates for independent Candidate J tests."""
    return {
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(
                    n_estimators=500, max_depth=12, min_samples_leaf=2,
                    class_weight="balanced", n_jobs=1, random_state=RANDOM_SEED,
                )),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(
                    n_estimators=500, max_depth=12, min_samples_leaf=2,
                    class_weight="balanced", n_jobs=1, random_state=RANDOM_SEED,
                )),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(
                    learning_rate=0.05, max_iter=250, max_leaf_nodes=31,
                    l2_regularization=2.0, random_state=RANDOM_SEED,
                )),
            ]
        ),
    }

# Excludes IDs, labels, parser format, row counts, and explicit threshold values.
C13_FEATURES = [
    "throughput_min",
    "low_throughput_mean",
    "low_serving_rsrp_mean",
    "low_serving_rsrp_min",
    "low_serving_sinr_mean",
    "low_distance_max_km",
    "low_distance_mean_km",
    "low_serving_total_tilt_max",
    "low_serving_total_tilt_mean",
    "low_tilt_excess_max_deg",
    "low_tilt_excess_mean_deg",
    "low_mechanical_tilt_max",
    "low_antenna_height_mean",
    "low_neighbor_margin_max_db",
    "low_neighbor_stronger_fraction",
    "low_neighbor_throughput_gain_max",
    "low_neighbor_throughput_gain_positive_fraction",
    "strong_noncolocated_fraction",
    "engineering_match_fraction",
]


def candidate_models() -> dict[str, Pipeline]:
    """Return a small, explicit candidate set; this is not AutoML."""
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=150,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }


@dataclass
class C13Resolver:
    """Fitted binary resolver plus confidence-aware hybrid decision logic."""

    model: Any
    confidence_threshold: float = 0.55

    def fit(self, frame: pd.DataFrame, labels: pd.Series) -> "C13Resolver":
        self.model.fit(frame[C13_FEATURES], labels)
        return self

    def predict_one(self, features: dict[str, float]) -> Diagnosis:
        high = diagnose_high_confidence(features)
        if high is not None:
            return high
        row = pd.DataFrame([{name: features.get(name, np.nan) for name in C13_FEATURES}])
        probabilities = self.model.predict_proba(row)[0]
        classes = self.model.classes_
        best = int(np.argmax(probabilities))
        label = str(classes[best])
        confidence = float(probabilities[best])
        tier = "medium" if confidence >= self.confidence_threshold else "low"
        return Diagnosis(label, f"C1/C3 resolver probability={confidence:.4f}", tier)
