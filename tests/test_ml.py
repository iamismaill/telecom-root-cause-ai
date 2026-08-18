import pandas as pd

from telecom_rca.ml import C13_FEATURES, C13Resolver, candidate_models


def _row(tilt: float) -> dict[str, float]:
    return {feature: (tilt if "tilt" in feature else 0.0) for feature in C13_FEATURES}


def test_candidates_are_explicit_and_seeded() -> None:
    assert set(candidate_models()) == {
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    }


def test_resolver_can_fit_and_predict_binary_boundary() -> None:
    frame = pd.DataFrame([_row(-2), _row(-1), _row(8), _row(9)])
    labels = pd.Series(["C3", "C3", "C1", "C1"])
    resolver = C13Resolver(candidate_models()["logistic_regression"]).fit(frame, labels)
    features = {
        **_row(10),
        "low_speed_max": 30,
        "low_distance_max_km": 0.2,
        "low_rbs_mean": 200,
        "ping_pong_count": 0,
        "handover_count": 1,
        "low_mod30_conflict_fraction": 0,
        "low_close_noncolocated_fraction": 0,
    }
    assert resolver.predict_one(features).label == "C1"
