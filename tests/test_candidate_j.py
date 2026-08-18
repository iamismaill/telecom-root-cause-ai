from telecom_rca.ml import ALL_CAUSE_FEATURES, all_cause_models


def test_all_cause_features_do_not_contain_identity_or_target_leaks() -> None:
    tokens = {token for name in ALL_CAUSE_FEATURES for token in name.lower().split("_")}
    assert tokens.isdisjoint({"id", "target", "answer", "parser", "label"})


def test_all_cause_models_are_reproducible_candidates() -> None:
    assert set(all_cause_models()) == {"random_forest", "extra_trees", "hist_gradient_boosting"}
