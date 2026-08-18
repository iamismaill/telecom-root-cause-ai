import math

from telecom_rca.data import load_current_csv
from telecom_rca.features import extract_diagnostic_features
from telecom_rca.options import standard_option_map
from telecom_rca.robustness import (
    anonymize_gnodeb_ids,
    combined_stress,
    duplicate_table_data_rows,
    drop_irrelevant_engineering_columns,
    normalize_pipe_spacing,
    reverse_table_data_rows,
    rename_supported_columns,
    rotate_table_columns,
    shift_dates,
    shuffle_and_relabel_options,
    swap_first_two_pipe_tables,
)
from telecom_rca.routing import Route, route_question


def _assert_same_numeric(left: dict[str, float], right: dict[str, float]) -> None:
    assert left.keys() == right.keys()
    for key in left:
        if math.isnan(left[key]) and math.isnan(right[key]):
            continue
        assert left[key] == right[key], key


def test_standard_transformations_preserve_features_and_route() -> None:
    question = load_current_csv("validation_questions.csv").iloc[0]["question"]
    original_parsed = route_question(question).parsed
    original_features = extract_diagnostic_features(question, original_parsed)
    transformations = [
        lambda q: shuffle_and_relabel_options(q, 42),
        swap_first_two_pipe_tables,
        normalize_pipe_spacing,
        shift_dates,
        anonymize_gnodeb_ids,
        rename_supported_columns,
        drop_irrelevant_engineering_columns,
        lambda q: combined_stress(q, 42),
    ]
    for transform in transformations:
        changed = transform(question)
        decision = route_question(changed)
        assert decision.route == Route.STANDARD_TELECOM
        changed_features = extract_diagnostic_features(changed, decision.parsed)
        _assert_same_numeric(original_features, changed_features)


def test_relabelled_options_keep_all_semantic_causes() -> None:
    question = load_current_csv("validation_questions.csv").iloc[0]["question"]
    changed = shuffle_and_relabel_options(question, 7)
    assert set(standard_option_map(changed)) == {f"C{i}" for i in range(1, 9)}
    assert {label for label in standard_option_map(changed).values()} == {str(i) for i in range(1, 9)}


def test_nonstandard_routes_survive_option_order_changes() -> None:
    test = load_current_csv("test.csv")
    examples = {}
    for question in test["question"].astype(str):
        route = route_question(question).route
        examples.setdefault(route, question)
    for route in (Route.MARKDOWN_TELECOM, Route.GENERAL_KNOWLEDGE):
        changed = shuffle_and_relabel_options(examples[route], 99)
        assert route_question(changed).route == route


def test_table_row_and_column_permutations_remain_parseable() -> None:
    question = next(
        q for q in load_current_csv("test.csv")["question"]
        if route_question(q).route == Route.MARKDOWN_TELECOM
    )
    for changed in (reverse_table_data_rows(question), rotate_table_columns(question)):
        assert route_question(changed).route == Route.MARKDOWN_TELECOM


def test_observation_duplication_preserves_route() -> None:
    question = load_current_csv("validation_questions.csv").iloc[0]["question"]
    assert route_question(duplicate_table_data_rows(question)).route == Route.STANDARD_TELECOM
