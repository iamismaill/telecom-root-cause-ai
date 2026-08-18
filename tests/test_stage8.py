from telecom_rca.data import load_current_csv
from telecom_rca.general_knowledge import categorize_general_question
from telecom_rca.markdown_features import (
    compact_evidence_summary,
    extract_markdown_evidence,
    markdown_option_map,
)
from telecom_rca.routing import Route, route_question


def test_all_markdown_records_extract_complete_evidence() -> None:
    test = load_current_csv("test.csv")
    count = 0
    for row in test.itertuples(index=False):
        decision = route_question(row.question)
        if decision.route != Route.MARKDOWN_TELECOM:
            continue
        features = extract_markdown_evidence(decision.parsed)
        assert features["drive_rows"] > 0
        assert features["low_row_count"] > 0
        assert set(markdown_option_map(row.question)) == {
            "overlap", "inter_frequency_threshold", "capacity", "transport",
            "missing_neighbor", "weak_coverage", "intra_frequency_high",
            "intra_frequency_low", "pdcch",
        }
        assert len(compact_evidence_summary(features)) < 1200
        count += 1
    assert count == 100


def test_all_general_questions_receive_non_answering_category() -> None:
    test = load_current_csv("test.csv")
    categories = []
    for row in test.itertuples(index=False):
        decision = route_question(row.question)
        if decision.route == Route.GENERAL_KNOWLEDGE:
            categories.append(categorize_general_question(row.question))
    assert len(categories) == 82
    assert set(categories) <= {"mathematics", "physics", "chemistry", "biology", "humanities", "other"}
    assert "mathematics" in categories
