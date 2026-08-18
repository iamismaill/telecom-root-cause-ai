from telecom_rca.data import load_current_csv
from telecom_rca.markdown_features import (
    MARKDOWN_CAUSES,
    evidence_guided_messages,
    evidence_hypothesis,
    evidence_scores,
    extract_markdown_evidence,
)
from telecom_rca.routing import Route, route_question


def _first_markdown() -> tuple[str, dict[str, float]]:
    for question in load_current_csv("test.csv")["question"].astype(str):
        decision = route_question(question)
        if decision.route == Route.MARKDOWN_TELECOM:
            return question, extract_markdown_evidence(decision.parsed)
    raise AssertionError("No Markdown question found")


def test_evidence_scores_cover_exact_taxonomy() -> None:
    _, features = _first_markdown()
    scores = evidence_scores(features)
    assert set(scores) == set(MARKDOWN_CAUSES)
    cause, score, margin = evidence_hypothesis(features)
    assert cause in MARKDOWN_CAUSES
    assert score >= 0
    assert margin >= 0


def test_guided_prompt_is_compact_and_preserves_all_options() -> None:
    question, features = _first_markdown()
    messages = evidence_guided_messages(question, features)
    content = messages[-1]["content"]
    assert len(content) < 2500
    for description in MARKDOWN_CAUSES.values():
        assert description in content
    assert "exactly one boxed choice" in content
