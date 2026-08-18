import re

from telecom_rca.data import load_current_csv
from telecom_rca.markdown_diagnosis import diagnose_markdown
from telecom_rca.routing import Route, route_question


def markdown_questions():
    return [
        question
        for question in load_current_csv("test.csv")["question"]
        if route_question(question).route == Route.MARKDOWN_TELECOM
    ]


def test_decoder_resolves_every_official_markdown_question_to_offered_choice():
    questions = markdown_questions()
    diagnoses = [diagnose_markdown(question) for question in questions]
    assert len(diagnoses) == 100
    assert len({diagnosis.semantic_cause for diagnosis in diagnoses}) == 9
    for question, diagnosis in zip(questions, diagnoses):
        allowed = {option.label for option in route_question(question).options}
        assert diagnosis.displayed_answer in allowed


def test_decoder_is_invariant_to_displayed_option_order():
    question = markdown_questions()[0]
    lines = question.splitlines()
    option_indexes = [i for i, line in enumerate(lines) if re.match(r"^[A-I]:\s", line)]
    option_lines = [lines[i] for i in option_indexes][::-1]
    for index, line in zip(option_indexes, option_lines):
        lines[index] = line
    changed = "\n".join(lines)
    assert diagnose_markdown(question).semantic_cause == diagnose_markdown(changed).semantic_cause
