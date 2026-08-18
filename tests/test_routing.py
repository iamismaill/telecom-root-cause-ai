from telecom_rca.data import load_current_csv
from telecom_rca.routing import Route, route_question


def test_official_route_counts_are_structural() -> None:
    test = load_current_csv("test.csv")
    routes = [route_question(question).route for question in test["question"].astype(str)]
    assert routes.count(Route.STANDARD_TELECOM) == 681
    assert routes.count(Route.MARKDOWN_TELECOM) == 100
    assert routes.count(Route.GENERAL_KNOWLEDGE) == 82


def test_all_validation_questions_route_standard() -> None:
    validation = load_current_csv("validation_questions.csv")
    assert all(
        route_question(question).route == Route.STANDARD_TELECOM
        for question in validation["question"].astype(str)
    )

