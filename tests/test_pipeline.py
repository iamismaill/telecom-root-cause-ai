from telecom_rca.data import load_current_csv
from telecom_rca.pipeline import UnifiedHybrid, load_c13_resolver
from telecom_rca.routing import Route, route_question


class FakeChoiceBackend:
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, question: str, allowed: set[str]) -> str:
        self.calls += 1
        return sorted(allowed)[0]


def test_standard_route_does_not_call_qwen_backend() -> None:
    resolver = load_c13_resolver(
        __import__("pathlib").Path("outputs/models/c13_resolver.joblib").resolve()
    )
    backend = FakeChoiceBackend()
    pipeline = UnifiedHybrid(resolver, backend)
    question = load_current_csv("validation_questions.csv").iloc[0]["question"]
    prediction = pipeline.predict(question)
    assert prediction.route == Route.STANDARD_TELECOM
    assert prediction.answer.startswith("C")
    assert backend.calls == 0


def test_nonstandard_routes_use_choice_backend_and_valid_domain() -> None:
    resolver = load_c13_resolver(
        __import__("pathlib").Path("outputs/models/c13_resolver.joblib").resolve()
    )
    backend = FakeChoiceBackend()
    pipeline = UnifiedHybrid(resolver, backend)
    test = load_current_csv("test.csv")
    selected = {}
    for question in test["question"].astype(str):
        route = route_question(question).route
        selected.setdefault(route, question)
    for route in (Route.MARKDOWN_TELECOM, Route.GENERAL_KNOWLEDGE):
        prediction = pipeline.predict(selected[route])
        allowed = {option.label for option in route_question(selected[route]).options}
        assert prediction.answer in allowed
    assert backend.calls == 2
