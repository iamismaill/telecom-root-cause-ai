from telecom_rca.data import load_current_csv
from telecom_rca.pipeline import UnifiedHybrid, load_c13_resolver
from telecom_rca.resilient import ResilientHybrid
from telecom_rca.routing import Route, route_question


class FixedBackend:
    def __init__(self):
        self.calls = 0

    def answer(self, question: str, allowed: set[str]) -> str:
        self.calls += 1
        return sorted(allowed)[0]


def build_pipeline(backend):
    resolver = load_c13_resolver(
        __import__("pathlib").Path("outputs/models/c13_resolver.joblib"),
        "62e07383991c679878552b90f187b1948daf1d11a63675a9f06b9f4fe1a9ce26",
    )
    standard = UnifiedHybrid(resolver, backend)
    return ResilientHybrid(standard, backend)


def test_supported_markdown_uses_decoder_without_fallback():
    question = next(
        q for q in load_current_csv("test.csv")["question"]
        if route_question(q).route == Route.MARKDOWN_TELECOM
    )
    backend = FixedBackend()
    result = build_pipeline(backend).predict(question)
    assert not result.used_fallback
    assert result.prediction.semantic_label is not None
    assert backend.calls == 0


def test_incomplete_markdown_uses_constrained_fallback():
    question = next(
        q for q in load_current_csv("test.csv")["question"]
        if route_question(q).route == Route.MARKDOWN_TELECOM
    )
    # Remove the configuration table while retaining the nine offered choices.
    start = question.index("**Configuration Data**")
    end = question.index("**Signaling Data**")
    incomplete = question[:start] + question[end:]
    backend = FixedBackend()
    result = build_pipeline(backend).predict(incomplete)
    assert result.used_fallback
    assert result.prediction.answer in {o.label for o in route_question(incomplete).options}
    assert backend.calls == 1
