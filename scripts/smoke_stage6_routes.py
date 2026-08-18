"""One bounded Qwen smoke inference for each non-standard route."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.pipeline import LazyQwenChoiceBackend, UnifiedHybrid, load_c13_resolver  # noqa: E402
from telecom_rca.routing import Route, route_question  # noqa: E402


def main() -> None:
    resolver = load_c13_resolver(ROOT / "outputs" / "models" / "c13_resolver.joblib")
    backend = LazyQwenChoiceBackend(ROOT / "models" / "Qwen2.5-1.5B-Instruct")
    pipeline = UnifiedHybrid(resolver, backend)
    test = load_current_csv("test.csv")

    selected = {}
    for question in test["question"].astype(str):
        route = route_question(question).route
        if route in {Route.MARKDOWN_TELECOM, Route.GENERAL_KNOWLEDGE} and route not in selected:
            selected[route] = question
    if set(selected) != {Route.MARKDOWN_TELECOM, Route.GENERAL_KNOWLEDGE}:
        raise RuntimeError("Could not locate both non-standard route examples")

    for route in (Route.GENERAL_KNOWLEDGE, Route.MARKDOWN_TELECOM):
        prediction = pipeline.predict(selected[route])
        print(f"route={prediction.route.value} answer={prediction.boxed_text}", flush=True)


if __name__ == "__main__":
    main()

