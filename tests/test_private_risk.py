import importlib.util
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "evaluate_private_risk",
    Path(__file__).resolve().parents[1] / "scripts/evaluate_private_risk.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
simulate_private = MODULE.simulate_private


def test_private_simulation_is_reproducible_and_bounded() -> None:
    probabilities = np.full(863, 0.95)
    first = simulate_private(probabilities, iterations=100)
    second = simulate_private(probabilities, iterations=100)
    assert first == second
    assert 0 <= first["mean_accuracy"] <= 1
    assert first["p05_correct"] <= first["median_correct"] <= first["p95_correct"]
