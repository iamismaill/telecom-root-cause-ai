import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "audit_markdown_private_risk",
    Path(__file__).resolve().parents[1] / "scripts/audit_markdown_private_risk.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
robust_margin = MODULE.robust_margin


def test_rule_specific_margin_boundaries() -> None:
    assert robust_margin("overlap", 0.25)
    assert not robust_margin("overlap", 0.24)
    assert robust_margin("transport", 300)
    assert not robust_margin("transport", 299)
