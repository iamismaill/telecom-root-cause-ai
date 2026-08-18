"""Rebuild Candidate K's deterministic post-inference chain and verify hashes."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "outputs/submissions/candidate_a_raw_markdown.csv":
        "40f1d1090b8ba2219a89b6dd264e273a33fc3e8ac612601e51cd2b5afed254d4",
    "outputs/submissions/candidate_c_markdown_decoder.csv":
        "222fecf0dca731e90506474353721bea4f7d17fb3eea0a3c2994ff0dc68e7139",
    "outputs/submissions/candidate_f_verified_math.csv":
        "f32022d7962018f0de6142e939bea79f642f31322249975ca47ce80628c0fa58",
    "outputs/private_candidates/candidate_j_unanimous_all_cause.csv":
        "d1c121fbd23c5a4b0a8a3fe125e0d21a99aeccfdc843081cc868d0adbee2925f",
    "outputs/private_candidates/candidate_k_verified_gk.csv":
        "1f80ec2c549a55106ca390a1b1ac99796bbe28b4894c889ee4f9af633b49bb2e",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(relative: str) -> None:
    path = ROOT / relative
    actual = digest(path)
    if actual != EXPECTED[relative]:
        raise RuntimeError(
            f"Hash mismatch for {relative}: expected={EXPECTED[relative]} actual={actual}"
        )
    print(f"verified {relative} {actual}", flush=True)


def run(script: str) -> None:
    print(f"running {script}", flush=True)
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)


def main() -> None:
    # Candidate A is the frozen model-inference stage. Its source and provenance
    # are included; this fast reproduction verifies its exact submitted input.
    verify("outputs/submissions/candidate_a_raw_markdown.csv")
    for script, output in (
        ("scripts/generate_candidate_c.py", "outputs/submissions/candidate_c_markdown_decoder.csv"),
        ("scripts/generate_candidate_f.py", "outputs/submissions/candidate_f_verified_math.csv"),
        ("scripts/generate_candidate_j.py", "outputs/private_candidates/candidate_j_unanimous_all_cause.csv"),
        ("scripts/generate_candidate_k.py", "outputs/private_candidates/candidate_k_verified_gk.csv"),
    ):
        run(script)
        verify(output)
    print("Candidate K reproduced exactly: submission 79SyDg8w", flush=True)


if __name__ == "__main__":
    main()
