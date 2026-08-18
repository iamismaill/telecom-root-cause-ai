"""Generate Candidate F with conservative programmatically verified math overrides."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.pipeline import sha256_file  # noqa: E402
from telecom_rca.qwen import extract_boxed_answer  # noqa: E402
from telecom_rca.routing import Route, route_question  # noqa: E402
from telecom_rca.submission import audit_submission, build_submission  # noqa: E402
from telecom_rca.verified_math import solve_verified_math  # noqa: E402


def main() -> None:
    test = load_current_csv("test.csv")
    sample = load_current_csv("SampleSubmission.csv")
    candidate_c = pd.read_csv(ROOT / "outputs/submissions/candidate_c_markdown_decoder.csv")
    baseline = dict(zip(candidate_c["ID"].astype(str), candidate_c["Target"].astype(str)))
    predictions: dict[str, str] = {}
    proofs = []
    for row in test.itertuples(index=False):
        base_id, question = str(row.ID), str(row.question)
        direct = baseline[f"{base_id}_1"]
        decision = route_question(question)
        verified = solve_verified_math(question) if decision.route == Route.GENERAL_KNOWLEDGE else None
        if verified is None:
            predictions[base_id] = direct
            continue
        allowed = {option.label for option in decision.options}
        if verified.label not in allowed:
            raise ValueError(f"Verified label outside offered choices for {base_id}")
        direct_answer = extract_boxed_answer(direct, allowed)
        predictions[base_id] = rf"\boxed{{{verified.label}}}"
        proofs.append(
            {
                "ID": base_id,
                "solver": verified.solver,
                "proof": verified.proof,
                "baseline_answer": direct_answer,
                "verified_answer": verified.label,
                "changed": direct_answer != verified.label,
            }
        )

    candidate_f = build_submission(test, sample, predictions)
    audit = audit_submission(candidate_f, test, sample)
    routes = [route_question(q).route for q in test["question"] for _ in range(4)]
    non_gk = [route != Route.GENERAL_KNOWLEDGE for route in routes]
    if not candidate_f.loc[non_gk, "Target"].reset_index(drop=True).equals(
        candidate_c.loc[non_gk, "Target"].reset_index(drop=True)
    ):
        raise AssertionError("Candidate F changed a telecom answer")
    output = ROOT / "outputs/submissions/candidate_f_verified_math.csv"
    candidate_f.to_csv(output, index=False)
    differences = candidate_f["Target"] != candidate_c["Target"]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "F",
        "file": str(output.relative_to(ROOT)),
        "sha256": sha256_file(output),
        "audit": audit.__dict__,
        "baseline": "Candidate C",
        "controlled_change": "reusable exact math solvers only",
        "telecom_identical_to_c": True,
        "verified_questions": len(proofs),
        "differing_rows": int(differences.sum()),
        "differing_questions": int(differences.sum() // 4),
        "proofs": proofs,
    }
    (ROOT / "outputs/submissions/candidate_f_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
