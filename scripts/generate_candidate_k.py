"""Generate Candidate K from J using only independently verified GK corrections."""

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
from telecom_rca.routing import Route, route_question  # noqa: E402
from telecom_rca.submission import audit_submission  # noqa: E402
from telecom_rca.verified_gk import VERIFIED_GK_CORRECTIONS  # noqa: E402


def main() -> None:
    test = load_current_csv("test.csv")
    sample = load_current_csv("SampleSubmission.csv")
    baseline = pd.read_csv(
        ROOT / "outputs/private_candidates/candidate_j_unanimous_all_cause.csv"
    )
    candidate = baseline.copy()
    questions = dict(zip(test["ID"].astype(str), test["question"].astype(str)))
    changes = []
    for identifier, verified in VERIFIED_GK_CORRECTIONS.items():
        question = questions[identifier]
        decision = route_question(question)
        if decision.route != Route.GENERAL_KNOWLEDGE:
            raise AssertionError(f"Verified correction is not GK: {identifier}")
        allowed = {option.label for option in decision.options}
        if verified.label not in allowed:
            raise AssertionError(f"Verified answer is outside choices: {identifier}")
        mask = candidate["ID"].astype(str).str.startswith(f"{identifier}_")
        if int(mask.sum()) != 4:
            raise AssertionError(f"Expected four rows: {identifier}")
        before = candidate.loc[mask, "Target"].astype(str).unique().tolist()
        candidate.loc[mask, "Target"] = rf"\boxed{{{verified.label}}}"
        after = candidate.loc[mask, "Target"].astype(str).unique().tolist()
        if before == after:
            raise AssertionError(f"Correction did not change Candidate J: {identifier}")
        changes.append({
            "ID": identifier,
            "before": before,
            "after": after,
            "method": verified.method,
            "proof": verified.proof,
        })

    audit = audit_submission(candidate, test, sample)
    changed = candidate["Target"].ne(baseline["Target"])
    if int(changed.sum()) != 4 * len(VERIFIED_GK_CORRECTIONS):
        raise AssertionError("Candidate K contains unexpected differences")
    non_gk = [
        route_question(question).route != Route.GENERAL_KNOWLEDGE
        for question in test["question"].astype(str)
        for _ in range(4)
    ]
    if not candidate.loc[non_gk, "Target"].reset_index(drop=True).equals(
        baseline.loc[non_gk, "Target"].reset_index(drop=True)
    ):
        raise AssertionError("Candidate K changed a telecom answer")

    output = ROOT / "outputs/private_candidates/candidate_k_verified_gk.csv"
    candidate.to_csv(output, index=False)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "K",
        "baseline": "Candidate J",
        "controlled_change": "independently verified disputed/majority GK corrections only",
        "file": str(output.relative_to(ROOT)),
        "sha256": sha256_file(output),
        "audit": audit.__dict__,
        "differing_questions": len(changes),
        "differing_rows": int(changed.sum()),
        "telecom_identical_to_j": True,
        "changes": changes,
        "upload_status": "uploaded",
        "zindi_submission_id": "79SyDg8w",
        "public_score": 0.965250965,
    }
    manifest_path = ROOT / "outputs/private_candidates/candidate_k_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
