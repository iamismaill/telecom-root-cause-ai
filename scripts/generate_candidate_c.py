"""Generate Candidate C by changing only the Markdown telecom component."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.markdown_diagnosis import diagnose_markdown  # noqa: E402
from telecom_rca.pipeline import sha256_file  # noqa: E402
from telecom_rca.routing import Route, route_question  # noqa: E402
from telecom_rca.submission import audit_submission, build_submission  # noqa: E402


def main() -> None:
    test = load_current_csv("test.csv")
    sample = load_current_csv("SampleSubmission.csv")
    candidate_a = pd.read_csv(ROOT / "outputs/submissions/candidate_a_raw_markdown.csv")
    checkpoint = json.loads(
        (ROOT / "outputs/checkpoints/stage10_predictions.json").read_text(encoding="utf-8")
    )
    sample_targets = dict(zip(candidate_a["ID"].astype(str), candidate_a["Target"].astype(str)))
    predictions: dict[str, str] = {}
    semantic_counts: dict[str, int] = {}
    for row in test.itertuples(index=False):
        base_id, question = str(row.ID), str(row.question)
        decision = route_question(question)
        if decision.route == Route.MARKDOWN_TELECOM:
            diagnosis = diagnose_markdown(question)
            predictions[base_id] = rf"\boxed{{{diagnosis.displayed_answer}}}"
            semantic_counts[diagnosis.semantic_cause] = semantic_counts.get(diagnosis.semantic_cause, 0) + 1
        else:
            # Reuse the exact frozen Candidate A answer for response 1.
            predictions[base_id] = sample_targets[f"{base_id}_1"]

    candidate_c = build_submission(test, sample, predictions)
    audit = audit_submission(candidate_c, test, sample)
    if not candidate_c.loc[
        [route_question(q).route != Route.MARKDOWN_TELECOM for q in test["question"] for _ in range(4)],
        "Target",
    ].reset_index(drop=True).equals(
        candidate_a.loc[
            [route_question(q).route != Route.MARKDOWN_TELECOM for q in test["question"] for _ in range(4)],
            "Target",
        ].reset_index(drop=True)
    ):
        raise AssertionError("Candidate C changed a standard telecom or GK answer")
    output = ROOT / "outputs/submissions/candidate_c_markdown_decoder.csv"
    candidate_c.to_csv(output, index=False)
    differences = candidate_c["Target"] != candidate_a["Target"]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "C",
        "file": str(output.relative_to(ROOT)),
        "sha256": sha256_file(output),
        "audit": audit.__dict__,
        "baseline": "Candidate A",
        "controlled_change": "deterministic Markdown anomaly decoder only",
        "standard_and_gk_identical_to_a": True,
        "differing_rows": int(differences.sum()),
        "differing_questions": int(differences.sum() // 4),
        "markdown_semantic_counts": semantic_counts,
    }
    manifest_path = ROOT / "outputs/submissions/candidate_c_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
