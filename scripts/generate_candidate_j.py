"""Generate Candidate J using only unanimous eight-cause model overrides."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.features import extract_diagnostic_features  # noqa: E402
from telecom_rca.ml import ALL_CAUSE_FEATURES, all_cause_models  # noqa: E402
from telecom_rca.options import map_standard_cause  # noqa: E402
from telecom_rca.parser import parse_question  # noqa: E402
from telecom_rca.pipeline import UnifiedHybrid, load_c13_resolver, sha256_file  # noqa: E402
from telecom_rca.routing import Route, route_question  # noqa: E402
from telecom_rca.submission import audit_submission  # noqa: E402


class RejectBackend:
    def answer(self, question: str, allowed: set[str]) -> str:
        raise AssertionError("Candidate J standard route must not invoke Qwen")


def features(questions: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([
        extract_diagnostic_features(question, parse_question(question))
        for question in questions.astype(str)
    ])[ALL_CAUSE_FEATURES]


def main() -> None:
    train = load_current_csv("train.csv")
    test = load_current_csv("test.csv")
    sample = load_current_csv("SampleSubmission.csv")
    candidate_f = pd.read_csv(ROOT / "outputs/submissions/candidate_f_verified_math.csv")
    output = candidate_f.copy()

    x_train = features(train["question"])
    models = []
    for name, model in all_cause_models().items():
        models.append((name, model.fit(x_train, train["answer"].astype(str))))

    baseline = UnifiedHybrid(
        load_c13_resolver(ROOT / "outputs/models/c13_resolver.joblib"), RejectBackend()
    )
    changes = []
    for row in test.itertuples(index=False):
        if route_question(row.question).route != Route.STANDARD_TELECOM:
            continue
        row_features = features(pd.Series([row.question]))
        votes = {name: str(model.predict(row_features)[0]) for name, model in models}
        if len(set(votes.values())) != 1:
            continue
        learned = next(iter(votes.values()))
        deterministic = baseline.predict(row.question)
        if learned == deterministic.semantic_label:
            continue
        displayed = map_standard_cause(row.question, learned)
        mask = output["ID"].astype(str).str.startswith(f"{row.ID}_")
        if int(mask.sum()) != 4:
            raise AssertionError(f"Expected four response rows for {row.ID}")
        before = output.loc[mask, "Target"].astype(str).unique().tolist()
        output.loc[mask, "Target"] = rf"\boxed{{{displayed}}}"
        changes.append({
            "ID": row.ID, "candidate_f_semantic": deterministic.semantic_label,
            "candidate_j_semantic": learned, "displayed_answer": displayed,
            "model_votes": votes, "candidate_f_target": before,
        })

    audit = audit_submission(output, test, sample)
    path = ROOT / "outputs/private_candidates/candidate_j_unanimous_all_cause.csv"
    output.to_csv(path, index=False)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "J", "baseline": "Candidate F",
        "controlled_change": "standard telecom only; unanimous RF/ExtraTrees/HGB override",
        "file": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
        "audit": audit.__dict__, "differing_questions": len(changes),
        "differing_rows": len(changes) * 4, "changes": changes,
        "upload_status": "uploaded",
        "zindi_submission_id": "Hc4XoCmN",
        "public_score": 0.949806949,
    }
    manifest_path = ROOT / "outputs/private_candidates/candidate_j_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
