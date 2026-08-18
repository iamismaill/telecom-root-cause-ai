"""Create Candidate J's immutable provenance manifest."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SUBMISSION_HASH = "d1c121fbd23c5a4b0a8a3fe125e0d21a99aeccfdc843081cc868d0adbee2925f"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    submission = ROOT / "outputs/private_candidates/candidate_j_unanimous_all_cause.csv"
    if digest(submission) != EXPECTED_SUBMISSION_HASH:
        raise ValueError("Candidate J differs from the exact uploaded artifact")
    patterns = [
        "src/telecom_rca/*.py", "tests/*.py", "requirements.txt",
        "CANDIDATE_J_README.md", "CANDIDATE_J_TECHNICAL_REPORT.md",
        "notebooks/Candidate_J_Reproducibility.ipynb",
        "scripts/generate_candidate_j.py", "scripts/evaluate_candidate_j.py",
        "reports/candidate_j_experiments.json", "reports/candidate_j_experiments.md",
        "outputs/private_candidates/candidate_j_unanimous_all_cause.csv",
        "outputs/private_candidates/candidate_j_manifest.json",
        "outputs/submissions/candidate_f_verified_math.csv",
        "outputs/models/c13_resolver.joblib", "current_challenge_data/*.csv",
    ]
    files = []
    for pattern in patterns:
        files.extend(ROOT.glob(pattern))
    files = sorted(set(path for path in files if path.is_file()))
    manifest = {
        "release": "candidate-j-public-0.949806949",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "zindi_submission_id": "Hc4XoCmN", "public_score": 0.949806949,
        "submission_sha256": EXPECTED_SUBMISSION_HASH,
        "baseline_submission_id": "hXj3RnvX", "baseline_public_score": 0.945945945,
        "files": {
            str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": digest(path)}
            for path in files
        },
    }
    output = ROOT / "release/candidate_j_release_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote={output.relative_to(ROOT)} files={len(files)}")


if __name__ == "__main__":
    main()
