"""Create Candidate K's immutable provenance manifest."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "1f80ec2c549a55106ca390a1b1ac99796bbe28b4894c889ee4f9af633b49bb2e"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    submission = ROOT / "outputs/private_candidates/candidate_k_verified_gk.csv"
    if digest(submission) != EXPECTED:
        raise ValueError("Candidate K differs from uploaded artifact")
    patterns = [
        "src/telecom_rca/*.py", "tests/*.py", "requirements.txt",
        "CANDIDATE_K_README.md", "reports/candidate_k_verified_gk.md",
        "notebooks/Candidate_K_Reproducibility.ipynb",
        "scripts/generate_candidate_k.py", "scripts/build_candidate_k_notebook.py",
        "scripts/freeze_candidate_k.py",
        "outputs/private_candidates/candidate_k_verified_gk.csv",
        "outputs/private_candidates/candidate_k_manifest.json",
        "outputs/private_candidates/candidate_j_unanimous_all_cause.csv",
        "release/candidate_j_release_manifest.json",
        "current_challenge_data/*.csv",
    ]
    files = []
    for pattern in patterns:
        files.extend(ROOT.glob(pattern))
    files = sorted(set(path for path in files if path.is_file()))
    manifest = {
        "release": "candidate-k-public-0.965250965",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "zindi_submission_id": "79SyDg8w",
        "public_score": 0.965250965,
        "public_correct": 250,
        "public_questions": 259,
        "submission_sha256": EXPECTED,
        "baseline_submission_id": "Hc4XoCmN",
        "baseline_public_score": 0.949806949,
        "files": {
            str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": digest(path)}
            for path in files
        },
    }
    output = ROOT / "release/candidate_k_release_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote={output.relative_to(ROOT)} files={len(files)}")


if __name__ == "__main__":
    main()
