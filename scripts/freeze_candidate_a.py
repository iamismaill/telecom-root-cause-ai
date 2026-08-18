"""Create a deterministic provenance manifest for the frozen Candidate A release."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_files() -> list[Path]:
    fixed = [ROOT / "README.md", ROOT / "TECHNICAL_REPORT.md", ROOT / "requirements.txt"]
    patterns = [
        "src/telecom_rca/*.py", "scripts/*.py", "tests/*.py", "reports/*.md",
        "notebooks/*.ipynb",
    ]
    files = fixed[:]
    for pattern in patterns:
        files.extend(ROOT.glob(pattern))
    files.extend(
        ROOT / "current_challenge_data" / name
        for name in [
            "train.csv", "validation_questions.csv", "validation_target.csv",
            "test.csv", "SampleSubmission.csv",
        ]
    )
    files.extend(
        [
            ROOT / "models/Qwen2.5-1.5B-Instruct/model.safetensors",
            ROOT / "outputs/models/c13_resolver.joblib",
            ROOT / "outputs/submissions/candidate_a_raw_markdown.csv",
            ROOT / "outputs/submissions/stage10_manifest.json",
        ]
    )
    return sorted(set(files))


def main() -> None:
    files = selected_files()
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Release inputs missing: {missing}")
    manifest = {
        "release": "candidate-a-public-0.861003861",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "zindi_submission_id": "YFKDgJ1K",
        "public_score": 0.861003861,
        "files": {
            str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        },
    }
    output = ROOT / "release/candidate_a_release_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote={output.relative_to(ROOT)} files={len(files)}")


if __name__ == "__main__":
    main()
