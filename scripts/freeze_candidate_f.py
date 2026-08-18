"""Create the immutable Candidate F provenance manifest."""

from datetime import datetime, timezone
import hashlib, json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): value.update(chunk)
    return value.hexdigest()


def main() -> None:
    files=[]
    for pattern in [
        'src/telecom_rca/*.py','tests/*.py','requirements.txt',
        'CANDIDATE_F_README.md','CANDIDATE_F_TECHNICAL_REPORT.md',
        'notebooks/Candidate_F_Reproducibility.ipynb',
        'reports/classical_hybrid.json','reports/stage7_robustness.json',
        'reports/private_style_robustness.json',
        'reports/synthetic_holdouts.json',
        'outputs/submissions/candidate_f_verified_math.csv',
        'outputs/submissions/candidate_f_manifest.json',
        'outputs/models/c13_resolver.joblib',
        'models/Qwen2.5-1.5B-Instruct/model.safetensors',
        'models/Qwen2.5-1.5B-Instruct/config.json',
        'models/Qwen2.5-1.5B-Instruct/tokenizer.json',
        'current_challenge_data/*.csv',
    ]:
        files.extend(ROOT.glob(pattern))
    files=sorted(set(files)); missing=[str(p) for p in files if not p.is_file()]
    if missing: raise FileNotFoundError(missing)
    manifest={
        'release':'candidate-f-public-0.945945945',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'zindi_submission_id':'hXj3RnvX','public_score':0.945945945,
        'files':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in files},
    }
    output=ROOT/'release/candidate_f_release_manifest.json'; output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding='utf-8')
    print(f'wrote={output.relative_to(ROOT)} files={len(files)}')


if __name__=='__main__': main()
