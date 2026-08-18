"""Build an unsubmitted private-distribution hedge relative to Candidate F."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.markdown_diagnosis import diagnose_markdown  # noqa: E402
from telecom_rca.pipeline import sha256_file  # noqa: E402
from telecom_rca.routing import Route, route_question  # noqa: E402
from telecom_rca.submission import audit_submission, build_submission  # noqa: E402


FALLBACK_SEMANTICS={'weak_coverage','overlap','transport'}


def main() -> None:
    test=load_current_csv('test.csv'); sample=load_current_csv('SampleSubmission.csv')
    candidate_f=pd.read_csv(ROOT/'outputs/submissions/candidate_f_verified_math.csv')
    baseline=dict(zip(candidate_f.ID.astype(str),candidate_f.Target.astype(str)))
    qwen_checkpoint=json.loads((ROOT/'outputs/checkpoints/stage10_predictions.json').read_text())
    predictions={}; fallback_counts={}; fallback_questions=[]
    for row in test.itertuples(index=False):
        base_id,question=str(row.ID),str(row.question)
        prediction=baseline[f'{base_id}_1']
        if route_question(question).route==Route.MARKDOWN_TELECOM:
            diagnosis=diagnose_markdown(question)
            if diagnosis.semantic_cause in FALLBACK_SEMANTICS:
                raw=str(qwen_checkpoint[f'markdown_raw:{base_id}']['answer'])
                prediction=rf'\boxed{{{raw}}}'
                fallback_counts[diagnosis.semantic_cause]=fallback_counts.get(diagnosis.semantic_cause,0)+1
                fallback_questions.append(base_id)
        predictions[base_id]=prediction
    candidate=build_submission(test,sample,predictions); audit=audit_submission(candidate,test,sample)
    output=ROOT/'outputs/private_candidates/candidate_h_rf_transport_qwen_hedge.csv'
    output.parent.mkdir(parents=True,exist_ok=True); candidate.to_csv(output,index=False)
    changed=candidate.Target!=candidate_f.Target
    manifest={
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),'candidate':'H-private-hedge',
        'status':'internal only; not submitted','file':str(output.relative_to(ROOT)),
        'sha256':sha256_file(output),'audit':audit.__dict__,'baseline':'Candidate F',
        'controlled_change':'raw local Qwen only for RF/transport Markdown diagnoses',
        'fallback_semantics':sorted(FALLBACK_SEMANTICS),'fallback_counts':fallback_counts,
        'fallback_questions':len(fallback_questions),'differing_questions':int(changed.sum()//4),
        'differing_rows':int(changed.sum()),
    }
    (ROOT/'outputs/private_candidates/candidate_h_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
