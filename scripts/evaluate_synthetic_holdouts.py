"""Evaluate frozen Candidate F components on label-preserving synthetic holdouts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.evaluation import validation_truth  # noqa: E402
from telecom_rca.pipeline import UnifiedHybrid, load_c13_resolver  # noqa: E402
from telecom_rca.robustness import (  # noqa: E402
    combined_stress,
    duplicate_table_data_rows,
    reverse_table_data_rows,
    rotate_table_columns,
    shuffle_and_relabel_options,
)


class RejectBackend:
    def answer(self, question: str, allowed: set[str]) -> str:
        raise AssertionError('Labelled standard holdout must not call Qwen')


def main() -> None:
    joined=validation_truth(load_current_csv('validation_questions.csv'),load_current_csv('validation_target.csv'))
    resolver=load_c13_resolver(ROOT/'outputs/models/c13_resolver.joblib','62e07383991c679878552b90f187b1948daf1d11a63675a9f06b9f4fe1a9ce26')
    pipeline=UnifiedHybrid(resolver,RejectBackend())
    baseline={row.ID:pipeline.predict(row.question).semantic_label for row in joined.itertuples(index=False)}
    transforms={
        'observation_duplication':lambda q,i:duplicate_table_data_rows(q),
        'row_reversal':lambda q,i:reverse_table_data_rows(q),
        'column_rotation':lambda q,i:rotate_table_columns(q),
        'option_shift':lambda q,i:shuffle_and_relabel_options(q,10000+i),
        'combined_format_shift':lambda q,i:combined_stress(q,20000+i),
    }
    results={}
    for name,transform in transforms.items():
        correct=agree=0; failures=[]; per_class={f'C{i}':{'correct':0,'total':0} for i in range(1,9)}
        for index,row in enumerate(joined.itertuples(index=False)):
            per_class[row.truth]['total']+=1
            try:
                prediction=pipeline.predict(transform(row.question,index)).semantic_label
                correct+=prediction==row.truth; agree+=prediction==baseline[row.ID]
                per_class[row.truth]['correct']+=prediction==row.truth
            except Exception as exc:
                failures.append(f'{row.ID}: {type(exc).__name__}: {exc}')
        results[name]={
            'accuracy':correct/len(joined),'baseline_agreement':agree/len(joined),
            'failures':len(failures),'failure_examples':failures[:5],
            'per_class_accuracy':{k:v['correct']/v['total'] for k,v in per_class.items()},
        }
        print(name,results[name],flush=True)
    baseline_accuracy=sum(baseline[row.ID]==row.truth for row in joined.itertuples(index=False))/len(joined)
    report={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'rows':len(joined),
            'baseline_accuracy':baseline_accuracy,'transformations':results,
            'training_use':'none; evaluation-only label-preserving transformations'}
    (ROOT/'reports/synthetic_holdouts.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# Synthetic labelled holdouts','',f"Baseline: **{baseline_accuracy:.4%}** on {len(joined)} labelled questions.",'',
           '| Holdout family | Accuracy | Baseline agreement | Failures |','|---|---:|---:|---:|']
    for name,value in results.items():
        lines.append(f"| `{name}` | {value['accuracy']:.4%} | {value['baseline_agreement']:.4%} | {value['failures']} |")
    lines.extend(['','All transformations are evaluation-only and were not used for training.',''])
    (ROOT/'reports/synthetic_holdouts.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__': main()
