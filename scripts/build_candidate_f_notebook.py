"""Build the executable Candidate F reproducibility notebook."""

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {"display_name": "Telecom Challenge", "language": "python", "name": "python3"}
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Candidate F reproducibility\n\nAudit of submission `hXj3RnvX` "
            "(public score **0.945945945**) using only current challenge data and local artifacts."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport json, sys\nimport pandas as pd\n"
            "ROOT=Path.cwd().resolve()\nif ROOT.name=='notebooks': ROOT=ROOT.parent\n"
            "sys.path.insert(0,str(ROOT/'src'))\n"
            "from telecom_rca.data import load_current_csv\n"
            "from telecom_rca.pipeline import sha256_file\n"
            "from telecom_rca.routing import route_question\n"
            "from telecom_rca.submission import audit_submission\nROOT"
        ),
        nbf.v4.new_markdown_cell("## Data and route census"),
        nbf.v4.new_code_cell(
            "test=load_current_csv('test.csv'); sample=load_current_csv('SampleSubmission.csv')\n"
            "routes=test.question.map(lambda q:route_question(q).route.value).value_counts().to_dict()\n"
            "assert routes=={'standard_telecom':681,'markdown_telecom':100,'general_knowledge':82}\n"
            "{'test':test.shape,'sample':sample.shape,'routes':routes}"
        ),
        nbf.v4.new_markdown_cell("## Exact Candidate F audit"),
        nbf.v4.new_code_cell(
            "path=ROOT/'outputs/submissions/candidate_f_verified_math.csv'\n"
            "candidate=pd.read_csv(path); audit=audit_submission(candidate,test,sample)\n"
            "expected='f32022d7962018f0de6142e939bea79f642f31322249975ca47ce80628c0fa58'\n"
            "assert sha256_file(path)==expected\nassert audit.rows==3452 and audit.invalid_answers==0\naudit"
        ),
        nbf.v4.new_markdown_cell("## Validation, proofs, and private-style robustness"),
        nbf.v4.new_code_cell(
            "validation=json.loads((ROOT/'reports/classical_hybrid.json').read_text())\n"
            "proofs=json.loads((ROOT/'outputs/submissions/candidate_f_manifest.json').read_text())\n"
            "robust=json.loads((ROOT/'reports/private_style_robustness.json').read_text())\n"
            "assert proofs['verified_questions']==15 and proofs['differing_questions']==9\n"
            "assert all(v['semantic_agreement']==1 for v in robust['markdown_transformations'].values())\n"
            "{'validation_accuracy':validation.get('official_validation_accuracy',validation.get('hybrid_accuracy')),"
            "'verified_math_questions':proofs['verified_questions'],'markdown_stress_families':len(robust['markdown_transformations'])}"
        ),
        nbf.v4.new_markdown_cell(
            "## Full regeneration\n\nFrom the repository root run:\n\n"
            "```bash\nPYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/generate_candidate_submissions.py\n"
            ".venv/bin/python scripts/generate_candidate_c.py\n"
            ".venv/bin/python scripts/generate_candidate_f.py\n```"
        ),
        nbf.v4.new_markdown_cell("## Frozen provenance"),
        nbf.v4.new_code_cell(
            "release=json.loads((ROOT/'release/candidate_f_release_manifest.json').read_text())\n"
            "assert release['zindi_submission_id']=='hXj3RnvX'\n"
            "{'release':release['release'],'public_score':release['public_score'],'hashed_files':len(release['files'])}"
        ),
    ]
    path=ROOT/'notebooks/Candidate_F_Reproducibility.ipynb'; path.parent.mkdir(exist_ok=True)
    nbf.write(nb,path); print(f'wrote={path.relative_to(ROOT)} cells={len(nb.cells)}')


if __name__ == '__main__':
    main()
