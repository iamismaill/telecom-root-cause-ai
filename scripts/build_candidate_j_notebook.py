"""Build Candidate J's executable audit notebook."""

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Telecom Challenge", "language": "python", "name": "telecom-venv"
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Candidate J reproducibility\n\nAudit of Zindi submission `Hc4XoCmN` "
            "with public score **0.949806949**."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport json, sys\nimport pandas as pd\n"
            "ROOT=Path.cwd().resolve()\nif ROOT.name=='notebooks': ROOT=ROOT.parent\n"
            "sys.path.insert(0,str(ROOT/'src'))\n"
            "from telecom_rca.data import load_current_csv\n"
            "from telecom_rca.submission import audit_submission\nROOT"
        ),
        nbf.v4.new_markdown_cell("## Exact uploaded artifact"),
        nbf.v4.new_code_cell(
            "test=load_current_csv('test.csv'); sample=load_current_csv('SampleSubmission.csv')\n"
            "path=ROOT/'outputs/private_candidates/candidate_j_unanimous_all_cause.csv'\n"
            "candidate=pd.read_csv(path); audit=audit_submission(candidate,test,sample)\n"
            "import hashlib\n"
            "actual=hashlib.sha256(path.read_bytes()).hexdigest()\n"
            "expected='d1c121fbd23c5a4b0a8a3fe125e0d21a99aeccfdc843081cc868d0adbee2925f'\n"
            "assert actual==expected\n"
            "assert audit.rows==3452 and audit.invalid_answers==0\naudit"
        ),
        nbf.v4.new_markdown_cell("## Controlled difference from Candidate F"),
        nbf.v4.new_code_cell(
            "baseline=pd.read_csv(ROOT/'outputs/submissions/candidate_f_verified_math.csv')\n"
            "changed=candidate.Target.ne(baseline.Target)\n"
            "assert changed.sum()==32\n"
            "assert candidate.loc[changed,'ID'].str.rsplit('_',n=1).str[0].nunique()==8\n"
            "{'changed_rows':int(changed.sum()),'changed_questions':8}"
        ),
        nbf.v4.new_markdown_cell("## Independent evaluation evidence"),
        nbf.v4.new_code_cell(
            "report=json.loads((ROOT/'reports/candidate_j_experiments.json').read_text())\n"
            "original=report['validation_and_shifts']['original']\n"
            "assert report['selected_model']=='hist_gradient_boosting'\n"
            "assert original['accuracy']==0.9895833333333334\n"
            "assert all(v['model_original_agreement']==1 for v in report['validation_and_shifts'].values())\n"
            "{'broad_model_accuracy':original['accuracy'],'format_shift_families':len(report['validation_and_shifts'])}"
        ),
        nbf.v4.new_markdown_cell(
            "## Full regeneration\n\nRun:\n\n"
            "```bash\nLOKY_MAX_CPU_COUNT=8 .venv/bin/python scripts/generate_candidate_j.py\n```"
        ),
        nbf.v4.new_markdown_cell("## Frozen provenance"),
        nbf.v4.new_code_cell(
            "release=json.loads((ROOT/'release/candidate_j_release_manifest.json').read_text())\n"
            "assert release['zindi_submission_id']=='Hc4XoCmN'\n"
            "assert release['public_score']==0.949806949\n"
            "{'release':release['release'],'public_score':release['public_score'],'hashed_files':len(release['files'])}"
        ),
    ]
    path = ROOT / "notebooks/Candidate_J_Reproducibility.ipynb"
    nbf.write(nb, path)
    print(f"wrote={path.relative_to(ROOT)} cells={len(nb.cells)}")


if __name__ == "__main__":
    main()
