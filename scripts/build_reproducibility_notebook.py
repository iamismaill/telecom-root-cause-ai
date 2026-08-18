"""Build the review-friendly Candidate A notebook from versioned cell sources."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.14"}
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Candidate A reproducibility\n\n"
            "This notebook audits the exact Candidate A artifact submitted to Zindi. "
            "It uses only current challenge files and our independently built pipeline."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport hashlib, json, sys\nimport pandas as pd\n"
            "ROOT = Path.cwd().resolve()\n"
            "if ROOT.name == 'notebooks': ROOT = ROOT.parent\n"
            "sys.path.insert(0, str(ROOT / 'src'))\n"
            "from telecom_rca.data import load_current_csv\n"
            "from telecom_rca.routing import route_question\n"
            "from telecom_rca.submission import audit_submission\n"
            "from telecom_rca.pipeline import sha256_file\nROOT"
        ),
        nbf.v4.new_markdown_cell("## Official data census and structural routing"),
        nbf.v4.new_code_cell(
            "test = load_current_csv('test.csv')\nsample = load_current_csv('SampleSubmission.csv')\n"
            "routes = test.question.map(lambda q: route_question(q).route.value).value_counts().to_dict()\n"
            "assert test.shape == (863, 2)\nassert sample.shape == (3452, 2)\n"
            "assert routes == {'standard_telecom': 681, 'markdown_telecom': 100, 'general_knowledge': 82}\n"
            "{'test_shape': test.shape, 'sample_shape': sample.shape, 'routes': routes}"
        ),
        nbf.v4.new_markdown_cell("## Candidate A integrity and submission audit"),
        nbf.v4.new_code_cell(
            "candidate_path = ROOT / 'outputs/submissions/candidate_a_raw_markdown.csv'\n"
            "candidate = pd.read_csv(candidate_path)\n"
            "audit = audit_submission(candidate, test, sample)\n"
            "expected = '40f1d1090b8ba2219a89b6dd264e273a33fc3e8ac612601e51cd2b5afed254d4'\n"
            "assert sha256_file(candidate_path) == expected\n"
            "assert audit.rows == 3452 and audit.questions == 863 and audit.invalid_answers == 0\n"
            "audit"
        ),
        nbf.v4.new_markdown_cell("## Labelled validation evidence"),
        nbf.v4.new_code_cell(
            "report = json.loads((ROOT / 'reports/classical_hybrid.json').read_text())\n"
            "robustness = json.loads((ROOT / 'reports/stage7_robustness.json').read_text())\n"
            "{'validation_accuracy': report.get('official_validation_accuracy', report.get('hybrid_accuracy')), "
            "'validation_errors': report.get('official_validation_errors', report.get('hybrid_errors')), "
            "'robustness_variants': len(robustness.get('variants', []))}"
        ),
        nbf.v4.new_markdown_cell(
            "## Full regeneration\n\nRun the following from the repository root to regenerate "
            "all local predictions. Generation is deterministic and checkpointed.\n\n"
            "```bash\nPYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python "
            "scripts/generate_candidate_submissions.py\n```\n\n"
            "The script verifies model and resolver hashes, generates all 863 predictions, "
            "expands them to four rows each, and rejects any answer outside its offered choices."
        ),
        nbf.v4.new_markdown_cell("## Frozen release manifest"),
        nbf.v4.new_code_cell(
            "manifest = json.loads((ROOT / 'release/candidate_a_release_manifest.json').read_text())\n"
            "assert manifest['zindi_submission_id'] == 'YFKDgJ1K'\n"
            "{'release': manifest['release'], 'public_score': manifest['public_score'], "
            "'hashed_files': len(manifest['files'])}"
        ),
    ]
    notebook_path = ROOT / "notebooks/Candidate_A_Reproducibility.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, notebook_path)
    print(f"wrote={notebook_path.relative_to(ROOT)} cells={len(notebook['cells'])}")


if __name__ == "__main__":
    main()
