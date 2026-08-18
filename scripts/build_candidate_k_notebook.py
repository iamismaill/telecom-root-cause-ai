"""Build Candidate K's executable audit notebook."""

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
            "# Candidate K reproducibility\n\nSubmission `79SyDg8w`, public score **0.965250965**."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport hashlib, json, sys\nimport pandas as pd\n"
            "ROOT=Path.cwd().resolve()\nif ROOT.name=='notebooks': ROOT=ROOT.parent\n"
            "sys.path.insert(0,str(ROOT/'src'))\nROOT"
        ),
        nbf.v4.new_markdown_cell("## Exact artifact"),
        nbf.v4.new_code_cell(
            "path=ROOT/'outputs/private_candidates/candidate_k_verified_gk.csv'\n"
            "candidate=pd.read_csv(path)\n"
            "actual=hashlib.sha256(path.read_bytes()).hexdigest()\n"
            "expected='1f80ec2c549a55106ca390a1b1ac99796bbe28b4894c889ee4f9af633b49bb2e'\n"
            "assert actual==expected\n"
            "assert len(candidate)==3452 and candidate.ID.nunique()==3452\n"
            "{'rows':len(candidate),'sha256':actual}"
        ),
        nbf.v4.new_markdown_cell("## Controlled difference from Candidate J"),
        nbf.v4.new_code_cell(
            "baseline=pd.read_csv(ROOT/'outputs/private_candidates/candidate_j_unanimous_all_cause.csv')\n"
            "changed=candidate.Target.ne(baseline.Target)\n"
            "assert changed.sum()==60\n"
            "assert candidate.loc[changed,'ID'].str.rsplit('_',n=1).str[0].nunique()==15\n"
            "{'changed_rows':int(changed.sum()),'changed_questions':15}"
        ),
        nbf.v4.new_markdown_cell("## Proof ledger"),
        nbf.v4.new_code_cell(
            "manifest=json.loads((ROOT/'outputs/private_candidates/candidate_k_manifest.json').read_text())\n"
            "assert manifest['differing_questions']==15\n"
            "assert all(item['proof'] and item['method'] for item in manifest['changes'])\n"
            "{'proofs':len(manifest['changes']),'telecom_identical_to_j':manifest['telecom_identical_to_j']}"
        ),
        nbf.v4.new_markdown_cell("## Frozen provenance"),
        nbf.v4.new_code_cell(
            "release=json.loads((ROOT/'release/candidate_k_release_manifest.json').read_text())\n"
            "assert release['zindi_submission_id']=='79SyDg8w'\n"
            "assert release['public_score']==0.965250965\n"
            "{'release':release['release'],'hashed_files':len(release['files'])}"
        ),
    ]
    path = ROOT / "notebooks/Candidate_K_Reproducibility.ipynb"
    nbf.write(nb, path)
    print(f"wrote={path.relative_to(ROOT)} cells={len(nb.cells)}")


if __name__ == "__main__":
    main()
