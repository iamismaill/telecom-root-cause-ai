"""Build the reviewer-facing Candidate K reproduction notebook."""

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Candidate K — final reproducibility notebook\n\n"
            "Zindi submission `79SyDg8w`; public score **0.965250965**. "
            "See `CANDIDATE_K_TECHNICAL_DOCUMENTATION.md` for architecture, "
            "ETL, models, runtimes, metrics, logging, and maintenance notes."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport hashlib, json, subprocess, sys\n"
            "import pandas as pd\n"
            "ROOT=Path.cwd().resolve()\n"
            "if ROOT.name=='notebooks': ROOT=ROOT.parent\n"
            "sys.path.insert(0,str(ROOT/'src'))\nROOT"
        ),
        nbf.v4.new_markdown_cell("## Verify original competition inputs"),
        nbf.v4.new_code_cell(
            "required=['train.csv','validation_questions.csv','validation_target.csv',"
            "'test.csv','SampleSubmission.csv']\n"
            "missing=[name for name in required if not (ROOT/'current_challenge_data'/name).is_file()]\n"
            "assert not missing, missing\n"
            "{name:(ROOT/'current_challenge_data'/name).stat().st_size for name in required}"
        ),
        nbf.v4.new_markdown_cell("## Rebuild C → F → J → K and verify every frozen hash"),
        nbf.v4.new_code_cell(
            "result=subprocess.run([sys.executable,str(ROOT/'scripts/reproduce_candidate_k.py')],"
            "cwd=ROOT,text=True,capture_output=True,check=True)\n"
            "print(result.stdout)"
        ),
        nbf.v4.new_markdown_cell("## Audit the exact submitted artifact"),
        nbf.v4.new_code_cell(
            "path=ROOT/'outputs/private_candidates/candidate_k_verified_gk.csv'\n"
            "candidate=pd.read_csv(path)\n"
            "actual=hashlib.sha256(path.read_bytes()).hexdigest()\n"
            "expected='1f80ec2c549a55106ca390a1b1ac99796bbe28b4894c889ee4f9af633b49bb2e'\n"
            "assert actual==expected\n"
            "assert len(candidate)==3452 and candidate.ID.nunique()==3452\n"
            "base=candidate.ID.str.rsplit('_',n=1).str[0]\n"
            "assert base.value_counts().eq(4).all() and base.nunique()==863\n"
            "{'rows':len(candidate),'questions':base.nunique(),'responses_per_question':4,'sha256':actual}"
        ),
        nbf.v4.new_markdown_cell("## Inspect the correction proof ledger"),
        nbf.v4.new_code_cell(
            "manifest=json.loads((ROOT/'outputs/private_candidates/candidate_k_manifest.json').read_text())\n"
            "assert manifest['differing_questions']==15 and manifest['differing_rows']==60\n"
            "pd.DataFrame(manifest['changes'])[['ID','before','after','method','proof']]"
        ),
        nbf.v4.new_markdown_cell("## Final result\n\nThe exact Candidate K artifact has been reproduced and audited."),
    ]
    output = ROOT / "notebooks/Candidate_K_Final_Reproduction.ipynb"
    nbf.write(nb, output)
    print(output)


if __name__ == "__main__":
    main()
