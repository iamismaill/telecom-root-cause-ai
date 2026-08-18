# Telecom Root-Cause AI

Open-source solution for the [Cassava AI Root Cause Detective Hackathon](https://zindi.africa/competitions/cassava-ai-root-cause-detective-hackathon), developed by **Abdulhakin M. Ismail**.

Telecom Root-Cause AI analyzes structured network-performance data to identify the likely causes of service degradation and BTS/cell-level fault scenarios. The system combines telecom domain knowledge, feature engineering, classical machine learning, local LLM inference, and deterministic verification.

**4th of 67 active participants · Private leaderboard: 0.98344 · 7 submissions**

> **Research release:** This repository contains the competition solution and its reproducibility artifacts. It is not a production OSS/NMS system. Deployment on a live operator network requires operator-specific validation, integration, and security controls.

## Architecture

![Telecom Root-Cause AI architecture](docs/assets/telecom-root-cause-ai-architecture.png)

The structural router sends every question through one of three specialized paths before mapping the result back to the answer labels offered in that question.

## Results

| Result | Value |
|---|---:|
| Final rank | 4 / 67 active participants |
| Private leaderboard | 0.98344 |
| Candidate K public leaderboard | 0.965250965 |
| Candidate J local validation | 99.07% |
| Test questions | 863 |
| Submission rows | 3,452 |

Leaderboard performance measures this specific challenge dataset and should not be interpreted as expected accuracy on a live operator network.

## How it works

- **Standard telecom:** parses drive-test and engineering tables, computes radio-network features, applies physical domain rules, and uses a conservative machine-learning ensemble for ambiguous cases.
- **Markdown telecom:** handles differently formatted telecom tables through a dedicated deterministic decoder.
- **General knowledge:** uses the local `Qwen2.5-1.5B-Instruct` model, exact mathematical solvers, and a documented correction ledger for supported questions.
- **Reliability:** verifies identifiers, response multiplicity, boxed-answer syntax, choice membership, stage differences, and frozen artifact hashes.

The system examines evidence such as throughput, speed, resource-block usage, handovers, serving distance, antenna downtilt, neighboring-cell signal margins, overlap, and PCI modulo-30 conflict.

Using this evidence, it distinguishes likely causes such as weak coverage, interference, congestion, handover problems, antenna-configuration issues, missing neighbors, PCI conflicts, and transport failures. It diagnoses competition scenarios; it does not automatically repair or reconfigure live BTS equipment.

## Why open source?

This project began as my submission to the Cassava AI Root Cause Detective Hackathon. After the competition, telecom engineers and other practitioners expressed interest in understanding and experimenting with the approach.

I am publishing the solution so engineers, students, and researchers can reproduce the work, study the methodology, identify its limitations, and build on it. Contributions and independent validation—particularly across different telecom environments—are welcome.

## Quick start

Python 3.11 or 3.12 is recommended.

```bash
git clone https://github.com/iamismaill/telecom-root-cause-ai.git
cd telecom-root-cause-ai
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q
```

## Reproducibility

The submitted development chain is preserved as:

```text
Candidate A -> Candidate C -> Candidate F -> Candidate J -> Candidate K
```

Rebuild the deterministic post-inference stages and verify every artifact hash:

```bash
python scripts/reproduce_candidate_k.py
```

The command writes the final artifact to:

```text
outputs/private_candidates/candidate_k_verified_gk.csv
```

Candidate K is submission `79SyDg8w`. Its expected SHA-256 is:

```text
1f80ec2c549a55106ca390a1b1ac99796bbe28b4894c889ee4f9af633b49bb2e
```

You can also run `notebooks/Candidate_K_Final_Reproduction.ipynb` from top to bottom. Fresh initial inference requires the public [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) checkpoint; model weights are not stored in this repository.

## Responsible use and limitations

- The challenge data is structured and does not represent every live-network environment.
- The reported leaderboard score is not evidence of 98.344% accuracy on arbitrary operator networks.
- Root-cause outputs are hypotheses for engineering review, not instructions for autonomous network changes.
- Equipment vendors, KPI definitions, schemas, configurations, and operating conditions vary by operator.
- Validate the approach on historical, labeled incidents before considering operational use.
- Do not load untrusted serialized model artifacts. The included model is covered by the release hashes.

## Repository structure

```text
current_challenge_data/  Challenge data (CC BY-SA 4.0)
docs/assets/             Architecture and documentation images
notebooks/               End-to-end reproduction notebook
outputs/                 Frozen stages, manifests, and final artifact
release/                 SHA-256 release manifests
scripts/                 Training, evaluation, audit, and generation commands
src/telecom_rca/         Parsers, features, models, solvers, and pipeline
tests/                   Unit and regression tests
```

See [CANDIDATE_K_TECHNICAL_DOCUMENTATION.md](CANDIDATE_K_TECHNICAL_DOCUMENTATION.md) for architecture details, features, runtime, logging, maintenance, and additional limitations.

## Data and licensing

The challenge page permits the provided data to be used and shared under the **Creative Commons Attribution-ShareAlike 4.0 International** license. See [DATA_LICENSE.md](DATA_LICENSE.md) for attribution and terms.

Source code in this repository is released under the [MIT License](LICENSE). Dataset files remain under CC BY-SA 4.0. Third-party models and libraries retain their own licenses.

## Contributing

Corrections, tests, documentation improvements, and new research experiments are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Citation

If you use this work in research, please cite it using [CITATION.cff](CITATION.cff).
