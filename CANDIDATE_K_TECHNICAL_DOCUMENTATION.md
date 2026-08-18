# Candidate K technical documentation

## 1. Overview and objective

This package reproduces Zindi submission `79SyDg8w` (Candidate K). The
submission contains 3,452 response rows for 863 test questions, with four
identical response rows per question. Its public score was **0.965250965**
(250/259 public questions).

Candidate K is the final stage of a frozen, versioned pipeline. It preserves
Candidate J for every telecom question and changes exactly 15 general-knowledge
questions using independently recorded calculations or source interpretations.

## 2. Architecture

```text
Official CSV files
       |
       v
Question parser and structural router
       |
       +--> standard telecom --> degraded-interval feature extraction
       |                         --> deterministic physical baseline
       |                         --> C1/C3 resolver + unanimous 8-class ensemble
       |
       +--> Markdown telecom --> Markdown table feature extraction/decoder
       |
       +--> general knowledge --> frozen Qwen result
                                  + exact math verification
                                  + 15 verified final corrections
       |
       v
Choice-description-to-label mapping
       |
       v
Four response rows per question --> submission audit --> Candidate K CSV
```

The version chain is:

```text
Candidate A -> Candidate C -> Candidate F -> Candidate J -> Candidate K
```

- Candidate A supplies the frozen standard, Markdown, and general-knowledge
  route outputs.
- Candidate C changes only the Markdown telecom route.
- Candidate F changes only questions supported by exact reusable math solvers.
- Candidate J applies a standard-telecom override only when Random Forest,
  Extra Trees, and Histogram Gradient Boosting unanimously agree and disagree
  with Candidate F.
- Candidate K changes exactly 15 general-knowledge questions. Each change has a
  method and concise proof in `outputs/private_candidates/candidate_k_manifest.json`.

## 3. Data and ETL

### Extract

The package uses the five original competition files in
`current_challenge_data/`:

- `train.csv`
- `validation_questions.csv`
- `validation_target.csv`
- `test.csv`
- `SampleSubmission.csv`

File hashes are recorded in `release/candidate_k_release_manifest.json`.

### Transform

1. Questions are parsed into prompt text, answer options, drive-test tables,
   and engineering-parameter tables.
2. Routing is structural: standard telecom, Markdown telecom, or
   general-knowledge.
3. Telecom features are calculated over degraded-throughput rows, including
   speed, scheduled resource blocks, handovers, serving distance, downtilt,
   neighboring signal margins, overlap, and PCI-modulo-30 conflict.
4. Standard option meanings are mapped to the option labels actually displayed
   in each question.
5. Candidate K's correction ledger is applied only to its 15 recorded IDs.

### Load

Predictions are expanded to four rows per test question following
`SampleSubmission.csv`. The audit requires:

- exactly 3,452 rows;
- 3,452 unique response IDs;
- exactly four rows for each question;
- every generated answer belongs to that question's offered choices.

## 4. Models and implementation

### C1/C3 resolver

A scikit-learn classifier resolves the difficult C1/C3 boundary using
training-derived radio features. The frozen artifact is
`outputs/models/c13_resolver.joblib` with SHA-256
`62e07383991c679878552b90f187b1948daf1d11a63675a9f06b9f4fe1a9ce26`.

### Eight-cause ensemble

Candidate J fits three independently seeded scikit-learn pipelines on all
2,400 labelled training questions:

- Random Forest;
- Extra Trees;
- Histogram Gradient Boosting.

An override is accepted only when all three predict the same semantic cause.

### General-knowledge verification

Exact reusable solvers cover supported arithmetic/algebra/calculus questions.
Candidate K adds 15 frozen corrections backed by a proof ledger. The correction
stage changes 60 rows and leaves all telecom rows identical to Candidate J.

## 5. Reproduction

Recommended environment: Python 3.11 or 3.12 on Linux/macOS, at least 16 GB RAM.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q
python scripts/reproduce_candidate_k.py
```

The final command regenerates C, F, J, and K from the frozen Candidate A
inference stage and original competition CSV files, then verifies every stage's
SHA-256. To rerun Candidate A model inference from scratch, install the official
`Qwen/Qwen2.5-1.5B-Instruct` checkpoint at
`models/Qwen2.5-1.5B-Instruct`, remove the cached stage-10 inference ledger, and
run `python scripts/generate_candidate_submissions.py` before the reproduction
command.

Expected final artifact:

```text
outputs/private_candidates/candidate_k_verified_gk.csv
SHA-256: 1f80ec2c549a55106ca390a1b1ac99796bbe28b4894c889ee4f9af633b49bb2e
```

## 6. Runtime

Approximate runtimes on an 8-core laptop with 16 GB RAM:

| Component | Expected runtime |
|---|---:|
| Environment installation | 3-10 minutes |
| Unit tests | under 2 minutes |
| C/F/K deterministic stages | under 2 minutes total |
| Candidate J feature extraction and fitting | 8-20 minutes |
| Frozen-chain reproduction | 10-25 minutes |
| Fresh Candidate A local Qwen inference | hardware-dependent, approximately 2-5 hours on CPU/MPS |

## 7. Validation and performance

- Candidate J local validation: **99.07%** in the recorded experiment.
- Candidate J public score: **0.949806949**.
- Candidate K public score: **0.965250965**.
- Candidate K public correct count: **250/259**.
- Candidate K structure audit: 3,452 rows, four responses per question, zero
  invalid answers.

The private score is intentionally not stated here because it was not included
in the verification email.

## 8. Error handling and logging

- Unsupported question structures raise an error instead of silently falling
  through to another route.
- Missing or duplicate answer choices raise an error.
- Artifact and dataset hashes detect accidental file changes.
- Every Candidate K correction records its previous answer, final answer,
  method, and proof.
- Submission auditing fails on missing IDs, duplicate IDs, invalid choices, or
  incorrect response multiplicity.

## 9. Maintenance and versioning

Candidate K is immutable. Do not update dependencies, seeds, correction IDs, or
artifacts when reproducing the competition entry. A changed final hash is a
different solution and must not be represented as submission `79SyDg8w`.

## 10. Known considerations

- Fresh Qwen inference is the slowest and most hardware-sensitive stage.
- The exact competition artifact is protected by a final SHA-256 assertion.
- The included intermediate outputs are retained for provenance and for rapid
  verification of the submitted artifact.
