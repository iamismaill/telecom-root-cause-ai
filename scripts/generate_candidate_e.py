"""Generate Candidate E with deterministic option-text-to-label GK mapping."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.general_knowledge import categorize_general_question  # noqa: E402
from telecom_rca.gk_reasoning import (  # noqa: E402
    deliberation_messages,
    option_text_selection_messages,
    resolve_exact_option_text,
)
from telecom_rca.pipeline import sha256_file  # noqa: E402
from telecom_rca.qwen import LocalQwen, extract_boxed_answer  # noqa: E402
from telecom_rca.routing import Route, route_question  # noqa: E402
from telecom_rca.submission import audit_submission, build_submission  # noqa: E402


MODEL_SHA256 = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
PROMPT_VERSION = 1


def question_hash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    test = load_current_csv("test.csv")
    sample = load_current_csv("SampleSubmission.csv")
    candidate_c = pd.read_csv(ROOT / "outputs/submissions/candidate_c_markdown_decoder.csv")
    baseline = dict(zip(candidate_c["ID"].astype(str), candidate_c["Target"].astype(str)))
    model_dir = ROOT / "models/Qwen2.5-1.5B-Instruct"
    if sha256_file(model_dir / "model.safetensors") != MODEL_SHA256:
        raise ValueError("Local Qwen model hash mismatch")
    checkpoint_path = ROOT / "outputs/checkpoints/candidate_e_gk.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    runtime: LocalQwen | None = None
    predictions: dict[str, str] = {}
    completed = 0
    for row in test.itertuples(index=False):
        base_id, question = str(row.ID), str(row.question)
        decision = route_question(question)
        if decision.route != Route.GENERAL_KNOWLEDGE:
            predictions[base_id] = baseline[f"{base_id}_1"]
            continue
        completed += 1
        allowed = {option.label for option in decision.options}
        direct = extract_boxed_answer(baseline[f"{base_id}_1"], allowed)
        record = checkpoint.get(base_id)
        if not (
            record
            and record.get("question_sha256") == question_hash(question)
            and record.get("prompt_version") == PROMPT_VERSION
        ):
            if runtime is None:
                runtime = LocalQwen(model_dir)
            scratch = runtime.generate_messages(deliberation_messages(question), max_new_tokens=192)
            selection = runtime.generate_messages(
                option_text_selection_messages(question, scratch.text), max_new_tokens=96
            )
            try:
                answer = resolve_exact_option_text(selection.text, question)
                exact_match = True
            except ValueError:
                answer = direct
                exact_match = False
            record = {
                "question_sha256": question_hash(question),
                "prompt_version": PROMPT_VERSION,
                "category": categorize_general_question(question),
                "direct_answer": direct,
                "answer": answer,
                "changed": answer != direct,
                "exact_option_text_match": exact_match,
                "selected_text": selection.text,
                "scratch": scratch.text,
                "scratch_seconds": scratch.elapsed_seconds,
                "selection_seconds": selection.elapsed_seconds,
            }
            checkpoint[base_id] = record
            write_atomic(checkpoint_path, checkpoint)
        answer = str(record["answer"])
        extract_boxed_answer(rf"\boxed{{{answer}}}", allowed)
        predictions[base_id] = rf"\boxed{{{answer}}}"
        print(
            f"completed_gk={completed}/82 changed={sum(bool(x.get('changed')) for x in checkpoint.values())} "
            f"exact={sum(bool(x.get('exact_option_text_match')) for x in checkpoint.values())}",
            flush=True,
        )

    candidate_e = build_submission(test, sample, predictions)
    audit = audit_submission(candidate_e, test, sample)
    routes = [route_question(q).route for q in test["question"] for _ in range(4)]
    non_gk = [route != Route.GENERAL_KNOWLEDGE for route in routes]
    if not candidate_e.loc[non_gk, "Target"].reset_index(drop=True).equals(
        candidate_c.loc[non_gk, "Target"].reset_index(drop=True)
    ):
        raise AssertionError("Candidate E changed a telecom answer")
    output = ROOT / "outputs/submissions/candidate_e_gk_option_text.csv"
    candidate_e.to_csv(output, index=False)
    differences = candidate_e["Target"] != candidate_c["Target"]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "E",
        "file": str(output.relative_to(ROOT)),
        "sha256": sha256_file(output),
        "audit": audit.__dict__,
        "baseline": "Candidate C",
        "controlled_change": "GK reasoning selects exact option text; code maps text to label",
        "telecom_identical_to_c": True,
        "differing_rows": int(differences.sum()),
        "differing_questions": int(differences.sum() // 4),
        "exact_text_matches": sum(bool(x.get("exact_option_text_match")) for x in checkpoint.values()),
        "fallbacks_to_candidate_c": sum(not bool(x.get("exact_option_text_match")) for x in checkpoint.values()),
    }
    (ROOT / "outputs/submissions/candidate_e_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
