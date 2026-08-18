"""Generate Candidate D by changing only GK inference relative to Candidate C."""

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
    repair_messages,
    resolve_boxed_choice,
    verification_messages,
)
from telecom_rca.pipeline import sha256_file  # noqa: E402
from telecom_rca.qwen import LocalQwen, extract_boxed_answer  # noqa: E402
from telecom_rca.routing import Route, route_question  # noqa: E402
from telecom_rca.submission import audit_submission, build_submission  # noqa: E402


MODEL_SHA256 = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
PROMPT_VERSION = 2


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
    baseline_targets = dict(zip(candidate_c["ID"].astype(str), candidate_c["Target"].astype(str)))
    model_dir = ROOT / "models/Qwen2.5-1.5B-Instruct"
    if sha256_file(model_dir / "model.safetensors") != MODEL_SHA256:
        raise ValueError("Local Qwen model hash mismatch")
    checkpoint_path = ROOT / "outputs/checkpoints/candidate_d_gk.json"
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.exists()
        else {}
    )
    runtime: LocalQwen | None = None
    predictions: dict[str, str] = {}
    gk_total = 0
    for row in test.itertuples(index=False):
        base_id, question = str(row.ID), str(row.question)
        decision = route_question(question)
        if decision.route != Route.GENERAL_KNOWLEDGE:
            predictions[base_id] = baseline_targets[f"{base_id}_1"]
            continue
        gk_total += 1
        allowed = {option.label for option in decision.options}
        record = checkpoint.get(base_id)
        if (
            record
            and record.get("question_sha256") == question_hash(question)
            and record.get("prompt_version") == PROMPT_VERSION
        ):
            answer = str(record["answer"])
            extract_boxed_answer(rf"\boxed{{{answer}}}", allowed)
        else:
            if runtime is None:
                runtime = LocalQwen(model_dir)
            scratch = runtime.generate_messages(deliberation_messages(question), max_new_tokens=192)
            verified = runtime.generate_messages(
                verification_messages(question, scratch.text), max_new_tokens=16
            )
            try:
                answer = resolve_boxed_choice(verified.text, question)
                repaired = False
            except ValueError:
                repair = runtime.generate_messages(
                    repair_messages(question, verified.text), max_new_tokens=16
                )
                answer = resolve_boxed_choice(repair.text, question)
                repaired = True
            direct = extract_boxed_answer(baseline_targets[f"{base_id}_1"], allowed)
            checkpoint[base_id] = {
                "question_sha256": question_hash(question),
                "prompt_version": PROMPT_VERSION,
                "category": categorize_general_question(question),
                "direct_answer": direct,
                "answer": answer,
                "changed": answer != direct,
                "scratch": scratch.text,
                "scratch_input_tokens": scratch.input_tokens,
                "scratch_output_tokens": scratch.output_tokens,
                "scratch_seconds": scratch.elapsed_seconds,
                "verification_input_tokens": verified.input_tokens,
                "verification_output_tokens": verified.output_tokens,
                "verification_seconds": verified.elapsed_seconds,
                "format_repaired": repaired,
            }
            write_atomic(checkpoint_path, checkpoint)
        predictions[base_id] = rf"\boxed{{{answer}}}"
        print(f"completed_gk={gk_total}/82 changed={sum(bool(x.get('changed')) for x in checkpoint.values())}", flush=True)

    candidate_d = build_submission(test, sample, predictions)
    audit = audit_submission(candidate_d, test, sample)
    routes_by_row = [route_question(q).route for q in test["question"] for _ in range(4)]
    non_gk = [route != Route.GENERAL_KNOWLEDGE for route in routes_by_row]
    if not candidate_d.loc[non_gk, "Target"].reset_index(drop=True).equals(
        candidate_c.loc[non_gk, "Target"].reset_index(drop=True)
    ):
        raise AssertionError("Candidate D changed a telecom answer")
    output = ROOT / "outputs/submissions/candidate_d_gk_two_pass.csv"
    candidate_d.to_csv(output, index=False)
    differences = candidate_d["Target"] != candidate_c["Target"]
    category_changes: dict[str, int] = {}
    for record in checkpoint.values():
        if record.get("changed"):
            category = str(record["category"])
            category_changes[category] = category_changes.get(category, 0) + 1
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "D",
        "file": str(output.relative_to(ROOT)),
        "sha256": sha256_file(output),
        "audit": audit.__dict__,
        "baseline": "Candidate C",
        "controlled_change": "two-pass local Qwen reasoning for GK only",
        "telecom_identical_to_c": True,
        "differing_rows": int(differences.sum()),
        "differing_questions": int(differences.sum() // 4),
        "category_changes": category_changes,
    }
    (ROOT / "outputs/submissions/candidate_d_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
