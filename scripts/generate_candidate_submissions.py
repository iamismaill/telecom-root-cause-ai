"""Generate controlled Candidate A/B submissions using local inference only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telecom_rca.data import load_current_csv  # noqa: E402
from telecom_rca.markdown_features import (  # noqa: E402
    evidence_guided_messages,
    extract_markdown_evidence,
)
from telecom_rca.pipeline import UnifiedHybrid, load_c13_resolver, sha256_file  # noqa: E402
from telecom_rca.qwen import LocalQwen, extract_boxed_answer  # noqa: E402
from telecom_rca.routing import Route, route_question  # noqa: E402
from telecom_rca.submission import audit_submission, build_submission  # noqa: E402


RESOLVER_SHA256 = "62e07383991c679878552b90f187b1948daf1d11a63675a9f06b9f4fe1a9ce26"
MODEL_SHA256 = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"


class RejectNonstandardBackend:
    def answer(self, question: str, allowed: set[str]) -> str:
        raise AssertionError("Standard-only hybrid unexpectedly called the Qwen backend")


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _load_checkpoint(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Checkpoint must be a JSON object")
    return value


def _question_hash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def _cached_answer(
    checkpoint: dict[str, dict[str, object]], key: str, question: str, allowed: set[str]
) -> str | None:
    record = checkpoint.get(key)
    if not record or record.get("question_sha256") != _question_hash(question):
        return None
    answer = str(record.get("answer", ""))
    extract_boxed_answer(rf"\boxed{{{answer}}}", allowed)
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Smoke-test only; do not write submissions")
    args = parser.parse_args()
    test = load_current_csv("test.csv")
    sample = load_current_csv("SampleSubmission.csv")
    resolver_path = ROOT / "outputs" / "models" / "c13_resolver.joblib"
    model_dir = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
    model_file = model_dir / "model.safetensors"
    if sha256_file(model_file) != MODEL_SHA256:
        raise ValueError("Local Qwen model hash mismatch")
    resolver = load_c13_resolver(resolver_path, RESOLVER_SHA256)
    standard_hybrid = UnifiedHybrid(resolver, RejectNonstandardBackend())

    decisions = {str(row.ID): route_question(row.question) for row in test.itertuples(index=False)}
    route_counts = pd.Series([decision.route.value for decision in decisions.values()]).value_counts().to_dict()
    if route_counts != {"standard_telecom": 681, "markdown_telecom": 100, "general_knowledge": 82}:
        raise ValueError(f"Unexpected route census: {route_counts}")

    checkpoint_path = ROOT / "outputs" / "checkpoints" / "stage10_predictions.json"
    checkpoint = _load_checkpoint(checkpoint_path)
    predictions_a: dict[str, str] = {}
    predictions_b: dict[str, str] = {}
    runtime: LocalQwen | None = None
    processed = 0
    selected_rows = list(test.itertuples(index=False))
    if args.limit is not None:
        selected_rows = selected_rows[: args.limit]

    for row in selected_rows:
        base_id, question = str(row.ID), str(row.question)
        decision = decisions[base_id]
        allowed = {option.label for option in decision.options}
        if decision.route == Route.STANDARD_TELECOM:
            answer = standard_hybrid.predict(question).answer
            extract_boxed_answer(rf"\boxed{{{answer}}}", allowed)
            predictions_a[base_id] = rf"\boxed{{{answer}}}"
            predictions_b[base_id] = rf"\boxed{{{answer}}}"
        else:
            if runtime is None:
                runtime = LocalQwen(model_dir)
            common_key = f"common:{base_id}"
            raw_key = f"markdown_raw:{base_id}"
            guided_key = f"markdown_guided:{base_id}"
            if decision.route == Route.GENERAL_KNOWLEDGE:
                answer = _cached_answer(checkpoint, common_key, question, allowed)
                if answer is None:
                    generated = runtime.generate(question, example_choice=sorted(allowed)[0])
                    answer = extract_boxed_answer(generated.text, allowed)
                    checkpoint[common_key] = {
                        "answer": answer,
                        "question_sha256": _question_hash(question),
                        "route": decision.route.value,
                        "input_tokens": generated.input_tokens,
                        "output_tokens": generated.output_tokens,
                        "elapsed_seconds": generated.elapsed_seconds,
                    }
                    _write_json_atomic(checkpoint_path, checkpoint)
                predictions_a[base_id] = rf"\boxed{{{answer}}}"
                predictions_b[base_id] = rf"\boxed{{{answer}}}"
            else:
                raw_answer = _cached_answer(checkpoint, raw_key, question, allowed)
                if raw_answer is None:
                    generated = runtime.generate(question, example_choice=sorted(allowed)[0])
                    raw_answer = extract_boxed_answer(generated.text, allowed)
                    checkpoint[raw_key] = {
                        "answer": raw_answer,
                        "question_sha256": _question_hash(question),
                        "route": decision.route.value,
                        "strategy": "raw_question",
                        "input_tokens": generated.input_tokens,
                        "output_tokens": generated.output_tokens,
                        "elapsed_seconds": generated.elapsed_seconds,
                    }
                    _write_json_atomic(checkpoint_path, checkpoint)
                guided_answer = _cached_answer(checkpoint, guided_key, question, allowed)
                if guided_answer is None:
                    features = extract_markdown_evidence(decision.parsed)
                    generated = runtime.generate_messages(
                        evidence_guided_messages(question, features), max_new_tokens=16
                    )
                    guided_answer = extract_boxed_answer(generated.text, allowed)
                    checkpoint[guided_key] = {
                        "answer": guided_answer,
                        "question_sha256": _question_hash(question),
                        "route": decision.route.value,
                        "strategy": "structured_evidence",
                        "input_tokens": generated.input_tokens,
                        "output_tokens": generated.output_tokens,
                        "elapsed_seconds": generated.elapsed_seconds,
                    }
                    _write_json_atomic(checkpoint_path, checkpoint)
                predictions_a[base_id] = rf"\boxed{{{raw_answer}}}"
                predictions_b[base_id] = rf"\boxed{{{guided_answer}}}"
        processed += 1
        if processed % 20 == 0 or processed == len(selected_rows):
            print(f"completed={processed}/{len(selected_rows)} route={decision.route.value}", flush=True)

    if args.limit is not None:
        print("Smoke run complete; submission files were not written.")
        return

    candidate_a = build_submission(test, sample, predictions_a)
    candidate_b = build_submission(test, sample, predictions_b)
    audit_a = audit_submission(candidate_a, test, sample)
    audit_b = audit_submission(candidate_b, test, sample)
    output_dir = ROOT / "outputs" / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    path_a = output_dir / "candidate_a_raw_markdown.csv"
    path_b = output_dir / "candidate_b_evidence_markdown.csv"
    candidate_a.to_csv(path_a, index=False)
    candidate_b.to_csv(path_b, index=False)
    if not candidate_a["ID"].equals(candidate_b["ID"]):
        raise AssertionError("Candidate IDs differ")
    differing_rows = int((candidate_a["Target"] != candidate_b["Target"]).sum())
    differing_questions = differing_rows // 4
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "route_counts": route_counts,
        "candidate_a": {"file": str(path_a.relative_to(ROOT)), "sha256": sha256_file(path_a), "audit": audit_a.__dict__},
        "candidate_b": {"file": str(path_b.relative_to(ROOT)), "sha256": sha256_file(path_b), "audit": audit_b.__dict__},
        "controlled_difference": "Markdown telecom prompt only; standard telecom and GK are identical",
        "differing_questions": differing_questions,
        "differing_rows": differing_rows,
        "resolver_sha256": sha256_file(resolver_path),
        "model_sha256": sha256_file(model_file),
    }
    _write_json_atomic(output_dir / "stage10_manifest.json", manifest)
    experiment = output_dir / "leaderboard_experiments.csv"
    pd.DataFrame(
        [
            {"candidate": "A", "file": path_a.name, "public_score": "", "submitted_at": "", "notes": "raw Markdown Qwen"},
            {"candidate": "B", "file": path_b.name, "public_score": "", "submitted_at": "", "notes": "evidence-guided Markdown Qwen"},
        ]
    ).to_csv(experiment, index=False)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
