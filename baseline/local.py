"""Replay saved database evidence through an installed local Ollama model, with no cloud calls."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from .answers import AnswerValidationError
from .common import REPO, object_hash, read_json, safe_error, sha256, write_json
from .local_reader import LocalReader
from .report import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Existing run containing retrieved contexts",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument(
        "--count", type=int, help="Optional limit within the available cached questions"
    )
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output:
        raise ValueError(
            "Use a separate output directory to preserve the original experiment"
        )
    source_manifest = read_json(source / "manifest.json")
    source_questions = read_json(source / "questions.json")
    available = [
        q
        for q in source_questions
        if (source / "retrieval" / (q["question_id"] + ".json")).exists()
    ]
    if args.count is not None:
        if not 1 <= args.count <= len(available):
            raise ValueError(
                f"Only {len(available)} questions have saved database evidence"
            )
        available = available[: args.count]
    if not available:
        raise ValueError("No saved database evidence to replay")
    reader = LocalReader(args.model)
    config = {
        **source_manifest["config"],
        "count": len(available),
        "model": args.model,
        "reader": "local_ollama",
        "model_digest": reader.metadata["digest"],
        "temperature": 0,
        "seed_local_generation": 42,
        "max_output_tokens": 256,
        "local_context_window": 16384,
        "reasoning_effort": None,
        "thinking": False,
        "ollama_version": reader.runtime["version"],
        "source_run_signature": source_manifest["signature"],
        "retrieval_sha256": {
            q["question_id"]: sha256(
                source / "retrieval" / (q["question_id"] + ".json")
            )
            for q in available
        },
        "code": {p.name: sha256(p) for p in sorted((REPO / "baseline").glob("*.py"))},
    }
    signature = object_hash(config)
    if (output / "manifest.json").exists() and read_json(output / "manifest.json")[
        "signature"
    ] != signature:
        raise ValueError(
            "Local model, code or saved evidence changed; use a new output directory"
        )
    manifest = {
        "signature": signature,
        "config": config,
        "split": "train",
        "created_at": datetime.now(UTC).isoformat(),
        "local_model": reader.metadata,
        "source_requested_questions": len(source_questions),
        "available_cached_questions": sum(
            (source / "retrieval" / (q["question_id"] + ".json")).exists()
            for q in source_questions
        ),
        "selected_questions": len(available),
        "protocol": "provided-table; replay of saved live-database contexts; no new database or cloud API calls",
        "limitations": [
            "Only questions with saved retrieval are evaluated; the interrupted 500-question run is not complete.",
            "Local inference uses quantized models with thinking disabled and greedy decoding.",
            "Context budget is measured by the original o200k tokenizer; local prompt tokens are recorded separately.",
            "Retrieval timings describe the original source run; replay makes no physical database calls.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "source-manifest.json", source_manifest)
    write_json(output / "questions.json", available)
    refs = read_json(source / "references.json")
    write_json(
        output / "references.json",
        {q["question_id"]: refs[q["question_id"]] for q in available},
    )
    done = failed = 0
    print(
        json.dumps(
            {
                "stage": "local_replay",
                "questions": len(available),
                "model": args.model,
                "paid_api_calls": 0,
                "new_database_calls": 0,
            }
        ),
        flush=True,
    )
    for question in available:
        qid = question["question_id"]
        retrieval = read_json(source / "retrieval" / (qid + ".json"))
        retrieval["source_calls"] = retrieval["calls"]
        retrieval["calls"] = {key: 0 for key in retrieval["calls"]}
        retrieval["replayed_from_saved_evidence"] = True
        write_json(output / "retrieval" / (qid + ".json"), retrieval)
        for plan in config["plans"]:
            path = output / "answers" / f"{qid}.{plan}.json"
            if path.exists():
                continue
            context = retrieval["contexts"][plan]
            try:
                result = reader.generate(context["context"], context["evidence"])
                answer = {"question_id": qid, "plan": plan, "ok": True, **result}
            except (AnswerValidationError, json.JSONDecodeError) as exc:
                answer = {
                    "question_id": qid,
                    "plan": plan,
                    "ok": False,
                    "error": safe_error(exc),
                    "reason": getattr(exc, "reason", None),
                    "rejected_output": getattr(exc, "parsed", None),
                    **getattr(exc, "metadata", {}),
                }
                failed += 1
            write_json(path, answer)
            done += 1
            if done % 8 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "local_generation",
                            "completed": done,
                            "failures": failed,
                        }
                    ),
                    flush=True,
                )
    report(output)


if __name__ == "__main__":
    main()
