"""python -m baseline --help. Checkpoints are bound to data, code and configuration."""

import argparse
import importlib.metadata
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from .common import (
    CORPUS_REVISION,
    MODEL,
    PLANS,
    QUERY_PREFIX,
    QUESTION_REVISION,
    REPO,
    object_hash,
    public_question,
    read_json,
    safe_error,
    select_questions,
    settings,
    sha256,
    write_json,
)
from .retrieval import build_plans, pack_context


def emit(stage, **values):
    print(json.dumps({"stage": stage, **values}, ensure_ascii=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=REPO.parent)
    parser.add_argument("--output", type=Path, default=REPO / "baseline/runs/train500")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", default="heterorag-baseline-v1")
    parser.add_argument("--plans", nargs="+", choices=PLANS, default=list(PLANS))
    parser.add_argument("--context-tokens", type=int, default=6000)
    parser.add_argument("--row-k", type=int, default=8)
    parser.add_argument("--passage-k", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--budget-usd", type=float, default=5.0)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--retrieve-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry failed answers, preserving the usage ledger",
    )
    parser.add_argument(
        "--allow-paid-api",
        action="store_true",
        help="Explicitly opt into paid OpenAI generation; disabled by default",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.getLogger("neo4j").setLevel(logging.CRITICAL)
    args.workspace, args.output = args.workspace.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        from .report import report

        report(args.output)
        return
    if not args.retrieve_only and not args.allow_paid_api:
        raise ValueError(
            "Paid API generation is disabled. Use --report-only for offline scoring, "
            "--retrieve-only for existing-store retrieval, or explicitly opt in with --allow-paid-api."
        )
    if args.workers < 1 or args.budget_usd <= 0 or args.context_tokens < 512:
        raise ValueError("Invalid workers, budget or context limit")
    cfg = settings(args.workspace)
    for key in (
        "DATABASE_URL",
        "PINECONE_API_KEY",
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
    ):
        if not cfg.get(key):
            raise ValueError("Missing configuration field: " + key)
    if not args.retrieve_only and not cfg.get("OPENAI_API_KEY"):
        raise ValueError("Missing OPENAI_API_KEY or DECISIONS-API-KEY")
    question_file = args.workspace / "data/questions/train.json"
    trace_file = args.workspace / "data/questions/train.traced.json"
    lock = read_json(REPO / "postgres/sources.lock.json")
    for name in ("questions", "corpus"):
        if (
            sha256(args.workspace / "data" / (name + ".zip"))
            != lock[name]["archive_sha256"]
        ):
            raise ValueError(
                "Source archive checksum does not match the repository lock"
            )
    import zipfile

    with zipfile.ZipFile(args.workspace / "data/questions.zip") as archive:
        for path in (question_file, trace_file):
            names = [
                name for name in archive.namelist() if name.endswith("/" + path.name)
            ]
            if len(names) != 1 or read_json(path) != json.loads(archive.read(names[0])):
                raise ValueError("Question file differs from pinned source archive")
    questions = select_questions(read_json(question_file), args.count, args.seed)
    config = {
        "count": args.count,
        "seed": args.seed,
        "plans": args.plans,
        "context_tokens": args.context_tokens,
        "row_k": args.row_k,
        "passage_k": args.passage_k,
        "model": args.model,
        "reasoning_effort": "low",
        "max_output_tokens": 1024,
        "embedding_model": MODEL,
        "query_prefix": QUERY_PREFIX,
        "question_file_sha256": sha256(question_file),
        "source_lock_sha256": sha256(args.workspace / "data/sources.json"),
        "question_revision": QUESTION_REVISION,
        "corpus_revision": CORPUS_REVISION,
        "code": {p.name: sha256(p) for p in sorted((REPO / "baseline").glob("*.py"))},
    }
    manifest_path = args.output / "manifest.json"
    signature = object_hash(config)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["signature"] != signature:
            raise ValueError(
                "Run configuration/code/data changed. Use a new output directory."
            )
    else:
        manifest = {
            "signature": signature,
            "config": config,
            "created_at": datetime.now(UTC).isoformat(),
            "split": "train",
            "protocol": "provided-table; deterministic question-ID hash sample",
            "timing_protocol": "shared live retrieval materialized once; generation per plan; no standalone plan latency claim",
            "question_ids": [q["question_id"] for q in questions],
            "versions": {
                pkg: importlib.metadata.version(pkg)
                for pkg in [
                    "psycopg",
                    "neo4j",
                    "pinecone",
                    "numpy",
                    "sentence-transformers",
                    "transformers",
                    "torch",
                    "tiktoken",
                ]
            },
        }
        write_json(manifest_path, manifest)
    write_json(args.output / "questions.json", [public_question(q) for q in questions])
    # Keep labels separate from retrieval and reader requests.
    wanted = {q["question_id"] for q in questions}
    traces = {
        q["question_id"]: q.get("answer-node", [])
        for q in read_json(trace_file)
        if q["question_id"] in wanted
    }
    write_json(
        args.output / "references.json",
        {
            q["question_id"]: {
                "answer": q["answer-text"],
                "trace_candidates": traces[q["question_id"]],
            }
            for q in questions
        },
    )
    import tiktoken

    encoder = tiktoken.get_encoding("o200k_base")
    retrieval_dir = args.output / "retrieval"
    retrieval_dir.mkdir(exist_ok=True)
    missing = [
        public_question(q)
        for q in questions
        if not (retrieval_dir / (q["question_id"] + ".json")).exists()
    ]
    if missing:
        retrieve(args, cfg, manifest, missing, encoder, retrieval_dir)
    if not args.retrieve_only:
        generate(args, cfg, questions, encoder, retrieval_dir)
    from .report import report

    report(args.output)


def retrieve(args, cfg, manifest, questions, encoder, directory):
    import torch
    from sentence_transformers import SentenceTransformer

    from .audit import SourceAudit
    from .stores import Stores

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    cache = args.workspace / ".local/hf-cache"
    started = perf_counter()
    model = SentenceTransformer(MODEL, cache_folder=str(cache), device="cpu")
    model.max_seq_length = 512
    embedding_start = perf_counter()
    embeddings = model.encode(
        [QUERY_PREFIX + q["question"] for q in questions],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=False,
    )
    embed_seconds = perf_counter() - embedding_start
    emit(
        "encoder_ready",
        questions=len(questions),
        startup_seconds=perf_counter() - started,
        batched_query_embedding_seconds=embed_seconds,
    )
    stores = Stores(cfg)
    audit = SourceAudit(args.workspace / "data/corpus.zip")
    try:
        manifest["stores"] = stores.metadata
        graph_hash = object_hash(stores.graph_tables())
        if manifest.get("graph_tables_hash", graph_hash) != graph_hash:
            raise ValueError("Graph table availability changed during a resumed run")
        manifest["graph_tables_hash"] = graph_hash
        manifest["encoder_commit"] = getattr(
            model[0].auto_model.config, "_commit_hash", None
        )
        write_json(args.output / "manifest.json", manifest)
        bundles = {}
        vector_fidelity_checked = False
        for i, (question, vector) in enumerate(zip(questions, embeddings), 1):
            qstart = perf_counter()
            tid, qid = question["table_id"], question["question_id"]
            reused = tid in bundles
            if not reused:
                bundle = stores.table_bundle(tid)
                bundle["source_audit"] = audit.check(bundle)
                bundles[tid] = bundle
            if not vector_fidelity_checked and bundle["passages"]["records"]:
                import numpy as np

                probes = list(bundle["passages"]["records"].values())[:3]
                reencoded = model.encode(
                    [p["title"] + ". " + p["text"] for p in probes],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                stored = np.array([p["values"] for p in probes])
                similarity = np.sum(reencoded * stored, axis=1) / np.linalg.norm(
                    stored, axis=1
                )
                if float(similarity.min()) < 0.999:
                    raise ValueError(
                        "Local encoder does not reproduce stored Pinecone passage embeddings"
                    )
                manifest["embedding_fidelity_probe"] = {
                    "passages": len(probes),
                    "minimum_cosine": float(similarity.min()),
                }
                write_json(args.output / "manifest.json", manifest)
                vector_fidelity_checked = True
            bundle = bundles[tid]
            local_start = perf_counter()
            plans = build_plans(
                question["question"],
                bundle["sql"]["rows"],
                bundle["graph"]["rows"],
                bundle["passages"]["records"],
                vector,
                args.row_k,
                args.passage_k,
            )
            contexts = {}
            for name in args.plans:
                text, items, token_count = pack_context(
                    question["question"],
                    bundle["sql"]["title"],
                    plans[name],
                    encoder,
                    args.context_tokens,
                )
                contexts[name] = {
                    "context": text,
                    "evidence": items,
                    "tokens": token_count,
                    "candidate_items": len(plans[name]),
                    "included_items": len(items),
                }
            result = {
                "question": question,
                "graph_covered": bundle["graph"]["covered"],
                "graph_sql_topology_equal": bundle["graph_sql_topology_equal"],
                "source_audit": bundle["source_audit"],
                "contexts": contexts,
                "table_cache_hit": reused,
                "missing_passage_ids": bundle["passages"]["missing_ids"],
                "timings": {
                    "sql_seconds": bundle["sql"]["seconds"],
                    "graph_seconds": bundle["graph"]["seconds"],
                    "pinecone_fetch_seconds": bundle["passages"]["seconds"],
                    "original_bundle_wall_seconds": bundle["seconds"],
                    "local_ranking_packing_seconds": perf_counter() - local_start,
                    "observed_question_retrieval_seconds": perf_counter() - qstart,
                    "batched_embedding_amortized_seconds": embed_seconds
                    / len(questions),
                },
                "calls": {
                    "postgres_queries": 0 if reused else bundle["sql"]["queries"],
                    "neo4j_queries": 0 if reused else bundle["graph"]["queries"],
                    "pinecone_fetch_calls": 0
                    if reused
                    else bundle["passages"]["fetch_calls"],
                    "pinecone_read_units": 0
                    if reused
                    else bundle["passages"]["read_units"],
                },
            }
            write_json(directory / (qid + ".json"), result)
            if i % 25 == 0 or i == len(questions):
                emit(
                    "retrieval_progress",
                    completed=i,
                    total=len(questions),
                    graph_covered=result["graph_covered"],
                )
    finally:
        stores.close()
        audit.close()


def generate(args, cfg, questions, encoder, retrieval_dir):
    from .reader import BudgetExceeded, Reader

    reader = Reader(
        cfg["OPENAI_API_KEY"],
        args.model,
        encoder,
        args.output / "api_calls.jsonl",
        args.budget_usd,
    )
    answers = args.output / "answers"
    answers.mkdir(exist_ok=True)
    jobs = []
    for question in questions:
        qid = question["question_id"]
        retrieval = read_json(retrieval_dir / (qid + ".json"))
        for plan in args.plans:
            path = answers / f"{qid}.{plan}.json"
            if not path.exists() or (args.retry_failures and not read_json(path)["ok"]):
                jobs.append((qid, plan, retrieval["contexts"][plan], path))
    emit(
        "generation_start",
        pending=len(jobs),
        workers=args.workers,
        accounted_usd=reader.spent,
        budget_usd=args.budget_usd,
    )

    def work(job):
        qid, plan, ctx, path = job
        try:
            result = reader.generate(qid, plan, ctx["context"], ctx["evidence"])
            write_json(path, {"question_id": qid, "plan": plan, "ok": True, **result})
            return True
        except BudgetExceeded:
            raise
        except (ValueError, RuntimeError, TypeError, KeyError, OSError) as exc:
            error = safe_error(exc)
            if hasattr(exc, "reason"):
                error["reason"] = exc.reason
            write_json(
                path,
                {
                    "question_id": qid,
                    "plan": plan,
                    "ok": False,
                    "error": error,
                    "rejected_output": getattr(exc, "parsed", None),
                },
            )
            return False

    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            try:
                failed += int(not future.result())
            except BudgetExceeded:
                for f in futures:
                    f.cancel()
                emit("budget_reached", accounted_usd=reader.spent)
                break
            if i % 25 == 0 or i == len(jobs):
                emit(
                    "generation_progress",
                    completed=i,
                    total=len(jobs),
                    failures=failed,
                    accounted_usd=round(reader.spent, 6),
                )
