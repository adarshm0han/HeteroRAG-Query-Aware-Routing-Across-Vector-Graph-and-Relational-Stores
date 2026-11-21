"""Freeze and run a bounded fresh routed/all-store development comparison."""

import argparse
import json
import logging
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from baseline.common import (
    REPO,
    object_hash,
    public_question,
    read_json,
    select_questions,
    sha256,
    write_json,
)
from evaluation.cohorts import code_hashes, load_bundle
from routing.pipeline import Config, Pipeline

SEED = "fresh-development-pair-50-v1"


def select_fresh(questions, excluded, n=50):
    if type(n) is not int or not 1 <= n <= 50:
        raise ValueError("Question count must be an integer from 1 to 50")
    pool = [public_question(q) for q in questions if q["question_id"] not in excluded]
    if len(pool) < n or len({q["question_id"] for q in pool}) != len(pool):
        raise ValueError("Insufficient unique fresh development questions")
    return select_questions(pool, n, SEED)


def schedule(n):
    # Same question pair, balanced first condition, one sequential GPU worker.
    return [
        {"index": i, "condition": c}
        for i in range(n)
        for c in (("all", "routed") if i % 2 == 0 else ("routed", "all"))
    ]


def prepare(bundle, source, n=50):
    if bundle.exists():
        raise ValueError("Use a new immutable bundle directory")
    protocol = load_bundle(source, check_code=False)
    prior = sorted((REPO / "baseline/runs").glob("*/questions.json")) + sorted(
        (REPO / "routing/runs").rglob("questions.json")
    )
    excluded = {"00009b9649d0dd0a", *protocol["exposed_dev_exclusions"]}
    prior_hashes = {}
    for path in prior:
        excluded.update(q["question_id"] for q in read_json(path))
        prior_hashes[str(path)] = sha256(path)
    excluded.update(
        q["question_id"] for q in read_json(source / "smoke/questions.json")
    )
    questions = select_fresh(
        read_json(source / "development/questions.json"), excluded, n
    )
    # Preparation selects by IDs, before opening answer references or coverage.
    references = read_json(source / "development/references.json")
    write_json(bundle / "questions.json", questions)
    write_json(
        bundle / "references.json",
        {q["question_id"]: references[q["question_id"]] for q in questions},
    )
    write_json(bundle / "schedule.json", schedule(len(questions)))
    names = ["questions.json", "references.json", "schedule.json"]
    if protocol.get("routing_catalog"):
        shutil.copyfile(
            source / "routing-catalog.json", bundle / "routing-catalog.json"
        )
        names.append("routing-catalog.json")
    plan = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "type": "fresh_development_paired_live_retrieval",
        "source_bundle": str(source.resolve()),
        "source_bundle_id": protocol["bundle_id"],
        "questions": len(questions),
        "conditions": ["routed", "all"],
        "max_decisions_requests": len(questions),
        "config": asdict(Config()),
        "reader_digest": protocol["reader_digest"],
        "code": code_hashes(),
        "routing_catalog": protocol.get("routing_catalog"),
        "prior_question_sources": prior_hashes,
        "excluded_question_ids": sorted(excluded),
        "inputs": {name: sha256(bundle / name) for name in names},
        "reference_policy": f"Opened by execution only after all {2 * len(questions)} predictions; no tuning during this cohort.",
        "limitations": [
            "Development only, not held-out validation.",
            "Balanced natural warm-cache order, not cold-cache or repeated timing.",
            "Graph coverage is partial; public schema/coverage metadata is shared across conditions when frozen in the source protocol.",
        ],
    }
    plan["bundle_id"] = object_hash(plan)
    write_json(bundle / "protocol.json", plan)
    return plan


def verify(bundle):
    plan = read_json(bundle / "protocol.json")
    if (
        object_hash({k: v for k, v in plan.items() if k != "bundle_id"})
        != plan["bundle_id"]
    ):
        raise ValueError("Development protocol changed")
    if code_hashes() != plan["code"] or any(
        sha256(bundle / name) != digest for name, digest in plan["inputs"].items()
    ):
        raise ValueError("Frozen code or cohort inputs changed")
    questions = read_json(bundle / "questions.json")
    n = plan["questions"]
    if (
        type(n) is not int
        or not 1 <= n <= 50
        or type(plan["max_decisions_requests"]) is not int
        or plan["max_decisions_requests"] != n
        or len(questions) != n
        or len({q["question_id"] for q in questions}) != n
        or any(q["question_id"] in plan["excluded_question_ids"] for q in questions)
    ):
        raise ValueError("Cohort size, uniqueness or freshness failed")
    if read_json(bundle / "schedule.json") != schedule(n):
        raise ValueError("Condition order changed")
    return plan, questions


def run(bundle, output, allow_paid=False):
    if not allow_paid:
        raise ValueError("Explicit --allow-decisions-api is required")
    plan, questions = verify(bundle)
    if output.exists():
        raise ValueError(
            "Use a new run directory; no automatic repeats or paid retries"
        )
    logging.getLogger("neo4j").setLevel(logging.CRITICAL)
    from baseline.local_reader import LocalReader
    from routing.audit import EvidenceAudit
    from routing.config import database_settings, decisions_key
    from routing.decisions import DecisionsClient
    from routing.embedding import LocalEmbedding
    from routing.reader import CheckedReader
    from routing.router import DecisionsRouter, FixedRouter
    from routing.sql_plan import LocalSQLPlanner
    from routing.stores import LiveStores
    from routing.tokenizer import local_tokenizer

    started = perf_counter()
    base = LocalReader()
    if base.metadata["digest"] != plan["reader_digest"]:
        raise ValueError("Reader model changed")
    lock = read_json(REPO / "postgres/sources.lock.json")
    archive = REPO.parent / "data/corpus.zip"
    if sha256(archive) != lock["corpus"]["archive_sha256"]:
        raise ValueError("Pinned source archive changed")
    client = DecisionsClient(
        decisions_key(REPO.parent),
        allow_paid=True,
        max_requests=plan["max_decisions_requests"],
        journal=lambda events: write_json(output / "decisions-ledger.json", events),
    )
    cfg = database_settings(REPO.parent)
    reader, embedding, tokenizer = (
        CheckedReader(base),
        LocalEmbedding(REPO.parent / ".local/hf-cache"),
        local_tokenizer(REPO.parent),
    )
    auditor = EvidenceAudit(archive)
    catalog = None
    if plan.get("routing_catalog"):
        from routing.catalog import RoutingCatalog

        catalog = RoutingCatalog(bundle / "routing-catalog.json")
    pipelines = {
        c: Pipeline(
            DecisionsRouter(client) if c == "routed" else FixedRouter("all"),
            lambda allowed: LiveStores(cfg, allowed, sql_planner=LocalSQLPlanner(base)),
            embedding,
            reader,
            tokenizer,
            Config(),
            auditor=auditor,
            catalog=catalog,
        )
        for c in ("routed", "all")
    }
    manifest = {
        "protocol": plan,
        "reader": base.metadata,
        "ollama": base.runtime,
        "startup_seconds": perf_counter() - started,
        "max_decisions_requests": plan["max_decisions_requests"],
        "verify_sources": True,
        "graph_catalog_used_for_routing": catalog is not None,
        "routing_catalog_sha256": catalog.digest if catalog else None,
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "questions.json", questions)
    try:
        for job_number, job in enumerate(schedule(len(questions))):
            i, condition = job["index"], job["condition"]
            write_json(
                output / "progress.json",
                {
                    "completed": job_number,
                    "active": job,
                    "decisions_requests": len(client.events),
                },
            )
            result = pipelines[condition].run(questions[i])
            write_json(output / condition / "results" / f"{i:05d}.json", result)
            print(
                json.dumps(
                    {
                        "completed": job_number + 1,
                        "condition": condition,
                        "question": i + 1,
                        "status": result["status"],
                        "option": result.get("decision", {}).get("option"),
                        "answer": result.get("answer", {}).get("answer"),
                        "seconds": round(result["total_seconds"], 2),
                        "decisions_requests": len(client.events),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            if result["status"] == "failed" and (
                job_number == 0
                or result.get("failure_details", {}).get("infrastructure")
            ):
                raise RuntimeError(
                    "Batch stopped on infrastructure/integration failure; saved attempts are preserved and no later requests were made"
                )
        write_json(
            output / "progress.json",
            {
                "completed": 2 * len(questions),
                "active": None,
                "decisions_requests": len(client.events),
                "complete": True,
            },
        )
    finally:
        auditor.close()
    report(bundle, output)


def report(bundle, output):
    plan, questions = verify(bundle)
    manifest = read_json(output / "manifest.json")
    if manifest["protocol"] != plan:
        raise ValueError("Run belongs to a different frozen protocol")
    ordered = {}
    for c in ("routed", "all"):
        paths = sorted((output / c / "results").glob("*.json"))
        if len(paths) != len(questions):
            raise ValueError("Incomplete results; no scoring partial runs")
        ordered[c] = [read_json(p) for p in paths]
        if [r["question"] for r in ordered[c]] != questions:
            raise ValueError("Question identity/order mismatch")
    # All predictions exist before answer labels are opened.
    references = read_json(bundle / "references.json")
    coverage = set(read_json(Path(plan["source_bundle"]) / "graph-table-ids.json"))
    from evaluation.metrics import aggregate, paired, score_question

    scored = {
        c: [
            score_question(r, references[q["question_id"]], q["table_id"] in coverage)
            for q, r in zip(questions, records)
        ]
        for c, records in ordered.items()
    }
    ledger = read_json(output / "decisions-ledger.json")
    if len(ledger) > plan["max_decisions_requests"]:
        raise ValueError("Decisions budget exceeded")
    summary = {
        "bundle_id": plan["bundle_id"],
        "complete": True,
        "questions_per_condition": len(questions),
        "conditions": {c: aggregate(rows) for c, rows in scored.items()},
        "paired_all_to_routed": paired(scored["all"], scored["routed"]),
        "graph_slices": {
            c: {
                label: aggregate(
                    [r for r in rows if r["graph_covered_snapshot"] is covered]
                )
                for label, covered in (("covered", True), ("uncovered", False))
            }
            for c, rows in scored.items()
        },
        "decisions_requests": len(ledger),
        "decision_statuses": {
            status: sum(e["status"] == status for e in ledger)
            for status in {e["status"] for e in ledger}
        },
        "source_audits_passed": sum(
            r.get("source_audit", {}).get("passed", False)
            for records in ordered.values()
            for r in records
        ),
        "reader_api_cost_usd": 0,
        "manual_support_audit": "pending",
        "results": scored,
    }
    write_json(output / "summary.json", summary)
    print(
        json.dumps(
            {
                "complete": True,
                "scores": {
                    c: {
                        k: summary["conditions"][c][k]
                        for k in (
                            "em_percent",
                            "f1_percent",
                            "failed_percent",
                            "abstained_percent",
                        )
                    }
                    for c in scored
                },
                "decisions_requests": len(ledger),
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, default=REPO / "artifacts/bundles/hybridqa-dev300"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-decisions-api", action="store_true")
    parser.add_argument("--questions-count", type=int, default=50)
    args = parser.parse_args()
    if args.action == "prepare":
        plan = prepare(args.bundle, args.source, args.questions_count)
        print(
            json.dumps(
                {
                    "bundle_id": plan["bundle_id"],
                    "questions": plan["questions"],
                    "max_decisions_requests": plan["max_decisions_requests"],
                    "jobs_executed": 0,
                }
            ),
            flush=True,
        )
    elif args.output is None:
        parser.error("--output is required")
    elif args.action == "run":
        run(args.bundle, args.output, args.allow_decisions_api)
    else:
        report(args.bundle, args.output)


if __name__ == "__main__":
    main()
