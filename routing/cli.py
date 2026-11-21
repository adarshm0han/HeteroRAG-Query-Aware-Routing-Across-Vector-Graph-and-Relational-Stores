"""Run the configured pipeline. Paid Decisions calls require explicit opt-in."""

import argparse
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from baseline.common import REPO, public_question, read_json, sha256, write_json
from evaluation.cohorts import code_hashes

from .config import database_settings, decisions_key
from .options import OPTIONS
from .passage_selection import VERSION as PASSAGE_POLICY
from .pipeline import Config, Pipeline
from .report import summarize
from .router import DecisionsRouter, FixedRouter, RecordedRouter


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=REPO.parent)
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="JSON list: question_id, question, table_id",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="A new, empty run directory"
    )
    parser.add_argument(
        "--option", choices=["decisions", *OPTIONS], default="decisions"
    )
    parser.add_argument(
        "--allow-decisions-api",
        action="store_true",
        help="Explicitly permit billed Decisions routing; never paid answer generation",
    )
    parser.add_argument("--max-decision-requests", type=int, default=1)
    parser.add_argument(
        "--decision-records",
        type=Path,
        help="Replay a previous run's Decisions choices without API requests",
    )
    parser.add_argument(
        "--graph-catalog",
        type=Path,
        help="Optional JSON list of covered table IDs, loaded locally",
    )
    parser.add_argument(
        "--routing-catalog",
        type=Path,
        help="Verified local public table schema/coverage catalog; no cells or answer labels",
    )
    parser.add_argument(
        "--references",
        type=Path,
        help="Optional answer mapping; loaded for scoring only after all predictions",
    )
    parser.add_argument(
        "--fusion", choices=["rrf", "concat", "weighted_rrf"], default="rrf"
    )
    parser.add_argument(
        "--collect-fusion-weights",
        action="store_true",
        help="Compatibility flag; probability-derived weights are now always recorded",
    )
    parser.add_argument("--evaluation-bundle", type=Path)
    parser.add_argument("--cohort")
    parser.add_argument("--condition")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--smoke-recovery",
        action="store_true",
        help="Replay a prior smoke's paid decisions after integration fixes; never new paid calls",
    )
    parser.add_argument("--row-k", type=int, default=8)
    parser.add_argument("--passage-k", type=int, default=12)
    parser.add_argument("--context-tokens", type=int, default=6000)
    parser.add_argument(
        "--verify-sources",
        action="store_true",
        help="Audit retrieved evidence against the local pinned corpus before answering",
    )
    args = parser.parse_args(argv)
    if args.smoke_recovery and (
        not args.evaluation_bundle
        or not args.decision_records
        or args.allow_decisions_api
    ):
        parser.error(
            "Smoke recovery requires a frozen bundle and saved decisions, with paid calls disabled"
        )
    # Reject paid operation before credentials, file writes, or network access.
    if (
        args.option == "decisions"
        and not args.allow_decisions_api
        and not args.decision_records
    ):
        parser.error(
            "Decisions is disabled. Live routing requires --allow-decisions-api after spending is authorized."
        )
    if args.option != "decisions" and (
        args.decision_records
        or args.fusion == "weighted_rrf"
        or args.collect_fusion_weights
    ):
        parser.error(
            "Recorded decisions and query-dependent fusion weights require --option decisions"
        )
    return args


def main(argv=None):
    started = perf_counter()
    args = arguments(argv)
    logging.getLogger("neo4j").setLevel(logging.CRITICAL)
    config = Config(args.row_k, args.passage_k, args.context_tokens, args.fusion)
    raw = read_json(args.questions)
    questions = [public_question(question) for question in raw]
    if not questions or len({q["question_id"] for q in questions}) != len(questions):
        raise ValueError("Provide a nonempty question list with unique IDs")
    if (
        args.option == "decisions"
        and not args.decision_records
        and args.max_decision_requests < len(questions)
    ):
        raise ValueError("Request limit is smaller than the question list")
    evaluation = None
    if args.evaluation_bundle:
        from evaluation.cohorts import validate_run

        evaluation = validate_run(args, config, questions)
    elif args.cohort or args.condition:
        raise ValueError("Cohort and condition require an evaluation bundle")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(
            "Use a new output directory; existing runs will not be overwritten or repeated"
        )
    coverage = set(read_json(args.graph_catalog)) if args.graph_catalog else None
    catalog = None
    if args.routing_catalog:
        from .catalog import RoutingCatalog

        catalog = RoutingCatalog(args.routing_catalog)

    from baseline.local_reader import LocalReader

    from .decisions import DecisionsClient
    from .embedding import LocalEmbedding
    from .reader import VERSION as ANSWER_CHECK_VERSION
    from .reader import CheckedReader
    from .sql_plan import VERSION as SQL_PLAN_VERSION
    from .sql_plan import LocalSQLPlanner
    from .stores import LiveStores
    from .tokenizer import local_tokenizer

    reader = LocalReader("qwen3.5:4b")
    if evaluation and reader.metadata["digest"] != evaluation["reader_digest"]:
        raise ValueError("Local reader differs from the frozen evaluation model")
    if args.decision_records:
        recorded = [
            read_json(path)
            for path in sorted((args.decision_records / "results").glob("*.json"))
        ]
        if not recorded:
            raise ValueError("No saved Decisions results found")
        router = RecordedRouter(recorded)
    elif args.option == "decisions":
        client = DecisionsClient(
            decisions_key(args.workspace),
            allow_paid=args.allow_decisions_api,
            max_requests=args.max_decision_requests,
            journal=lambda events: write_json(
                args.output / "decisions-ledger.json", events
            ),
        )
        router = DecisionsRouter(client)
    else:
        router = FixedRouter(args.option)
    cfg = database_settings(args.workspace)
    auditor = None
    if args.verify_sources:
        from .audit import EvidenceAudit

        archive = args.workspace / "data/corpus.zip"
        lock = read_json(REPO / "postgres/sources.lock.json")
        if sha256(archive) != lock["corpus"]["archive_sha256"]:
            raise ValueError("Source archive differs from the pinned corpus")
        auditor = EvidenceAudit(archive)
    pipeline = Pipeline(
        router,
        lambda allowed: LiveStores(cfg, allowed, sql_planner=LocalSQLPlanner(reader)),
        LocalEmbedding(args.workspace / ".local/hf-cache"),
        CheckedReader(reader),
        local_tokenizer(args.workspace),
        config,
        auditor=auditor,
        catalog=catalog,
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": "provided-table; live independent retrieval for each question",
        "config": asdict(config),
        "option": args.option,
        "max_decision_requests": args.max_decision_requests
        if args.option == "decisions" and not args.decision_records
        else 0,
        "fusion_weight_method": "route_marginals_v1",
        "evaluation": evaluation,
        "smoke_recovery": args.smoke_recovery,
        "questions_sha256": sha256(args.questions),
        "graph_catalog_sha256": sha256(args.graph_catalog)
        if args.graph_catalog
        else None,
        "routing_catalog_sha256": catalog.digest if catalog else None,
        "reader": reader.metadata,
        "answer_check": ANSWER_CHECK_VERSION,
        "passage_membership": PASSAGE_POLICY,
        "packing_policy": "table_evidence_first_v1",
        "sql_planner": {
            "version": SQL_PLAN_VERSION,
            "model": reader.metadata,
            "local_inference": True,
            "api_cost_usd": 0,
        },
        "ollama": reader.runtime,
        "code": code_hashes(),
        "decision_records": str(args.decision_records)
        if args.decision_records
        else None,
        "startup_seconds": perf_counter() - started,
        "verify_sources": args.verify_sources,
    }
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "questions.json", questions)
    results = []
    stopped_on_failure = False
    for index, question in enumerate(questions):
        available = question["table_id"] in coverage if coverage is not None else None
        result = pipeline.run(question, available)
        results.append(result)
        # Ordinal filenames cannot escape the output directory through input IDs.
        write_json(args.output / "results" / f"{index:05d}.json", result)
        write_json(args.output / "summary.json", summarize(results))
        print(
            json.dumps(
                {
                    "completed": index + 1,
                    "status": result["status"],
                    "option": result.get("decision", {}).get("option"),
                }
            ),
            flush=True,
        )
        if result["status"] == "failed" and (
            not args.continue_on_error
            or result.get("failure_details", {}).get("infrastructure")
        ):
            stopped_on_failure = True
            break
    if auditor is not None:
        auditor.close()
    references = read_json(args.references) if args.references else None
    summary = summarize(results, references)
    summary["requested_questions"] = len(questions)
    summary["run_complete"] = len(results) == len(questions)
    summary["all_calls_successful"] = all(row["status"] != "failed" for row in results)
    write_json(args.output / "summary.json", summary)
    if (
        stopped_on_failure
        or not summary["run_complete"]
        or (not args.continue_on_error and not summary["all_calls_successful"])
    ):
        raise SystemExit(
            "Run stopped after a failure; completed results and call logs are saved."
        )
