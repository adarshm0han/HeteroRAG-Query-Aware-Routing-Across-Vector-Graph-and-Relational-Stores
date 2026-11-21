"""Checkpointed, resumable evaluation runner. No automatic external retries."""

import argparse
import json
import logging
import os
import threading
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from baseline.common import REPO, read_json, sha256
from evaluation.checkpoints import PaidLedger, load, save, study_lock
from evaluation.execution import execute
from evaluation.protocol import verify
from routing.pipeline import Config


def deadline_record(protocol, created_at):
    start = datetime.fromisoformat(created_at)
    if start.tzinfo is None:
        raise ValueError("Study creation timestamp must include a timezone")
    return {
        "bundle_id": protocol["bundle_id"],
        "started_at": start.isoformat(),
        "expires_at": (
            start + timedelta(seconds=protocol["wall_time_budget_seconds"])
        ).isoformat(),
    }


@contextmanager
def wall_deadline(
    root,
    protocol,
    created_at,
    *,
    now=None,
    timer_factory=threading.Timer,
    exit_process=os._exit,
):
    """Stops this worker only; never touches Ollama or a database service.

    Killing an in-flight call makes its persisted started record uncertain. The
    existing durable ledger never retries it. Complete results remain available.
    Expired studies cannot acquire a fresh time allowance by resuming.
    """
    if not protocol.get("wall_time_budget_seconds"):
        yield
        return
    record = deadline_record(protocol, created_at)
    path = root / "deadline.json"
    if path.exists() and load(path) != record:
        raise ValueError("Persisted study deadline mismatch")
    if not path.exists():
        save(path, record)
    remaining = (
        datetime.fromisoformat(record["expires_at"]) - (now or datetime.now(UTC))
    ).total_seconds()
    if remaining <= 0:
        raise TimeoutError(
            "Study deadline has expired; saved work remains available for offline reporting"
        )

    def expire():
        try:
            save(
                root / "deadline-expired.json",
                {
                    **record,
                    "status": "deadline_reached",
                    "outcome": "Worker stopped; incomplete phases must not be presented as complete",
                },
            )
        finally:
            exit_process(124)

    timer = timer_factory(remaining, expire)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def run(bundle, output, workspace, *, phase, resume, allow_live, allow_paid, paid_cap):
    protocol, jobs = verify(bundle, check_code=True)
    if not allow_live or not allow_paid or paid_cap != protocol["paid_request_cap"]:
        raise PermissionError(
            "Launch requires --allow-live --allow-decisions-api and the exact frozen --max-decision-requests cap"
        )
    phases = protocol["phases"] if phase == "all" else [phase]
    if not set(phases) <= set(protocol["phases"]):
        raise ValueError("Phase outside the frozen schedule")
    existed = output.exists() and any(output.iterdir())
    if existed and not resume:
        raise ValueError("Existing evaluation requires explicit --resume")
    with study_lock(output), ExitStack() as lifecycle:
        path = output / "study.json"
        identity = {"bundle_id": protocol["bundle_id"], "paid_request_cap": paid_cap}
        if path.exists():
            manifest = load(path)
            if any(manifest[k] != v for k, v in identity.items()):
                raise ValueError("Resume study/budget mismatch")
        else:
            if list((output / "jobs").glob("*/result.json")) or list(
                (output / "paid").glob("*.json")
            ):
                raise ValueError("Existing work has no study manifest")
            manifest = {
                **identity,
                "created_at": datetime.now(UTC).isoformat(),
                "sessions": [],
            }
            save(path, manifest)
        ledger = PaidLedger(
            output / "paid", paid_cap, [j["id"] for j in jobs if j["paid"]]
        )
        if all(
            (output / "jobs" / j["id"] / "result.json").exists()
            for j in jobs
            if j["phase"] in phases
        ):
            from evaluation.report import report

            for selected_phase in phases:
                report(bundle, output, selected_phase)
            return
        lifecycle.enter_context(wall_deadline(output, protocol, manifest["created_at"]))
        for package, expected in protocol["dependency_versions"].items():
            if version(package) != expected:
                raise ValueError("Dependency version changed: " + package)
        archive = workspace / "data/corpus.zip"
        if sha256(archive) != protocol["sources"]["corpus"]["archive_sha256"]:
            raise ValueError("Corpus archive differs from the freeze")
        # All authorization and integrity checks precede service initialization.
        from baseline.local_reader import LocalReader
        from evaluation.report import report
        from routing.audit import EvidenceAudit
        from routing.catalog import RoutingCatalog
        from routing.config import database_settings, decisions_key
        from routing.embedding import LocalEmbedding
        from routing.reader import CheckedReader
        from routing.router import DecisionsRouter, FixedRouter
        from routing.sql_plan import LocalSQLPlanner
        from routing.stores import LiveStores
        from routing.tokenizer import local_tokenizer

        start = perf_counter()
        local = LocalReader("qwen3.5:4b")
        if local.metadata["digest"] != protocol["reader_digest"]:
            raise ValueError("Reader model digest changed")
        cfg = database_settings(workspace)
        api_key = decisions_key(workspace)
        embedding = LocalEmbedding(workspace / ".local/hf-cache")
        tokenizer = local_tokenizer(workspace)
        catalog = RoutingCatalog(bundle / "routing-catalog.json")
        auditor = EvidenceAudit(archive)
        session = {
            "started_at": datetime.now(UTC).isoformat(),
            "startup_seconds": perf_counter() - start,
            "phases": phases,
            "resume": resume,
            "reader": local.metadata,
            "ollama": local.runtime,
        }
        manifest["sessions"].append(session)
        save(path, manifest)
        by_phase_question = {
            (j["phase"], j["question_id"], j["condition"]): j for j in jobs
        }
        try:
            for selected_phase in phases:
                preceding = protocol["phases"][
                    : protocol["phases"].index(selected_phase)
                ]
                for prior in preceding:
                    prior_jobs = [j for j in jobs if j["phase"] == prior]
                    if not all(
                        (output / "jobs" / j["id"] / "result.json").exists()
                        for j in prior_jobs
                    ):
                        raise ValueError("Previous phase is incomplete: " + prior)
                if selected_phase != "smoke":
                    smoke = [
                        load(output / "jobs" / j["id"] / "result.json")
                        for j in jobs
                        if j["phase"] == "smoke"
                    ]
                    if any(r["status"] == "failed" for r in smoke):
                        raise ValueError(
                            "Smoke has failures; diagnose and freeze a new study before proceeding"
                        )
                if (
                    selected_phase == "evaluation"
                    and not (output / "best-fixed.json").exists()
                ):
                    raise ValueError(
                        "Validation must seal best-fixed before evaluation"
                    )
                source = (
                    "timing" if selected_phase.startswith("timing-") else selected_phase
                )
                questions = {
                    q["question_id"]: q
                    for q in read_json(bundle / source / "questions.json")
                }
                for job in (j for j in jobs if j["phase"] == selected_phase):
                    directory = output / "jobs" / job["id"]
                    if (directory / "result.json").exists():
                        load(directory / "result.json")
                        continue
                    condition = protocol["conditions"][job["condition"]]
                    saved_source = None
                    if job["dependency"]:
                        dependency = by_phase_question[
                            (selected_phase, job["question_id"], job["dependency"])
                        ]
                        saved_source = load(
                            output / "jobs" / dependency["id"] / "result.json"
                        )
                    router = (
                        DecisionsRouter(
                            ledger.client(job["id"], api_key, allow_paid=True)
                        )
                        if job["paid"]
                        else FixedRouter(condition["option"])
                        if condition["option"] != "decisions"
                        else None
                    )
                    result = execute(
                        questions[job["question_id"]],
                        job,
                        directory,
                        router=router,
                        stores_factory=lambda allowed, journal: LiveStores(
                            cfg,
                            allowed,
                            sql_planner=LocalSQLPlanner(local),
                            journal=journal,
                        ),
                        embedding=embedding,
                        reader=CheckedReader(local),
                        tokenizer=tokenizer,
                        config=Config(
                            **{**protocol["config"], "fusion": condition["fusion"]}
                        ),
                        auditor=auditor,
                        catalog=catalog,
                        source=saved_source,
                    )
                    print(
                        json.dumps(
                            {
                                "phase": selected_phase,
                                "question_index": job["index"],
                                "condition": job["condition"],
                                "status": result["status"],
                                "paid_attempts": len(ledger.records),
                            }
                        ),
                        flush=True,
                    )
                    if result["status"] == "failed":
                        raise SystemExit(
                            "Stopped after retained failure. Inspect logs; explicit --resume continues without repeating this outcome or paid request."
                        )
                report(bundle, output, selected_phase)
        finally:
            auditor.close()
            session["ended_at"] = datetime.now(UTC).isoformat()
            session["session_seconds"] = perf_counter() - start
            save(path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=REPO.parent)
    parser.add_argument("--phase", default="smoke")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--allow-decisions-api", action="store_true")
    parser.add_argument("--max-decision-requests", type=int, default=0)
    args = parser.parse_args()
    logging.getLogger("neo4j").setLevel(logging.CRITICAL)
    run(
        args.bundle,
        args.output,
        args.workspace,
        phase=args.phase,
        resume=args.resume,
        allow_live=args.allow_live,
        allow_paid=args.allow_decisions_api,
        paid_cap=args.max_decision_requests,
    )


if __name__ == "__main__":
    main()
