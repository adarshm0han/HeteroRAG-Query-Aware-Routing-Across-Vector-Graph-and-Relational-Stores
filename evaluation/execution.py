"""Checkpointed execution of the frozen inference algorithms, one job at a time."""

from copy import deepcopy
from time import perf_counter

from baseline.common import object_hash, public_question, safe_error
from baseline.retrieval import pack_context
from evaluation.checkpoints import Checkpoints, InterruptedStage, load, save
from routing.failures import failure_details
from routing.fusion import fuse
from routing.options import OPTIONS
from routing.retrieval import retrieve
from routing.router import confidence_weights


def raw_decision(source):
    decision = deepcopy(source)
    if "raw_option" not in decision or "raw_probabilities" not in decision:
        raise ValueError("Guard ablation requires a preserved raw Decisions response")
    decision.update(
        option=decision["raw_option"],
        probabilities=decision["raw_probabilities"],
        confidence=decision["raw_confidence"],
        method="recorded_decisions",
        api_requests=0,
        seconds=0.0,
        guard_disabled=True,
        probability_method="raw_decisions_distribution",
    )
    decision.pop("evidence_policy", None)
    decision.update(confidence_weights(decision["option"], decision["probabilities"]))
    return decision


def replay_fusion(evidence, config, weights):
    candidate = deepcopy(evidence)
    records = candidate["candidate_records"]
    if (
        object_hash([records[k] for k in sorted(records)])
        != candidate["candidate_evidence_sha256"]
        or object_hash(candidate["rankings"]) != candidate["rankings_sha256"]
    ):
        raise ValueError("Fusion source candidate snapshot is corrupt")
    start = perf_counter()
    candidate["items"] = fuse(
        candidate["rankings"],
        records,
        config.row_k,
        config.passage_k,
        config.fusion,
        weights,
        protected=candidate["protected_passages"],
    )
    for key in ("context", "evidence", "context_tokens", "packing_dropped_ids"):
        candidate.pop(key, None)
    candidate.update(
        fusion=config.fusion,
        fusion_weights=weights if config.fusion == "weighted_rrf" else None,
        fusion_seconds=perf_counter() - start,
        seconds=None,
        embedding_seconds=None,
        retrieval_replayed=True,
    )
    return candidate


def execute(
    question,
    job,
    root,
    *,
    router,
    stores_factory,
    embedding,
    reader,
    tokenizer,
    config,
    auditor,
    catalog,
    source=None,
):
    """Factories are injected so the complete control flow can be tested offline."""
    question = public_question(question)
    checkpoint = Checkpoints(
        root, {"job": job, "question": question, "config": vars(config)}
    )
    result_path = root / "result.json"
    if result_path.exists():
        return load(result_path)
    started = perf_counter()
    stage = "routing"
    client = getattr(router, "client", None)
    result = {
        "question": question,
        "job_id": job["id"],
        "status": "failed",
        "database_calls": [],
        "decisions_calls": [],
        "route_replayed": job["mode"] != "live",
        "timing_eligible": not checkpoint.resumed and job["mode"] == "live",
        "resumed": checkpoint.resumed,
        "retrieval_replayed": job["mode"] == "fusion_replay",
    }
    events_path = root / "database-events.json"
    try:
        metadata = catalog.describe(question["table_id"])
        result.update(
            routing_metadata=metadata,
            catalog_sha256=catalog.digest,
            graph_availability_hint=metadata["graph_available"],
        )
        if source is not None and (
            source["question"] != question
            or source.get("routing_metadata") != metadata
            or source.get("catalog_sha256") != catalog.digest
        ):
            raise ValueError("Replay source identity or public metadata differs")

        def choose():
            if job["mode"] == "live":
                return router.choose(
                    question, metadata["graph_available"], table_metadata=metadata
                )
            if source is None or "decision" not in source:
                raise InterruptedStage(
                    "Source routing failed; preserve dependent failure"
                )
            if job["mode"] == "raw_route":
                return raw_decision(source["decision"])
            decision = deepcopy(source["decision"])
            decision.update(api_requests=0, seconds=0.0)
            if decision["method"] == "decisions":
                decision["method"] = "recorded_decisions"
            return decision

        decision = checkpoint.step("routing", choose, recover=True)
        selected = OPTIONS[decision["option"]]
        result.update(decision=decision, selected_stores=list(selected))
        stage = "retrieval"

        def fetch():
            if job["mode"] == "fusion_replay":
                if source is None or "retrieval" not in source:
                    raise InterruptedStage(
                        "Source retrieval failed; preserve dependent failure"
                    )
                return replay_fusion(
                    source["retrieval"], config, decision.get("store_weights")
                )
            stores = stores_factory(selected, lambda events: save(events_path, events))
            try:
                evidence = retrieve(
                    question,
                    stores,
                    embedding,
                    config.row_k,
                    config.passage_k,
                    config.fusion,
                    decision.get("store_weights"),
                )
                if not {event["store"] for event in stores.events} <= set(selected):
                    raise RuntimeError("Database access outside selected stores")
                if not evidence["title"]:
                    evidence["title"] = metadata["title"]
                return evidence
            finally:
                save(events_path, list(stores.events))
                stores.close()

        evidence = checkpoint.step(
            "retrieval", fetch, recover=job["mode"] == "fusion_replay"
        )
        result["retrieval"] = evidence
        if any(not set(item["sources"]) <= set(selected) for item in evidence["items"]):
            raise ValueError("Evidence claims an unselected database")
        stage = "source_audit"

        def audit():
            start = perf_counter()
            checked = auditor.check(question["table_id"], evidence["items"])
            return {"check": checked, "seconds": perf_counter() - start}

        audited = checkpoint.step("audit", audit, recover=True)
        result.update(
            source_audit=audited["check"], source_audit_seconds=audited["seconds"]
        )
        stage = "packing"

        def pack():
            items = [i for i in evidence["items"] if i["kind"] != "passage"] + [
                i for i in evidence["items"] if i["kind"] == "passage"
            ]
            context, packed, tokens = pack_context(
                question["question"],
                evidence["title"],
                items,
                tokenizer,
                config.context_tokens,
            )
            return {
                "context": context,
                "evidence": packed,
                "context_tokens": tokens,
                "packing_policy": "table_evidence_first_v1",
                "packing_dropped_ids": [
                    i["id"]
                    for i in evidence["items"]
                    if i["id"] not in {p["source_id"] for p in packed}
                ],
            }

        packed = checkpoint.step("packing", pack, recover=True)
        result["retrieval"] = {**evidence, **packed}
        stage = "answer"

        def answer():
            if packed["evidence"]:
                return reader.generate(packed["context"], packed["evidence"])
            return {
                "answer": "",
                "citations": [],
                "abstain": True,
                "seconds": 0.0,
                "generation_skipped": "no_evidence",
                "local_inference": True,
                "api_cost_usd": 0,
            }

        result["answer"] = checkpoint.step("answer", answer, recover=True)
        result["status"] = "completed" if packed["evidence"] else "empty_evidence"
        if source is not None:
            result["replay_source"] = {
                "job_id": source["job_id"],
                "sha256": object_hash(source),
            }
    except Exception as exc:  # noqa: BLE001 -- retain every attempted outcome
        result.update(
            status="failed",
            failed_stage=stage,
            error=safe_error(exc),
            failure_details=failure_details(exc),
        )
        if isinstance(exc, InterruptedStage):
            result["failure_details"] = {
                "infrastructure": True,
                "interrupted_external_stage": True,
            }
        if hasattr(exc, "metadata"):
            result["failed_call"] = exc.metadata
    finally:
        if events_path.exists():
            result["database_calls"] = load(events_path)
        result["decisions_calls"] = [
            {k: v for k, v in event.items() if k != "response"}
            for event in getattr(client, "events", [])
        ]
        result["total_seconds"] = perf_counter() - started
        if checkpoint.resumed or getattr(client, "reused", False):
            result["timing_eligible"] = False
    save(result_path, result)
    return result
