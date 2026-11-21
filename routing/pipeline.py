"""Question -> Decisions -> selected stores -> fusion -> local Qwen."""

import math
from dataclasses import dataclass
from time import perf_counter

from baseline.common import public_question, safe_error
from baseline.retrieval import pack_context

from .failures import failure_details
from .options import OPTIONS
from .retrieval import retrieve


@dataclass(frozen=True)
class Config:
    row_k: int = 8
    passage_k: int = 12
    context_tokens: int = 6000
    fusion: str = "rrf"

    def __post_init__(self):
        if (
            self.row_k < 1
            or self.passage_k < 1
            or not 512 <= self.context_tokens <= 10000
        ):
            raise ValueError("Invalid evidence limits")
        if self.fusion not in ("rrf", "concat", "weighted_rrf"):
            raise ValueError("Invalid fusion mode")


class Pipeline:
    def __init__(
        self,
        router,
        stores_factory,
        embedding,
        reader,
        tokenizer,
        config=Config(),
        auditor=None,
        catalog=None,
    ):
        self.router = router
        self.stores_factory = stores_factory
        self.embedding = embedding
        self.reader = reader
        self.tokenizer = tokenizer
        self.config = config
        self.auditor = auditor
        self.catalog = catalog

    def run(self, question, graph_available=None):
        question = public_question(question)
        if any(
            not isinstance(value, str) or not value.strip()
            for value in question.values()
        ):
            raise ValueError(
                "Question ID, text, and provided table ID must be nonempty strings"
            )
        started = perf_counter()
        stage, stores = "routing", None
        client = getattr(self.router, "client", None)
        decision_events = getattr(client, "events", [])
        decision_offset = len(decision_events)
        result = {
            "question": question,
            "graph_availability_hint": graph_available,
            "route_replayed": getattr(self.router, "is_replay", False),
        }
        try:
            metadata = None
            if self.catalog is not None:
                lookup_started = perf_counter()
                metadata = self.catalog.describe(question["table_id"])
                known = metadata["graph_available"]
                if (
                    graph_available is not None
                    and known is not None
                    and graph_available != known
                ):
                    raise ValueError("Conflicting graph availability metadata")
                if graph_available is None:
                    graph_available = known
                result.update(
                    routing_metadata=metadata,
                    catalog_sha256=self.catalog.digest,
                    metadata_lookup_seconds=perf_counter() - lookup_started,
                    graph_availability_hint=graph_available,
                )
            decision = self.router.choose(
                question,
                graph_available,
                **({"table_metadata": metadata} if metadata is not None else {}),
            )
            selected = OPTIONS[decision["option"]]
            result.update(decision=decision, selected_stores=list(selected))
            if self.config.fusion == "weighted_rrf":
                weights = decision.get("store_weights", {})
                if not isinstance(weights, dict) or any(
                    store not in weights
                    or not isinstance(weights[store], (int, float))
                    or not math.isfinite(weights[store])
                    or weights[store] <= 0
                    for store in selected
                ):
                    raise ValueError("Decisions response has no valid fusion weights")
            stage = "retrieval"
            stores = self.stores_factory(selected)
            evidence = retrieve(
                question,
                stores,
                self.embedding,
                self.config.row_k,
                self.config.passage_k,
                self.config.fusion,
                decision.get("store_weights"),
            )
            if not evidence["title"] and metadata is not None:
                evidence["title"] = metadata["title"]
            result["retrieval"] = evidence
            if not {event["store"] for event in stores.events} <= set(selected):
                raise RuntimeError(
                    "Observed access to a database outside the selected option"
                )
            if any(
                not set(item["sources"]) <= set(selected) for item in evidence["items"]
            ):
                raise RuntimeError(
                    "Evidence claims a source outside the selected option"
                )
            if self.auditor is not None:
                stage = "source_audit"
                audit_started = perf_counter()
                try:
                    result["source_audit"] = self.auditor.check(
                        question["table_id"], evidence["items"]
                    )
                finally:
                    result["source_audit_seconds"] = perf_counter() - audit_started
            stage = "packing"
            # Keep the compact table-selection evidence before long passages.
            # Preserve fusion order within each type; never truncate a row merely
            # because a verbose, highly ranked passage consumed the token budget.
            packing_items = [
                item for item in evidence["items"] if item["kind"] != "passage"
            ] + [item for item in evidence["items"] if item["kind"] == "passage"]
            context, packed, tokens = pack_context(
                question["question"],
                evidence["title"],
                packing_items,
                self.tokenizer,
                self.config.context_tokens,
            )
            result["retrieval"] = {
                **evidence,
                "context": context,
                "evidence": packed,
                "context_tokens": tokens,
                "packing_policy": "table_evidence_first_v1",
                "packing_dropped_ids": [
                    item["id"]
                    for item in evidence["items"]
                    if item["id"] not in {p["source_id"] for p in packed}
                ],
            }
            stage = "answer"
            if packed:
                result["answer"] = self.reader.generate(context, packed)
                result["status"] = "completed"
            else:
                result["answer"] = {
                    "answer": "",
                    "citations": [],
                    "abstain": True,
                    "seconds": 0.0,
                    "generation_skipped": "no_evidence",
                    "local_inference": True,
                    "api_cost_usd": 0,
                }
                result["status"] = "empty_evidence"
        except Exception as exc:  # noqa: BLE001 -- persist failures from all database SDKs
            result.update(
                status="failed",
                failed_stage=stage,
                error=safe_error(exc),
                failure_details=failure_details(exc),
            )
            if hasattr(exc, "metadata"):
                result["failed_call"] = exc.metadata
        finally:
            if stores is not None:
                result["database_calls"] = list(stores.events)
                try:
                    stores.close()
                except Exception as exc:  # noqa: BLE001 -- keep the original run failure
                    result["cleanup_error"] = safe_error(exc)
            else:
                result["database_calls"] = []
            result["decisions_calls"] = list(decision_events[decision_offset:])
            result["total_seconds"] = perf_counter() - started
        return result
