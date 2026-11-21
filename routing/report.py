"""Offline run summaries; answer labels are loaded only after execution."""

import collections
import statistics

from baseline.common import answer_scores


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(results, references=None):
    references = references or {}
    statuses = collections.Counter(row["status"] for row in results)
    calls = [call for row in results for call in row["database_calls"]]
    latencies = [row["total_seconds"] for row in results if row["status"] != "failed"]
    scored = []
    for row in results:
        qid = row["question"]["question_id"]
        if qid in references:
            # Failures count as empty predictions; never silently drop hard cases.
            predicted = row.get("answer", {}).get("answer", "")
            scored.append(
                {"em": 0.0, "f1": 0.0}
                if row["status"] == "failed"
                else answer_scores(references[qid]["answer"], predicted)
            )
    return {
        "questions": len(results),
        "graph_availability_hints": dict(
            collections.Counter(
                "covered"
                if row.get("graph_availability_hint") is True
                else "uncovered"
                if row.get("graph_availability_hint") is False
                else "unknown"
                for row in results
            )
        ),
        "statuses": dict(statuses),
        "options": dict(
            collections.Counter(
                row.get("decision", {}).get("option", "routing_failed")
                for row in results
            )
        ),
        "raw_decisions_options": dict(
            collections.Counter(
                row["decision"].get("raw_option", row["decision"]["option"])
                for row in results
                if row.get("decision", {}).get("method")
                in ("decisions", "recorded_decisions")
            )
        ),
        "evidence_policy_corrections": sum(
            bool(row.get("decision", {}).get("evidence_policy", {}).get("changed"))
            for row in results
        ),
        "database_operations": dict(
            collections.Counter(call["store"] for call in calls)
        ),
        "database_data_requests": dict(
            collections.Counter(
                call["store"] for call in calls if call["kind"] == "data"
            )
        ),
        "database_operation_failures": sum(
            call["status"] == "failed" for call in calls
        ),
        "decisions_requests": sum(
            len(row.get("decisions_calls", [])) for row in results
        ),
        "replayed_decisions": sum(
            row.get("decision", {}).get("method") == "recorded_decisions"
            for row in results
        ),
        "decisions_usage": [
            call.get("usage", {})
            for row in results
            for call in row.get("decisions_calls", [])
        ],
        "answer_model_api_cost_usd": 0,
        "candidate_passages": sum(
            row.get("retrieval", {}).get("candidate_passage_count", 0)
            for row in results
        ),
        "candidate_passage_text_bytes": sum(
            row.get("retrieval", {}).get("candidate_passage_text_bytes", 0)
            for row in results
        ),
        "pinecone_fetch_ids": sum(
            call.get("id_count", 0) for call in calls if call["store"] == "pinecone"
        ),
        "sql_execution_statuses": dict(
            collections.Counter(
                sql["status"]
                for row in results
                if (sql := row.get("retrieval", {}).get("sql"))
            )
        ),
        "sql_planner_seconds": sum(
            (row.get("retrieval", {}).get("sql") or {})
            .get("planner", {})
            .get("seconds", 0)
            for row in results
        ),
        "latency_seconds_successful_only": {
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "mean": statistics.mean(latencies) if latencies else None,
        },
        "scored_questions": len(scored),
        "em": 100 * statistics.mean(score["em"] for score in scored)
        if scored
        else None,
        "f1": 100 * statistics.mean(score["f1"] for score in scored)
        if scored
        else None,
        "notes": [
            "Connection/control operations are separate from data requests.",
            "Timing includes routing, local SQL planning, embedding, retrieval and local answer generation; startup is in the manifest.",
            "Citation-ID validity does not establish citation entailment or complete supporting evidence.",
        ],
    }
