"""Score persisted predictions without making any model or database requests."""

import collections
import csv
import json
from datetime import UTC, datetime

import numpy as np

from .common import answer_scores, normalize_answer, read_json, write_json


def cluster_interval(values, tables, draws=2000):
    """Percentile bootstrap by source table, preserving questions within each cluster."""
    if not values:
        return [None, None]
    groups = collections.defaultdict(list)
    for value, table in zip(values, tables):
        groups[table].append(value)
    sums = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(20261116)
    # Bounded batches avoid allocating a draws x corpus-size array on larger runs.
    samples = []
    for _ in range(draws):
        chosen = rng.integers(0, len(sums), len(sums))
        samples.append(sums[chosen].sum() / counts[chosen].sum())
    return [float(v) for v in np.percentile(samples, [2.5, 97.5])]


def evidence_diagnostics(reference, items):
    row_ids = {e["row_index"] for e in items if e["kind"] == "row"}
    passage_ids = {e["passage_id"] for e in items if e["kind"] == "passage"}
    traces = reference["trace_candidates"]
    hits = [
        node[1][0] in row_ids if node[3] == "table" else node[2] in passage_ids
        for node in traces
    ]
    gold = normalize_answer(reference["answer"])
    return {
        "weak_trace_any_hit": bool(any(hits)) if hits else None,
        "answer_string_in_context": bool(
            gold and any(gold in normalize_answer(e["text"]) for e in items)
        ),
    }


def report(output):
    manifest = read_json(output / "manifest.json")
    questions = read_json(output / "questions.json")
    refs = read_json(output / "references.json")
    plans = manifest["config"]["plans"]
    rows, retrievals = [], []
    for question in questions:
        qid = question["question_id"]
        path = output / "retrieval" / (qid + ".json")
        if not path.exists():
            continue
        retrieval = read_json(path)
        retrievals.append(retrieval)
        trace_category = (
            "+".join(sorted({n[3] for n in refs[qid]["trace_candidates"]}))
            or "untraced"
        )
        for plan in plans:
            answer_path = output / "answers" / f"{qid}.{plan}.json"
            answer = (
                read_json(answer_path)
                if answer_path.exists()
                else {"ok": False, "pending": True}
            )
            ctx = retrieval["contexts"][plan]
            prediction = answer.get("answer", "") if answer["ok"] else ""
            scores = (
                answer_scores(refs[qid]["answer"], prediction)
                if answer["ok"]
                else {"em": 0.0, "f1": 0.0}
            )
            rows.append(
                {
                    "question_id": qid,
                    "table_id": question["table_id"],
                    "question": question["question"],
                    "plan": plan,
                    "graph_covered": retrieval["graph_covered"],
                    "trace_category": trace_category,
                    "ok": answer["ok"],
                    "pending": answer.get("pending", False),
                    "prediction": prediction,
                    "reference": refs[qid]["answer"],
                    **scores,
                    "abstain": answer.get("abstain", False),
                    "context_tokens": ctx["tokens"],
                    "reader_seconds": answer.get("seconds"),
                    "citation_count": len(answer.get("citations", [])),
                    **evidence_diagnostics(refs[qid], ctx["evidence"]),
                }
            )
    ledger_path = output / "api_calls.jsonl"
    calls = (
        [json.loads(line) for line in ledger_path.read_text().splitlines() if line]
        if ledger_path.exists()
        else []
    )

    def aggregate(group):
        n = len(group)
        if not n:
            return {"questions": 0}
        latency = [
            r["reader_seconds"] for r in group if r["reader_seconds"] is not None
        ]
        traced = [r for r in group if r["weak_trace_any_hit"] is not None]
        return {
            "questions": n,
            "successful_answers": sum(r["ok"] for r in group),
            "pending": sum(r["pending"] for r in group),
            "failed": sum(not r["ok"] and not r["pending"] for r in group),
            "em_percent": 100 * sum(r["em"] for r in group) / n,
            "f1_percent": 100 * sum(r["f1"] for r in group) / n,
            "f1_95ci_percent": [
                100 * v
                for v in cluster_interval(
                    [r["f1"] for r in group], [r["table_id"] for r in group]
                )
            ],
            "abstention_percent": 100 * sum(r["abstain"] for r in group) / n,
            "answer_string_presence_percent": 100
            * sum(r["answer_string_in_context"] for r in group)
            / n,
            "weak_trace_any_hit_percent": 100
            * sum(r["weak_trace_any_hit"] for r in traced)
            / len(traced)
            if traced
            else None,
            "mean_context_tokens": sum(r["context_tokens"] for r in group) / n,
            "reader_latency_p50_seconds": float(np.median(latency))
            if latency
            else None,
            "reader_latency_p95_seconds": float(np.percentile(latency, 95))
            if latency
            else None,
        }

    results = {}
    for plan in plans:
        group = [r for r in rows if r["plan"] == plan]
        results[plan] = aggregate(group)
        results[plan]["by_graph_coverage"] = {
            str(covered).lower(): aggregate(
                [r for r in group if r["graph_covered"] == covered]
            )
            for covered in (True, False)
        }
        results[plan]["by_trace_category"] = {
            category: aggregate([r for r in group if r["trace_category"] == category])
            for category in sorted({r["trace_category"] for r in group})
        }
        results[plan]["estimated_reader_usd"] = sum(
            c["accounted_usd"] for c in calls if c["plan"] == plan
        )
    comparisons = {}
    if "all_stores" in plans:
        all_rows = {r["question_id"]: r for r in rows if r["plan"] == "all_stores"}
        for plan in plans:
            if plan == "all_stores":
                continue
            pairs = [
                (all_rows[r["question_id"]], r)
                for r in rows
                if r["plan"] == plan
                and not r["pending"]
                and r["question_id"] in all_rows
                and not all_rows[r["question_id"]]["pending"]
            ]
            differences = [a["f1"] - b["f1"] for a, b in pairs]
            comparisons[plan] = {
                "paired_questions": len(pairs),
                "all_stores_minus_baseline_f1_points": 100 * float(np.mean(differences))
                if pairs
                else None,
                "paired_table_bootstrap_95ci_points": [
                    100 * v
                    for v in cluster_interval(
                        differences, [a["table_id"] for a, b in pairs]
                    )
                ]
                if pairs
                else None,
            }
    complete = len(rows) == len(questions) * len(plans) and all(
        not r["pending"] for r in rows
    )
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_signature": manifest["signature"],
        "complete": complete,
        "all_calls_successful": complete and all(r["ok"] for r in rows),
        "requested_questions": len(questions),
        "retrieved_questions": len(retrievals),
        "unique_tables": len({q["table_id"] for q in questions}),
        "graph_covered_questions": sum(r["graph_covered"] for r in retrievals),
        "graph_sql_topology_mismatches": sum(
            r["graph_sql_topology_equal"] is False for r in retrievals
        ),
        "source_audits_passed": all(r["source_audit"]["passed"] for r in retrievals),
        "api_requests": len(calls),
        "estimated_reader_usd": sum(c["accounted_usd"] for c in calls),
        "api_input_tokens": sum(c["usage"].get("input_tokens", 0) for c in calls),
        "api_output_tokens": sum(c["usage"].get("output_tokens", 0) for c in calls),
        "physical_retrieval_calls": dict(
            sum(
                (collections.Counter(r["calls"]) for r in retrievals),
                collections.Counter(),
            )
        ),
        "results": results,
        "paired_comparisons": comparisons,
        "limitations": [
            "Training pilot, not held-out test results; one generation per question/plan.",
            "Provided-table protocol. Exact cosine search over linked vectors fetched from Pinecone; not corpus-wide ANN.",
            "Vector-only also uses PostgreSQL to resolve complete candidate membership; it exposes only passages to the reader.",
            "Shared retrieval is materialized once across plans; timings do not estimate independent deployed plan latency.",
            "Source auditing and batched embedding startup are harness overhead, not production retrieval latency.",
            "Answer-string presence and weak trace hits do not establish sufficient or complete supporting evidence.",
            "Citation IDs are validated, but citation entailment has not been human-graded.",
            "API cost is usage-derived at recorded list prices; hosting, database and local compute costs are excluded.",
            "Confidence intervals are exploratory table-cluster bootstrap intervals, without multiplicity adjustment.",
        ]
        + manifest.get("limitations", []),
    }
    write_json(output / "summary.json", summary)
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(
        json.dumps(
            {
                "stage": "report",
                "complete": complete,
                "all_calls_successful": summary["all_calls_successful"],
                "questions": len(questions),
                "estimated_reader_usd": summary["estimated_reader_usd"],
                "results": {
                    p: {
                        k: r.get(k)
                        for k in ["em_percent", "f1_percent", "failed", "pending"]
                    }
                    for p, r in results.items()
                },
            }
        ),
        flush=True,
    )
    return summary
