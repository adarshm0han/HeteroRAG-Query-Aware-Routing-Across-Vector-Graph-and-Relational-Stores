"""Complete-denominator evaluation reports, validation selection, and repeated timing."""

import argparse
import csv
from pathlib import Path

from baseline.common import object_hash, read_json
from evaluation.checkpoints import atomic_json, load, save
from evaluation.metrics import aggregate, fusion_controls, paired, score_question
from evaluation.protocol import verify


def compact_result(result):
    """Keep reporting fields; full candidate snapshots stay in per-job artifacts.

    Holding 41,568 full contexts in memory would require many gigabytes. Preserve
    the full artifact hash before dropping prose and reader-generation traces.
    """
    compact = dict(result)
    compact["artifact_sha256"] = object_hash(result)
    evidence = result.get("retrieval", {})
    compact["retrieval"] = {
        key: value
        for key, value in evidence.items()
        if key
        not in ("candidate_records", "context", "items", "evidence", "sql", "graph")
    }
    if evidence:
        compact["retrieval"].update(
            items=[
                {k: i[k] for k in ("id", "kind")} for i in evidence.get("items", [])
            ],
            evidence=[
                {"source_id": i["source_id"]} for i in evidence.get("evidence", [])
            ],
            sql={
                "planner": {
                    "seconds": (evidence.get("sql") or {})
                    .get("planner", {})
                    .get("seconds")
                }
            },
        )
    compact["answer"] = {
        k: v
        for k, v in result.get("answer", {}).items()
        if k in ("answer", "citations", "abstain", "seconds")
    }
    compact["database_calls"] = [
        {
            k: v
            for k, v in call.items()
            if k
            in (
                "store",
                "kind",
                "status",
                "id_count",
                "read_units",
                "seconds",
                "response_bytes_estimate",
                "returned_records",
            )
        }
        for call in result["database_calls"]
    ]
    return compact


def phase_results(bundle, root, phase):
    protocol, jobs = verify(bundle)
    selected = [j for j in jobs if j["phase"] == phase]
    if not selected:
        raise ValueError("Phase is not in this study")
    source = "timing" if phase.startswith("timing-") else phase
    questions = read_json(bundle / source / "questions.json")
    by_id = {q["question_id"]: q for q in questions}
    results = {}
    for job in selected:
        path = root / "jobs" / job["id"] / "result.json"
        if not path.exists():
            raise ValueError("Phase incomplete; references and scores remain unopened")
        result = load(path)
        identity = load(path.parent / "identity.json")
        expected_identity = {
            "job": job,
            "question": by_id[job["question_id"]],
            "config": {
                **protocol["config"],
                "fusion": protocol["conditions"][job["condition"]]["fusion"],
            },
        }
        if identity != expected_identity:
            raise ValueError("Result checkpoint configuration differs from frozen job")
        if (
            result["question"] != by_id[job["question_id"]]
            or result["job_id"] != job["id"]
            or result["status"] not in ("completed", "empty_evidence", "failed")
        ):
            raise ValueError("Result identity or outcome invalid")
        if result["status"] != "failed" and (
            not result.get("source_audit", {}).get("passed")
            or result.get("catalog_sha256") != protocol["routing_catalog"]["sha256"]
        ):
            raise ValueError("Missing source audit or mismatched catalog")
        condition = protocol["conditions"][job["condition"]]
        if condition["option"] != "decisions" and result.get("decision", {}).get(
            "option"
        ) not in (None, condition["option"]):
            raise ValueError("Fixed condition changed its route")
        if job["mode"] == "fusion_replay" and result["database_calls"]:
            raise ValueError("Controlled fusion contacted a database")
        results.setdefault(job["condition"], []).append(compact_result(result))
    for name, group in results.items():
        mapping = {r["question"]["question_id"]: r for r in group}
        if len(mapping) != len(questions):
            raise ValueError("Missing or duplicate question")
        results[name] = [mapping[q["question_id"]] for q in questions]
    return protocol, source, results


def selection_for(protocol, summaries, raw):
    scores = {
        name: {"f1": v["f1_percent"], "requests": v["mean_data_requests"]}
        for name, v in summaries.items()
        if name.startswith("fixed_")
    }
    best = min(
        scores, key=lambda name: (-scores[name]["f1"], scores[name]["requests"], name)
    )
    return {
        "bundle_id": protocol["bundle_id"],
        "condition": best,
        "scores": scores,
        "validation_results_sha256": object_hash(raw),
    }


def report(bundle, root, phase):
    protocol, source, raw = phase_results(bundle, root, phase)
    manifest = load(root / "study.json")
    if manifest["bundle_id"] != protocol["bundle_id"]:
        raise ValueError("Results belong to another study")
    # All requested results have been checked before opening any reference answers.
    references = read_json(bundle / source / "references.json")
    coverage = set(read_json(bundle / "graph-table-ids.json"))
    rows, summary = {}, {}
    for condition, group in raw.items():
        rows[condition] = []
        for result in group:
            row = score_question(
                result,
                references[result["question"]["question_id"]],
                result["question"]["table_id"] in coverage,
            )
            row.update(
                timing_eligible=result["timing_eligible"],
                retrieval_replayed=result["retrieval_replayed"],
                raw_option=result.get("decision", {}).get("raw_option"),
                pinecone_response_bytes_estimate=sum(
                    call.get("response_bytes_estimate", 0)
                    for call in result["database_calls"]
                    if call["store"] == "pinecone"
                ),
                uncertain_database_operations=sum(
                    call["status"] == "started" for call in result["database_calls"]
                ),
            )
            if not result["timing_eligible"]:
                for key in (
                    "total_seconds",
                    "service_seconds",
                    "routing_seconds",
                    "retrieval_seconds",
                    "sql_planner_seconds",
                    "embedding_seconds",
                    "fusion_seconds",
                    "answer_seconds",
                ):
                    row[key] = None
            rows[condition].append(row)
        summary[condition] = aggregate(rows[condition])
        summary[condition]["pinecone_response_bytes_estimate"] = sum(
            r["pinecone_response_bytes_estimate"] for r in rows[condition]
        )
        summary[condition]["uncertain_database_operations"] = sum(
            r["uncertain_database_operations"] for r in rows[condition]
        )
        summary[condition]["live_latency_eligible"] = all(
            r["timing_eligible"] for r in rows[condition]
        )
        summary[condition]["timing_eligible_questions"] = sum(
            r["timing_eligible"] for r in rows[condition]
        )
        summary[condition]["by_graph_coverage"] = {
            str(flag).lower(): aggregate(
                [r for r in rows[condition] if r["graph_covered_snapshot"] == flag]
            )
            for flag in (True, False)
        }
        if any(r["retrieval_replayed"] for r in rows[condition]):
            summary[condition]["request_interpretation"] = (
                "Observed replay execution only; no retrieval-efficiency or live-latency claim"
            )
    comparisons, controls = {}, {}
    pairs = list(protocol["primary_pairs"])
    if phase == "evaluation":
        selection = load(root / "best-fixed.json")
        validation = load(root / "reports" / "validation.json")
        if selection != selection_for(
            protocol,
            validation["results"],
            phase_results(bundle, root, "validation")[2],
        ):
            raise ValueError("Best-fixed selection changed after validation")
        pairs.append([selection["condition"], "routed_rrf"])
    if source != "timing":
        for left, right in pairs:
            measured = paired(
                rows[left], rows[right], protocol["noninferiority_margin_f1_points"]
            )
            measured["timing_comparison_eligible"] = all(
                r["timing_eligible"] for r in rows[left] + rows[right]
            )
            if any(r["retrieval_replayed"] for r in rows[left] + rows[right]):
                for key in (
                    "data_requests_right_minus_left",
                    "data_requests_paired_95ci",
                    "data_request_reduction_percent",
                    "lower_requests_and_within_quality_margin",
                ):
                    measured[key] = None
                measured["request_claim_disabled"] = (
                    "Fusion reuses prior retrieval; only answer-quality differences are comparable"
                )
            comparisons[left + " -> " + right] = measured
        controls = {
            "weighted": fusion_controls(raw["routed_rrf"], raw["routed_weighted"]),
            "concat": fusion_controls(raw["fixed_all"], raw["all_concat"]),
            "routed_concat": fusion_controls(raw["routed_rrf"], raw["routed_concat"]),
        }
        for dependency, replay in (
            ("routed_rrf", "routed_weighted"),
            ("fixed_all", "all_concat"),
            ("routed_rrf", "routed_concat"),
        ):
            for a, b in zip(raw[dependency], raw[replay]):
                if (
                    b["status"] != "failed"
                    and b.get("replay_source", {}).get("sha256") != a["artifact_sha256"]
                ):
                    raise ValueError(
                        "Fusion replay does not bind to the matching source outcome"
                    )
    payload = {
        "bundle_id": protocol["bundle_id"],
        "phase": phase,
        "results": summary,
        "paired_comparisons": comparisons,
        "fusion_controls": controls,
        "results_sha256": object_hash(raw),
        "gold_evidence_precision_recall": None,
        "notes": [
            "All scheduled questions including failures are scored; unattempted questions block reporting.",
            "Weak evidence overlap is an automatic trace proxy, not gold evidence precision or recall.",
            "The guarded router includes the deterministic evidence policy. Raw ablation shares the paid routing decision.",
            "Resumed or replayed outcomes are excluded from latency distributions; denominators are explicit.",
            "Timing repeats are correlated, with natural warm caches. Live timing routes may differ; no pure-fusion quality claim.",
            "Intervals use the prespecified paired table-cluster bootstrap; no multiplicity correction.",
        ],
    }
    save(root / "reports" / (phase + ".json"), payload)
    if phase == "validation":
        selection = selection_for(protocol, summary, raw)
        selection_path = root / "best-fixed.json"
        if selection_path.exists() and load(selection_path) != selection:
            raise ValueError("Refusing to replace the frozen validation selection")
        save(selection_path, selection)
    output = root / "reports"
    flat = [{"condition": name, **row} for name, group in rows.items() for row in group]
    with (output / (phase + "-questions.csv")).open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = [
        "# Paper evaluation: " + phase,
        "",
        "| Condition | EM % | F1 % | Failed % | Requests / question | Eligible timing n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in summary.items():
        requests = (
            "replay; no efficiency claim"
            if "request_interpretation" in value
            else f"{value['mean_data_requests']:.2f}"
        )
        lines.append(
            f"| {name} | {value['em_percent']:.2f} | {value['f1_percent']:.2f} | {value['failed_percent']:.2f} | {requests} | {value['timing_eligible_questions']} |"
        )
    lines += ["", *payload["notes"]]
    (output / (phase + ".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def timing_report(bundle, root):
    protocol, _ = verify(bundle)
    reports = {
        str(i): report(bundle, root, f"timing-{i}")
        for i in range(1, protocol["timing"]["repeats"] + 1)
    }
    payload = {
        "repeats": reports,
        "interpretation": "Compare each repeat; no pooled independence assumption or pure fusion claim. Interrupted outcomes are timing-ineligible.",
    }
    atomic_json(root / "reports" / "timing.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    if args.phase == "timing":
        timing_report(args.bundle, args.output)
    else:
        report(args.bundle, args.output, args.phase)
    print("Complete-phase report written")


if __name__ == "__main__":
    main()
