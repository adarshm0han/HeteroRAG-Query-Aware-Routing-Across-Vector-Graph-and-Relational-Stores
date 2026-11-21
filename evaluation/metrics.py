"""Offline, paired evaluation of frozen runs. Never calls models or databases."""

import argparse
import collections
import csv
import statistics
from pathlib import Path

from baseline.common import answer_scores, object_hash, read_json, sha256, write_json
from baseline.report import cluster_interval
from evaluation.cohorts import CONDITIONS, load_bundle
from routing.report import percentile


def evidence_metrics(ids, relevant):
    selected, wanted = set(ids), set(relevant)
    if not wanted:
        return {"recall": None, "precision": None, "any_hit": None}
    hits = len(selected & wanted)
    return {
        "recall": hits / len(wanted),
        "precision": hits / len(selected) if selected else 0.0,
        "any_hit": float(hits > 0),
    }


def score_question(result, reference, graph_covered, manual=None):
    retrieval = result.get("retrieval", {})
    answer = result.get("answer", {})
    failed = result["status"] == "failed"
    scores = (
        {"em": 0.0, "f1": 0.0}
        if failed
        else answer_scores(reference["answer"], answer["answer"])
    )
    ranked_ids = [i["id"] for i in retrieval.get("items", [])]
    packed_ids = [i["source_id"] for i in retrieval.get("evidence", [])]
    weak = reference.get("weak_evidence_ids", [])
    calls = result.get("database_calls", [])
    elapsed = result["total_seconds"]
    manual_complete = bool(
        manual and manual.get("complete") is True and manual.get("reviewer", "").strip()
    )
    if manual_complete and not manual.get("evidence_ids"):
        raise ValueError("Completed manual evidence labels cannot be empty")
    row = {
        **result["question"],
        **scores,
        "status": result["status"],
        "prediction": answer.get("answer", ""),
        "reference": reference["answer"],
        "graph_covered_snapshot": graph_covered,
        "graph_covered_observed": retrieval.get("graph_covered"),
        "failed": int(failed),
        "failed_stage": result.get("failed_stage"),
        "abstained": int(not failed and bool(answer.get("abstain"))),
        "empty_evidence": int(result["status"] == "empty_evidence"),
        "option": result.get("decision", {}).get("option"),
        "selected_stores": len(result.get("selected_stores", [])),
        "data_requests": sum(c["kind"] == "data" for c in calls),
        "candidate_passages": retrieval.get("candidate_passage_count"),
        "candidate_passage_text_bytes": retrieval.get("candidate_passage_text_bytes"),
        "pinecone_fetch_ids": sum(
            c.get("id_count", 0) for c in calls if c["store"] == "pinecone"
        ),
        "control_requests": sum(c["kind"] == "control" for c in calls),
        "connection_attempts": sum(c["kind"] == "connection" for c in calls),
        "database_failures": sum(c["status"] == "failed" for c in calls),
        "pinecone_read_units": sum(c.get("read_units", 0) for c in calls),
        "decisions_requests": len(result.get("decisions_calls", [])),
        "replayed_route": int(
            result.get("route_replayed", False)
            or result.get("decision", {}).get("method") == "recorded_decisions"
        ),
        "total_seconds": elapsed,
        "source_audit_seconds": result.get("source_audit_seconds", 0.0),
        "service_seconds": max(0.0, elapsed - result.get("source_audit_seconds", 0.0)),
        "routing_seconds": result.get("decision", {}).get("seconds"),
        "retrieval_seconds": retrieval.get("seconds"),
        "sql_planner_seconds": (retrieval.get("sql") or {})
        .get("planner", {})
        .get("seconds"),
        "computed_evidence_items": sum(
            i["kind"] == "aggregate" for i in retrieval.get("items", [])
        ),
        "embedding_seconds": retrieval.get("embedding_seconds"),
        "fusion_seconds": retrieval.get("fusion_seconds"),
        "answer_seconds": answer.get(
            "seconds", result.get("failed_call", {}).get("seconds")
        ),
        "context_tokens": retrieval.get("context_tokens", 0),
        "packed_items": len(packed_ids),
        "manual_annotation_complete": manual_complete,
    }
    for stage, ids in (("retrieved", ranked_ids), ("packed", packed_ids)):
        row.update(
            {
                f"weak_{stage}_{key}": value
                for key, value in evidence_metrics(ids, weak).items()
            }
        )
        measured = (
            evidence_metrics(ids, manual["evidence_ids"])
            if manual_complete
            else {"recall": None, "precision": None, "any_hit": None}
        )
        row.update({f"manual_{stage}_{key}": value for key, value in measured.items()})
    for store in ("postgres", "neo4j", "pinecone"):
        row[store + "_data_requests"] = sum(
            c["store"] == store and c["kind"] == "data" for c in calls
        )
    return row


def average(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def aggregate(rows):
    if not rows:
        return {"questions": 0}
    summary = {"questions": len(rows), "tables": len({r["table_id"] for r in rows})}
    for name in ("em", "f1", "failed", "abstained", "empty_evidence"):
        summary[name + "_percent"] = 100 * average([r[name] for r in rows])
    summary["f1_table_cluster_95ci_points"] = [
        100 * v
        for v in cluster_interval(
            [r["f1"] for r in rows], [r["table_id"] for r in rows]
        )
    ]
    for name in (
        "data_requests",
        "control_requests",
        "connection_attempts",
        "selected_stores",
        "pinecone_read_units",
        "context_tokens",
        "packed_items",
        "postgres_data_requests",
        "neo4j_data_requests",
        "pinecone_data_requests",
    ):
        summary["mean_" + name] = average([r[name] for r in rows])
    for name in ("database_failures", "decisions_requests", "replayed_route"):
        summary[name] = sum(r[name] for r in rows)
    for name in (
        "candidate_passages",
        "candidate_passage_text_bytes",
        "pinecone_fetch_ids",
    ):
        values = [r.get(name) for r in rows if r.get(name) is not None]
        summary["mean_" + name] = average(values) if values else None
    summary["failed_stages"] = dict(
        collections.Counter(r["failed_stage"] for r in rows if r["failed"])
    )
    summary["options"] = dict(
        collections.Counter(r["option"] or "routing_failed" for r in rows)
    )
    summary["observed_graph_snapshot_mismatches"] = sum(
        r["graph_covered_observed"] is not None
        and r["graph_covered_observed"] != r["graph_covered_snapshot"]
        for r in rows
    )
    for label in ("weak", "manual"):
        for stage in ("retrieved", "packed"):
            for metric in ("recall", "precision", "any_hit"):
                name = f"{label}_{stage}_{metric}"
                values = [r[name] for r in rows if r[name] is not None]
                summary[name + "_percent"] = 100 * average(values) if values else None
                summary[name + "_questions"] = len(values)
    summary["latency"] = {}
    for name in (
        "total_seconds",
        "service_seconds",
        "routing_seconds",
        "retrieval_seconds",
        "sql_planner_seconds",
        "embedding_seconds",
        "fusion_seconds",
        "answer_seconds",
    ):
        values = [r[name] for r in rows if r[name] is not None]
        successful = [r[name] for r in rows if not r["failed"] and r[name] is not None]
        summary["latency"][name] = {
            "observations": len(values),
            "mean": average(values),
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "successful_observations": len(successful),
            "successful_p50": percentile(successful, 0.5),
            "successful_p95": percentile(successful, 0.95),
        }
    summary["live_latency_eligible"] = not any(r["replayed_route"] for r in rows)
    return summary


def paired(left, right, margin=2.0):
    if [r["question_id"] for r in left] != [r["question_id"] for r in right]:
        raise ValueError("Paired comparisons require identical question order")
    if not left:
        return {"questions": 0}
    differences = [b["f1"] - a["f1"] for a, b in zip(left, right)]
    interval = [
        100 * v for v in cluster_interval(differences, [r["table_id"] for r in left])
    ]
    baseline_calls = average([r["data_requests"] for r in left])
    new_calls = average([r["data_requests"] for r in right])
    return {
        "questions": len(left),
        "f1_right_minus_left_points": 100 * average(differences),
        "paired_table_cluster_95ci_points": interval,
        "data_requests_right_minus_left": new_calls - baseline_calls,
        "data_requests_paired_95ci": cluster_interval(
            [b["data_requests"] - a["data_requests"] for a, b in zip(left, right)],
            [r["table_id"] for r in left],
        ),
        "data_request_reduction_percent": 100
        * (baseline_calls - new_calls)
        / baseline_calls
        if baseline_calls
        else None,
        "within_prespecified_quality_margin": interval[0] >= -margin,
        "lower_requests_and_within_quality_margin": new_calls < baseline_calls
        and interval[0] >= -margin,
        "timing_comparison_eligible": not any(
            r["replayed_route"] for r in left + right
        ),
    }


def load_run(path, protocol, cohort, condition, questions):
    manifest = read_json(path / "manifest.json")
    tag = manifest.get("evaluation") or {}
    if (tag.get("bundle_id"), tag.get("cohort"), tag.get("condition")) != (
        protocol["bundle_id"],
        cohort,
        condition,
    ):
        raise ValueError("Run does not match the frozen bundle/cohort/condition")
    expected_config = {**protocol["config"], "fusion": CONDITIONS[condition]["fusion"]}
    if (
        manifest["config"] != expected_config
        or manifest["option"] != CONDITIONS[condition]["option"]
    ):
        raise ValueError("Run configuration differs from the protocol")
    normalized_code = {
        name.replace("\\", "/"): digest for name, digest in manifest["code"].items()
    }
    if (
        normalized_code != protocol["code"]
        or manifest["reader"]["digest"] != protocol["reader_digest"]
    ):
        raise ValueError("Runs differ from frozen code or local reader")
    if read_json(path / "questions.json") != questions:
        raise ValueError("Run questions differ from the frozen cohort")
    if not manifest.get("verify_sources") or manifest.get("graph_catalog_sha256"):
        raise ValueError("Run violated source-audit or routing-availability protocol")
    expected_catalog = (protocol.get("routing_catalog") or {}).get("sha256")
    if manifest.get("routing_catalog_sha256") != expected_catalog:
        raise ValueError("Run used different public routing metadata")
    results = [read_json(p) for p in sorted((path / "results").glob("*.json"))]
    by_id = {r["question"]["question_id"]: r for r in results}
    if len(by_id) != len(results) or set(by_id) != {
        q["question_id"] for q in questions
    }:
        raise ValueError(
            "Incomplete or duplicate results; missing questions cannot be silently scored or dropped"
        )
    ordered = [by_id[q["question_id"]] for q in questions]
    for q, row in zip(questions, ordered):
        if row["question"] != q or row["status"] not in (
            "completed",
            "empty_evidence",
            "failed",
        ):
            raise ValueError("Invalid result identity or status")
        if row["status"] != "failed" and not row.get("source_audit", {}).get("passed"):
            raise ValueError("Successful result lacks a passed source audit")
        if row["status"] != "failed" and row.get("catalog_sha256") != expected_catalog:
            raise ValueError("Result used different public routing metadata")
        decision = row.get("decision", {})
        if (
            row["status"] != "failed"
            and CONDITIONS[condition]["option"] != "decisions"
            and decision.get("option") != CONDITIONS[condition]["option"]
        ):
            raise ValueError("Fixed baseline changed its route")
        if (
            decision
            and condition.startswith("routed_")
            and decision.get("weight_method") != protocol["fusion_weight_method"]
        ):
            raise ValueError("Routing weights differ from the frozen method")
    return manifest, ordered


def fusion_controls(left, right):
    mismatches, unavailable = [], []
    for a, b in zip(left, right):
        qid = a["question"]["question_id"]
        if a.get("decision", {}).get("option") != b.get("decision", {}).get("option"):
            mismatches.append({"question_id": qid, "field": "route"})
        if not a.get("retrieval") or not b.get("retrieval"):
            unavailable.append(qid)
            continue
        for key in ("candidate_evidence_sha256", "rankings_sha256", "title"):
            if (
                key not in a["retrieval"]
                or key not in b["retrieval"]
                or a["retrieval"][key] != b["retrieval"][key]
            ):
                mismatches.append({"question_id": qid, "field": key})
        if a.get("decision", {}).get("probabilities") != b.get("decision", {}).get(
            "probabilities"
        ):
            mismatches.append({"question_id": qid, "field": "routing_probabilities"})
    return {
        "matched": not mismatches and not unavailable,
        "mismatches": mismatches,
        "unverifiable_questions": unavailable,
        "pure_fusion_claim_allowed": not mismatches and not unavailable,
    }


def compare(
    bundle, cohort, run_map, output, validation_selection=None, manual_path=None
):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new report directory")
    protocol = load_bundle(bundle)
    if cohort not in ("smoke", "development", "validation", "evaluation"):
        raise ValueError("Use timing_report for repeated timing runs")
    if set(run_map) != set(CONDITIONS):
        raise ValueError("Comparison requires all ten frozen conditions")
    questions = read_json(bundle / cohort / "questions.json")
    references = read_json(bundle / cohort / "references.json")
    coverage = set(read_json(bundle / "graph-table-ids.json"))
    manual = read_json(manual_path) if manual_path else {}
    allowed_audit = set(read_json(bundle / "manual-audit-template.json"))
    if manual and (cohort != "development" or not set(manual) <= allowed_audit):
        raise ValueError("Manual audit must use the prespecified development sample")
    raw, rows, manifests, provenance = {}, {}, {}, {}
    for name in CONDITIONS:
        path = Path(run_map[name])
        manifests[name], raw[name] = load_run(path, protocol, cohort, name, questions)
        rows[name] = [
            score_question(
                r,
                references[r["question"]["question_id"]],
                r["question"]["table_id"] in coverage,
                manual.get(r["question"]["question_id"]),
            )
            for r in raw[name]
        ]
        provenance[name] = {
            "path": str(path.resolve()),
            "manifest_sha256": sha256(path / "manifest.json"),
            "results_sha256": object_hash(raw[name]),
        }
    summary = {
        name: {
            **aggregate(group),
            "startup_seconds": manifests[name]["startup_seconds"],
            "by_graph_coverage": {
                str(flag).lower(): aggregate(
                    [r for r in group if r["graph_covered_snapshot"] == flag]
                )
                for flag in (True, False)
            },
        }
        for name, group in rows.items()
    }
    pairs = list(protocol["primary_pairs"]) + [["all_concat", "fixed_all"]]
    selection = None
    if cohort == "validation":
        fixed = [name for name in CONDITIONS if name.startswith("fixed_")]
        best = min(
            fixed,
            key=lambda name: (
                -summary[name]["f1_percent"],
                summary[name]["mean_data_requests"],
                name,
            ),
        )
        selection = {
            "bundle_id": protocol["bundle_id"],
            "cohort": "validation",
            "condition": best,
            "validation_questions_sha256": protocol["inputs"][
                "validation/questions.json"
            ],
            "fixed_results": {
                name: {
                    "f1": summary[name]["f1_percent"],
                    "calls": summary[name]["mean_data_requests"],
                }
                for name in fixed
            },
            "provenance": provenance,
        }
        selection["sha256"] = object_hash(selection)
    if cohort == "evaluation":
        if validation_selection is None:
            raise ValueError(
                "Final evaluation requires the best-fixed selection from validation"
            )
        selection = read_json(validation_selection)
        unsigned = {k: v for k, v in selection.items() if k != "sha256"}
        if (
            object_hash(unsigned) != selection.get("sha256")
            or selection.get("bundle_id") != protocol["bundle_id"]
            or selection.get("cohort") != "validation"
        ):
            raise ValueError("Invalid validation selection")
        if (
            selection.get("validation_questions_sha256")
            != protocol["inputs"]["validation/questions.json"]
        ):
            raise ValueError("Validation selection used different questions")
        fixed = selection["fixed_results"]
        if set(fixed) != {name for name in CONDITIONS if name.startswith("fixed_")}:
            raise ValueError("Validation did not compare all fixed conditions")
        best = min(
            fixed, key=lambda name: (-fixed[name]["f1"], fixed[name]["calls"], name)
        )
        if selection["condition"] != best:
            raise ValueError("Best fixed condition contradicts the selection rule")
        pairs.append([best, "routed_rrf"])
    controls = fusion_controls(raw["routed_rrf"], raw["routed_weighted"])
    report = {
        "bundle_id": protocol["bundle_id"],
        "cohort": cohort,
        "questions": len(questions),
        "results": summary,
        "provenance": provenance,
        "manual_labels_sha256": sha256(manual_path) if manual_path else None,
        "paired_comparisons": {
            f"{a} -> {b}": paired(rows[a], rows[b]) for a, b in pairs
        },
        "fusion_controls": controls,
        "best_fixed": selection["condition"] if selection else None,
        "notes": [
            "All attempted questions including failures are scored; failures receive zero EM/F1.",
            "Weak precision/recall measure overlap with automatic answer-span candidates, not true evidence precision/recall.",
            "Manual metrics are null until complete reviewer labels are provided; ID validity is not citation entailment.",
            "Retrieved metrics use the fused 8-row/12-passage list; packed metrics use whole evidence items within 6000 tokens.",
            "Latency reports observed failures as well as successful-only samples. Missing stage timings remain null.",
            "Service latency excludes source-audit time; total includes it; startup is separate. Stage timers can overlap.",
            "Saved-route runs cannot establish live routing latency. Use the separate repeated timing protocol.",
            "Pure fusion claims are disabled when routes or candidate evidence differ or a pair cannot be verified.",
            "Intervals are exploratory paired table-cluster bootstrap intervals; no multiplicity correction.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", report)
    if selection and cohort == "validation":
        write_json(output / "best-fixed.json", selection)
    flat = [{"condition": name, **row} for name, group in rows.items() for row in group]
    with (output / "questions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = [
        "# Frozen routing evaluation",
        "",
        f"Cohort: {cohort}; questions: {len(questions)}.",
        "",
        "| Condition | EM % | F1 % | Failed % | Data requests/question | Service p50 s | Service p95 s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in summary.items():
        timing = value["latency"]["service_seconds"]
        suffix = " (replayed routing)" if not value["live_latency_eligible"] else ""
        lines.append(
            f"| {name}{suffix} | {value['em_percent']:.2f} | {value['f1_percent']:.2f} | {value['failed_percent']:.2f} | {value['mean_data_requests']:.2f} | {timing['p50']:.3f} | {timing['p95']:.3f} |"
        )
    lines += [
        "",
        "## Paired F1 differences",
        "",
        "| Comparison | Difference, points | 95% interval |",
        "|---|---:|---|",
    ]
    for name, value in report["paired_comparisons"].items():
        lo, hi = value["paired_table_cluster_95ci_points"]
        lines.append(
            f"| {name} | {value['f1_right_minus_left_points']:+.2f} | {lo:+.2f} to {hi:+.2f} |"
        )
    lines += [
        "",
        f"Fusion evidence controls passed: {controls['matched']}.",
        "",
        "## Interpretation",
        "",
    ]
    lines += ["- " + note for note in report["notes"]]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def timing_report(bundle, run_map, output):
    """Run map: repeat names 1,2,3 -> the four condition-to-directory maps."""
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new report directory")
    protocol = load_bundle(bundle)
    if set(run_map) != {"1", "2", "3"}:
        raise ValueError("Timing requires all three repeats")
    questions = read_json(bundle / "timing/questions.json")
    refs = read_json(bundle / "timing/references.json")
    coverage = set(read_json(bundle / "graph-table-ids.json"))
    repeats, pooled, provenance = {}, collections.defaultdict(list), {}
    seen_paths = set()
    for repeat, paths in run_map.items():
        if set(paths) != set(protocol["timing"]["conditions"]):
            raise ValueError("Timing repeat lacks required conditions")
        repeats[repeat], provenance[repeat] = {}, {}
        for condition, path in paths.items():
            resolved = Path(path).resolve()
            if resolved in seen_paths:
                raise ValueError("Repeated timing must use distinct executions")
            seen_paths.add(resolved)
            manifest, results = load_run(
                resolved, protocol, "timing", condition, questions
            )
            if manifest.get("decision_records") or any(
                r.get("decision", {}).get("method") == "recorded_decisions"
                for r in results
            ):
                raise ValueError("Timing cannot use replayed Decisions")
            rows = [
                score_question(
                    r,
                    refs[r["question"]["question_id"]],
                    r["question"]["table_id"] in coverage,
                )
                for r in results
            ]
            repeats[repeat][condition] = {
                **aggregate(rows),
                "startup_seconds": manifest["startup_seconds"],
            }
            provenance[repeat][condition] = {
                "path": str(resolved),
                "manifest_sha256": sha256(resolved / "manifest.json"),
                "results_sha256": object_hash(results),
            }
            pooled[condition].extend(rows)
    result = {
        "bundle_id": protocol["bundle_id"],
        "repeats": repeats,
        "pooled_descriptive": {name: aggregate(rows) for name, rows in pooled.items()},
        "provenance": provenance,
        "notes": [
            "Repeated observations are correlated; pooled distributions are descriptive.",
            "No cold-start claim. Startup and audit overhead are separate.",
            "Routes can differ across live calls; this measures deployed latency, not a controlled fusion-quality effect.",
        ],
    }
    write_json(output / "timing.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--cohort",
        choices=("smoke", "development", "validation", "evaluation", "timing"),
        required=True,
    )
    parser.add_argument(
        "--runs",
        type=Path,
        required=True,
        help="JSON condition-to-run map; timing nests maps under 1,2,3",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-selection", type=Path)
    parser.add_argument("--manual-labels", type=Path)
    args = parser.parse_args()
    if args.cohort == "timing":
        timing_report(args.bundle, read_json(args.runs), args.output)
    else:
        compare(
            args.bundle,
            args.cohort,
            read_json(args.runs),
            args.output,
            args.validation_selection,
            args.manual_labels,
        )
    print("Offline report written to", args.output)


if __name__ == "__main__":
    main()
