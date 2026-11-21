"""Validate and summarize the published experiment without external services.

The public package retains main evaluation outcomes only. Its two phases use
different question cohorts and must never be pooled into one accuracy estimate.
"""

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

from baseline.common import answer_scores

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = REPO / "results/heterorag"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(
        math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10),
        f"Metric mismatch: {label}: {actual} != {expected}",
    )


def aggregate(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["phase"], row["condition"]].append(row)
    return [
        {
            "phase": phase,
            "condition": condition,
            "questions": len(group),
            "failed": sum(row["status"] == "failed" for row in group),
            "em_percent": 100 * sum(float(row["em"]) for row in group) / len(group),
            "f1_percent": 100 * sum(float(row["f1"]) for row in group) / len(group),
            "mean_data_requests": sum(float(row["data_requests"]) for row in group)
            / len(group),
        }
        for (phase, condition), group in groups.items()
    ]


def verify(package=DEFAULT_PACKAGE):
    package = Path(package).resolve()
    manifest = read_json(package / "manifest.json")
    for name, digest in manifest["files"].items():
        path = (package / name).resolve()
        require(path.is_relative_to(package), f"Invalid package path: {name}")
        require(path.is_file(), f"Missing package file: {name}")
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == digest,
            f"Checksum mismatch: {name}",
        )
    definition = read_json(package / "experiment.json")
    analysis = read_json(package / "analysis.json")
    rows = read_csv(package / "outcomes.csv")
    require(len(rows) == definition["outcomes"] == 6000, "Expected 6,000 outcomes")
    seen, cohorts = set(), {}
    indexed = collections.defaultdict(dict)
    for row in rows:
        phase, condition, qid = row["phase"], row["condition"], row["question_id"]
        require(phase in definition["phases"], f"Unexpected phase: {phase}")
        spec = definition["phases"][phase]
        require(condition in spec["conditions"], f"Unexpected condition: {condition}")
        require(qid in spec["question_ids"], f"Unexpected question: {qid}")
        key = phase, condition, qid
        require(key not in seen, f"Duplicate outcome: {key}")
        seen.add(key)
        indexed[phase, condition][qid] = row
        require(row["source_phase"] == "evaluation", f"Non-main outcome: {key}")
        require(row["status"] in {"completed", "failed", "empty_evidence"}, key)
        for field in ("em", "f1"):
            value = float(row[field])
            require(math.isfinite(value) and 0 <= value <= 1, f"Invalid {field}: {key}")
        scores = (
            {"em": 0, "f1": 0}
            if row["status"] == "failed"
            else answer_scores(row["reference"], row["prediction"])
        )
        for field, value in scores.items():
            close(row[field], value, str(key) + " " + field)
        require(
            row["postgres_links"] == spec["conditions"][condition]["postgres_links"],
            f"Link setting mismatch: {key}",
        )
        require(len(row["source_result_sha256"]) == 64, f"Missing provenance: {key}")
    for phase, spec in definition["phases"].items():
        expected = set(spec["question_ids"])
        require(len(expected) == spec["questions"] == 300, f"Invalid cohort: {phase}")
        for condition in spec["conditions"]:
            require(
                set(indexed[phase, condition]) == expected, f"Incomplete {condition}"
            )
        phase_rows = [row for row in rows if row["phase"] == phase]
        require(len(phase_rows) == spec["outcomes"], f"Wrong phase count: {phase}")
        require(
            sum(row["status"] == "failed" for row in phase_rows) == spec["failed"] == 7,
            f"Failure denominator changed: {phase}",
        )
        first = indexed[phase, next(iter(spec["conditions"]))]
        require(
            sum(row["graph_covered_snapshot"] == "True" for row in first.values())
            == spec["graph_covered_questions"],
            f"Graph coverage mismatch: {phase}",
        )
        for condition in spec["conditions"]:
            for qid, row in indexed[phase, condition].items():
                for field in ("table_id", "reference", "graph_covered_snapshot"):
                    require(
                        row[field] == first[qid][field], f"Paired input mismatch: {qid}"
                    )
        cohorts[phase] = expected
    require(
        not cohorts["routing_fusion"] & cohorts["link_access"],
        "The two question cohorts must be disjoint",
    )
    trace_keys = set()
    with (package / "evidence-traces.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            trace = json.loads(line)
            key = trace["phase"], trace["condition"], trace["question_id"]
            require(key in seen and key not in trace_keys, f"Unexpected trace: {key}")
            trace_keys.add(key)
            row = indexed[key[0], key[1]][key[2]]
            for field in ("job_id", "source_result_sha256"):
                require(trace[field] == row[field], f"Trace provenance mismatch: {key}")
    require(trace_keys == seen, "Missing evidence traces")
    failed_keys = {
        (row["phase"], row["condition"], row["question_id"])
        for row in rows
        if row["status"] == "failed"
    }
    reported_failures = {
        (row["phase"], row["condition"], row["question_id"])
        for row in analysis["failures"]
    }
    require(reported_failures == failed_keys, "Failure inventory mismatch")
    summaries = aggregate(rows)
    exported = {
        (row["phase"], row["condition"]): row
        for row in read_csv(package / "condition-metrics.csv")
    }
    require(len(exported) == len(summaries) == 20, "Expected 20 condition summaries")
    for row in summaries:
        phase, condition = row["phase"], row["condition"]
        report = analysis["phases"][phase]["results"][condition]
        for field in ("em_percent", "f1_percent", "mean_data_requests"):
            close(row[field], report[field], f"{phase}/{condition}/{field}")
        for field in (
            "questions",
            "failed",
            "em_percent",
            "f1_percent",
            "mean_data_requests",
        ):
            close(
                row[field],
                exported[phase, condition][field],
                f"export/{condition}/{field}",
            )
    for phase, report in analysis["phases"].items():
        for pair, result in report["paired_comparisons"].items():
            left, right = pair.split(" -> ")
            qids = definition["phases"][phase]["question_ids"]
            delta = sum(
                float(indexed[phase, right][qid]["f1"])
                - float(indexed[phase, left][qid]["f1"])
                for qid in qids
            ) / len(qids)
            close(delta * 100, result["f1_right_minus_left_points"], pair)
    for stratum, result in analysis["link_access_interaction"].items():
        qids = definition["phases"]["link_access"]["question_ids"]
        values = []
        for qid in qids:
            covered = (
                indexed["link_access", "fixed_all_links_on"][qid][
                    "graph_covered_snapshot"
                ]
                == "True"
            )
            if stratum != "all" and covered != (stratum == "graph_covered"):
                continue
            values.append(
                float(indexed["link_access", "fixed_all_links_off"][qid]["f1"])
                - float(indexed["link_access", "fixed_pgpc_links_off"][qid]["f1"])
                - float(indexed["link_access", "fixed_all_links_on"][qid]["f1"])
                + float(indexed["link_access", "fixed_pgpc_links_on"][qid]["f1"])
            )
        close(len(values), result["questions"], stratum)
        close(
            100 * sum(values) / len(values),
            result["graph_increment_off_minus_on_f1_points"],
            stratum + " interaction",
        )
    return {
        "experiment": definition["experiment"],
        "outcomes": len(rows),
        "distinct_questions": len(set.union(*cohorts.values())),
        "conditions": len(summaries),
        "retained_failures": sum(row["status"] == "failed" for row in rows),
        "verified_files": len(manifest["files"]),
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "summarize"))
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    result = verify(args.package)
    if args.command == "summarize":
        result["condition_metrics"] = aggregate(read_csv(args.package / "outcomes.csv"))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
