"""Offline preparation of the resumable evaluation study; preparation never executes it."""

import argparse
import json
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path

from baseline.common import (
    REPO,
    object_hash,
    public_question,
    read_json,
    select_questions,
    sha256,
)
from evaluation.checkpoints import atomic_json
from evaluation.cohorts import CONDITIONS as LEGACY_CONDITIONS
from evaluation.cohorts import code_hashes, load_bundle, weak_ids
from routing.pipeline import Config

CONDITIONS = {
    **LEGACY_CONDITIONS,
    "routed_raw": {"option": "decisions", "fusion": "rrf"},
    "routed_concat": {"option": "decisions", "fusion": "concat"},
}
DEFAULT_PHASES = [
    "smoke",
    "validation",
    "evaluation",
    "timing-1",
    "timing-2",
    "timing-3",
]
TIMING_CONDITIONS = [
    "fixed_all",
    "fixed_postgres_pinecone",
    "all_concat",
    "routed_rrf",
    "routed_weighted",
]
DEPENDENCIES = {
    "all_concat": "fixed_all",
    "routed_weighted": "routed_rrf",
    "routed_raw": "routed_rrf",
    "routed_concat": "routed_rrf",
}
DEV300_COUNTS = {"smoke": 5, "validation": 50, "evaluation": 300, "timing": 20}
DEV300_SEED = "heterorag-dev300-v1"


def select_dev300(cohorts):
    """Select public question IDs before opening references; never select by score."""
    selected = dict(cohorts)
    for cohort in ("smoke", "validation", "evaluation"):
        if len(cohorts[cohort]) < DEV300_COUNTS[cohort]:
            raise ValueError("Insufficient questions for dev300 cohort: " + cohort)
        selected[cohort] = select_questions(
            cohorts[cohort], DEV300_COUNTS[cohort], DEV300_SEED + ":" + cohort
        )
    selected["timing"] = select_questions(
        selected["evaluation"], DEV300_COUNTS["timing"], DEV300_SEED + ":timing"
    )
    return selected


def schedule(questions, phase):
    timing = phase.startswith("timing-")
    conditions = TIMING_CONDITIONS if timing else list(CONDITIONS)
    jobs = []
    for index, question in enumerate(questions):
        qid = question["question_id"]
        pending = sorted(
            conditions,
            key=lambda c: object_hash(["heterorag-condition-order-v1", phase, qid, c]),
        )
        done = set()
        while pending:
            name = next(
                c
                for c in pending
                if timing or DEPENDENCIES.get(c) is None or DEPENDENCIES[c] in done
            )
            pending.remove(name)
            done.add(name)
            dependency = None if timing else DEPENDENCIES.get(name)
            mode = (
                "fusion_replay"
                if dependency and name != "routed_raw"
                else "raw_route"
                if dependency
                else "live"
            )
            identity = object_hash([phase, name, qid])
            jobs.append(
                {
                    "id": identity,
                    "phase": phase,
                    "index": index,
                    "question_id": qid,
                    "condition": name,
                    "mode": mode,
                    "dependency": dependency,
                    "paid": mode == "live"
                    and CONDITIONS[name]["option"] == "decisions",
                }
            )
    return jobs


def prepare(
    base, output, *, include_development=False, profile="full", max_runtime_hours=None
):
    prior = load_bundle(base)
    if profile not in ("full", "dev300") or (
        profile == "dev300" and include_development
    ):
        raise ValueError("Dev300 profile excludes a development rerun")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new evaluation bundle; never overwrite a frozen study")
    output.mkdir(parents=True, exist_ok=True)
    cohort_names = ("smoke", "development", "validation", "evaluation", "timing")
    selected = {
        name: read_json(base / name / "questions.json") for name in cohort_names
    }
    if profile == "dev300":
        selected = select_dev300(selected)
    for cohort in cohort_names:
        for name in ("questions.json", "references.json"):
            target = output / cohort / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if profile == "full" or cohort == "development":
                shutil.copyfile(base / cohort / name, target)
            elif name == "questions.json":
                atomic_json(target, selected[cohort])
            else:
                references = (
                    read_json(base / cohort / name)
                    if cohort != "timing"
                    else read_json(base / "evaluation" / name)
                )
                atomic_json(
                    target,
                    {
                        q["question_id"]: references[q["question_id"]]
                        for q in selected[cohort]
                    },
                )
    for name in ("graph-table-ids.json", "routing-catalog.json"):
        shutil.copyfile(base / name, output / name)
    phases = list(DEFAULT_PHASES)
    repeats = 2 if profile == "dev300" else 3
    if profile == "dev300":
        phases.remove("timing-3")
    if include_development:
        phases.insert(1, "development")
    jobs = [
        j
        for phase in phases
        for j in schedule(
            read_json(
                output
                / ("timing" if phase.startswith("timing-") else phase)
                / "questions.json"
            ),
            phase,
        )
    ]
    atomic_json(output / "schedule.json", jobs)
    protocol = {
        **{
            k: v
            for k, v in prior.items()
            if k
            not in (
                "bundle_id",
                "inputs",
                "code",
                "conditions",
                "timing",
                "manual_evidence_audit",
                "primary_pairs",
            )
        },
        "version": 2,
        "study": "hybridqa",
        "profile": profile,
        "wall_time_budget_seconds": runtime_seconds(
            max_runtime_hours, 21 * 3600 if profile == "dev300" else None
        ),
        "question_selection": {
            "method": "deterministic question-ID hash, independent of labels and graph coverage",
            "seed": DEV300_SEED,
            "heldout_source_questions": len(
                read_json(base / "evaluation/questions.json")
            ),
            "reported_as": "300-question held-out subset; not the full HybridQA dev split",
        }
        if profile == "dev300"
        else {"method": "original full frozen cohorts"},
        "base_bundle_id": prior["bundle_id"],
        "conditions": CONDITIONS,
        "config": asdict(Config()),
        "phases": phases,
        "outcomes": len(jobs),
        "paid_request_cap": sum(j["paid"] for j in jobs),
        "paid_retries": 0,
        "continue_on_error": False,
        "primary_pairs": [
            ["fixed_all", "routed_rrf"],
            ["fixed_postgres_pinecone", "routed_rrf"],
            ["routed_raw", "routed_rrf"],
            ["routed_rrf", "routed_weighted"],
            ["routed_concat", "routed_weighted"],
            ["all_concat", "fixed_all"],
        ],
        "fusion_quality": "Replay the complete candidate records, rankings, protected IDs, route and probabilities; no new retrieval.",
        "guard_ablation": "Reuse the same paid raw choice and distribution; retrieve independently without the guard. No live-router latency claim.",
        "recovery": "Atomic stage checkpoints; study lock; uncertain external stages become retained failures; no paid retries; local answer stages may restart and are timing-ineligible.",
        "timing": {
            "questions": len(selected["timing"]),
            "repeats": repeats,
            "conditions": TIMING_CONDITIONS,
            "order": "deterministically shuffled per question and repeat",
            "router": "fresh paid call for each routed condition and repeat",
            "cache": "natural warm caches; startup separate; no cold-start claim",
        },
        "evidence_claims": {
            "gold_precision_recall_enabled": False,
            "allowed": [
                "source provenance fidelity",
                "automatic answer-trace overlap (weak proxy)",
                "answer EM/F1",
            ],
            "annotation": "Two independent blinded reviewers and adjudication are required for evidence labels. Gold-evidence precision and recall are disabled. Development labels do not establish held-out evidence quality.",
        },
        "phase_gate": "smoke must have no failures; validation must complete and best fixed be sealed before evaluation; no tuning after validation starts",
        "code": code_hashes(),
    }
    covered = set(read_json(output / "graph-table-ids.json"))
    protocol["cohorts"] = {
        name: {
            "questions": len(rows),
            "tables": len({q["table_id"] for q in rows}),
            "graph_covered": sum(q["table_id"] in covered for q in rows),
        }
        for name, rows in selected.items()
    }
    protocol["inputs"] = {
        p.relative_to(output).as_posix(): sha256(p)
        for p in sorted(output.rglob("*.json"))
    }
    protocol["bundle_id"] = object_hash(protocol)
    atomic_json(output / "protocol.json", protocol)
    for relative in protocol["code"]:
        target = output / "source-code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    return protocol


def runtime_seconds(hours, default):
    if hours is None:
        return default
    if not 0 < hours < float("inf"):
        raise ValueError("Runtime limit must be a finite positive number of hours")
    return hours * 3600


def write_input(path, value, serialization):
    """Reproduce pinned input bytes independently of the operating system."""
    text = json.dumps(value, indent=serialization["indent"], ensure_ascii=False)
    if serialization["trailing_newline"]:
        text += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.replace("\n", serialization["newline"]).encode("utf-8"))


def prepare_definition(definition, workspace, output, *, max_runtime_hours=None):
    """Rebuild the prespecified sample from pinned archives without local history."""
    from routing.catalog import prepare as prepare_catalog

    config = read_json(definition / "config.json")
    lock = read_json(REPO / "postgres/sources.lock.json")
    if config["protocol"]["sources"] != lock:
        raise ValueError("Experiment source lock differs from the repository")
    for source, filename in (("questions", "questions.zip"), ("corpus", "corpus.zip")):
        if sha256(workspace / "data" / filename) != lock[source]["archive_sha256"]:
            raise ValueError("Archive checksum mismatch: " + filename)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new evaluation bundle; never overwrite a frozen study")
    output.mkdir(parents=True, exist_ok=True)
    protocol = config["protocol"].copy()
    protocol.update(
        version=3,
        experiment=config["experiment"],
        wall_time_budget_seconds=runtime_seconds(
            max_runtime_hours, protocol.get("wall_time_budget_seconds")
        ),
    )
    ids = config["cohort_ids"]
    if set(ids) != {"smoke", "development", "validation", "evaluation", "timing"}:
        raise ValueError("Unexpected experiment cohorts")
    if any(len(values) != len(set(values)) for values in ids.values()):
        raise ValueError("Duplicate question IDs in experiment definition")
    if not set(ids["smoke"]) <= set(ids["development"]):
        raise ValueError("Smoke questions must be a development subset")
    if not set(ids["timing"]) <= set(ids["evaluation"]):
        raise ValueError("Timing questions must be an evaluation subset")
    for left, right in (
        ("development", "validation"),
        ("development", "evaluation"),
        ("validation", "evaluation"),
    ):
        if set(ids[left]) & set(ids[right]):
            raise ValueError("Development, validation, and evaluation overlap")
    records, split_ids = {}, {}
    with zipfile.ZipFile(workspace / "data/questions.zip") as source:
        for split in ("train", "dev"):
            name = next(
                n
                for n in source.namelist()
                if n.endswith(f"/released_data/{split}.traced.json")
            )
            rows = json.loads(source.read(name))
            split_ids[split] = {q["question_id"] for q in rows}
            for row in rows:
                if row["question_id"] in records:
                    raise ValueError("Duplicate source question ID")
                records[row["question_id"]] = row
    for cohort, question_ids in ids.items():
        split = "dev" if cohort in ("evaluation", "timing") else "train"
        if not set(question_ids) <= split_ids[split]:
            raise ValueError("Question outside intended source split: " + cohort)
        rows = [records[qid] for qid in question_ids]
        values = {
            "questions.json": [public_question(q) for q in rows],
            "references.json": {
                q["question_id"]: {
                    "answer": q["answer-text"],
                    "weak_evidence_ids": weak_ids(q),
                }
                for q in rows
            },
        }
        for filename, value in values.items():
            relative = f"{cohort}/{filename}"
            write_input(
                output / relative, value, config["input_serialization"][relative]
            )
    shutil.copyfile(
        definition / "graph-table-ids.json", output / "graph-table-ids.json"
    )
    catalog = prepare_catalog(
        workspace, output / "routing-catalog.json", output / "graph-table-ids.json"
    )
    write_input(
        output / "routing-catalog.json",
        catalog,
        config["input_serialization"]["routing-catalog.json"],
    )
    for relative, expected in config["expected_inputs"].items():
        if sha256(output / relative) != expected:
            raise ValueError(
                "Rebuilt input differs from prespecified experiment: " + relative
            )
    covered = set(read_json(output / "graph-table-ids.json"))
    counts = {
        name: {
            "questions": len(question_ids),
            "tables": len({records[qid]["table_id"] for qid in question_ids}),
            "graph_covered": sum(
                records[qid]["table_id"] in covered for qid in question_ids
            ),
        }
        for name, question_ids in ids.items()
    }
    if counts != protocol["cohorts"]:
        raise ValueError("Rebuilt cohort metadata differs from the definition")
    jobs = [
        job
        for phase in protocol["phases"]
        for job in schedule(
            read_json(
                output
                / ("timing" if phase.startswith("timing-") else phase)
                / "questions.json"
            ),
            phase,
        )
    ]
    if (
        len(jobs) != protocol["outcomes"]
        or sum(j["paid"] for j in jobs) != protocol["paid_request_cap"]
    ):
        raise ValueError("Experiment outcome or paid-request count mismatch")
    atomic_json(output / "schedule.json", jobs)
    shutil.copyfile(definition / "config.json", output / "experiment.json")
    protocol["code"] = code_hashes()
    protocol["inputs"] = {
        p.relative_to(output).as_posix(): sha256(p)
        for p in sorted(output.rglob("*.json"))
    }
    protocol["bundle_id"] = object_hash(protocol)
    atomic_json(output / "protocol.json", protocol)
    for relative in protocol["code"]:
        target = output / "source-code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    verify(output, check_code=True)
    return protocol


def verify(bundle, *, check_code=False):
    protocol = load_bundle(bundle, check_code=check_code)
    if protocol.get("study") != "hybridqa" or protocol["conditions"] != CONDITIONS:
        raise ValueError("Not a supported evaluation study")
    if protocol.get("profile") == "dev300":
        if (
            protocol["phases"] != [p for p in DEFAULT_PHASES if p != "timing-3"]
            or protocol["timing"]["repeats"] != 2
        ):
            raise ValueError("Dev300 phase or timing-repeat mismatch")
        for cohort, count in DEV300_COUNTS.items():
            if (
                len(read_json(bundle / cohort / "questions.json")) != count
                or protocol["cohorts"][cohort]["questions"] != count
            ):
                raise ValueError("Dev300 cohort count mismatch")
    seconds = protocol.get("wall_time_budget_seconds")
    if seconds is not None and (
        type(seconds) not in (int, float) or not 0 < seconds < float("inf")
    ):
        raise ValueError("Invalid frozen runtime limit")
    jobs = read_json(bundle / "schedule.json")
    expected = [
        j
        for phase in protocol["phases"]
        for j in schedule(
            read_json(
                bundle
                / ("timing" if phase.startswith("timing-") else phase)
                / "questions.json"
            ),
            phase,
        )
    ]
    if (
        jobs != expected
        or len(jobs) != protocol["outcomes"]
        or sum(j["paid"] for j in jobs) != protocol["paid_request_cap"]
    ):
        raise ValueError("Schedule or paid budget mismatch")
    for relative, digest in protocol["code"].items():
        if sha256(bundle / "source-code" / relative) != digest:
            raise ValueError("Frozen source snapshot mismatch")
    return protocol, jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--base", type=Path, help="Optional existing cohort bundle for a custom study"
    )
    parser.add_argument(
        "--definition", type=Path, default=REPO / "experiments/heterorag/routing-fusion"
    )
    parser.add_argument("--workspace", type=Path, default=REPO.parent)
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        help="Freeze a positive wall-clock limit; resumes retain the original expiration",
    )
    parser.add_argument("--include-development", action="store_true")
    parser.add_argument(
        "--profile", choices=("full", "dev300"), help="Selection profile for --base"
    )
    args = parser.parse_args()
    if args.action == "verify":
        protocol = verify(args.bundle, check_code=True)[0]
    elif args.base:
        protocol = prepare(
            args.base,
            args.bundle,
            include_development=args.include_development,
            profile=args.profile or "full",
            max_runtime_hours=args.max_runtime_hours,
        )
    else:
        if args.include_development or args.profile:
            parser.error("--include-development and --profile require --base")
        protocol = prepare_definition(
            args.definition,
            args.workspace,
            args.bundle,
            max_runtime_hours=args.max_runtime_hours,
        )
    print(
        {
            k: protocol[k]
            for k in ("bundle_id", "outcomes", "paid_request_cap", "phases")
        }
    )


if __name__ == "__main__":
    main()
