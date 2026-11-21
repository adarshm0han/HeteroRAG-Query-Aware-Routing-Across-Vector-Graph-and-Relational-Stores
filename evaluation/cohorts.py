"""Cohort selection, source fingerprints, and individual-run manifest validation."""

from dataclasses import asdict
from importlib.metadata import version

from baseline.common import (
    REPO,
    object_hash,
    read_json,
    select_questions,
    sha256,
)
from routing.options import OPTIONS

SEED = "heterorag-evaluation-v1"
EXPOSED_DEV = {
    "00153f694413a536": "Excluded after development-time inspection",
    "5407f39941c147fa": "Excluded after development-time inspection",
}
CONDITIONS = {
    **{f"fixed_{name}": {"option": name, "fusion": "rrf"} for name in OPTIONS},
    "all_concat": {"option": "all", "fusion": "concat"},
    "routed_rrf": {"option": "decisions", "fusion": "rrf"},
    "routed_weighted": {"option": "decisions", "fusion": "weighted_rrf"},
}


def code_hashes():
    paths = [
        path
        for package in ("baseline", "routing", "evaluation", "tools")
        for path in (REPO / package).glob("*.py")
    ]
    paths += [
        path
        for package in ("baseline", "routing", "postgres", "vector")
        for path in (REPO / package).glob("requirements*.txt")
    ]
    paths += [
        REPO / "postgres/sources.lock.json",
        REPO / "requirements.txt",
        REPO / "requirements-dev.txt",
        REPO / "pyproject.toml",
        REPO / ".gitattributes",
    ]
    return {p.relative_to(REPO).as_posix(): sha256(p) for p in sorted(paths)}


def split_questions(
    train, dev, prior_ids, reader_ids, development_n=200, validation_n=100
):
    """Select by IDs only; reserve all previously planned pilot questions from validation."""
    ids = [q["question_id"] for q in [*train, *dev]]
    if len(ids) != len(set(ids)):
        raise ValueError("Question IDs must be unique across source splits")
    anchors = [q for q in train if q["question_id"] in reader_ids]
    if len(anchors) > development_n:
        raise ValueError("Development cohort cannot contain all prior reader questions")
    remaining = [q for q in train if q["question_id"] not in reader_ids]
    development = (
        anchors
        + select_questions(
            remaining, development_n - len(anchors), SEED + ":development"
        )
        if len(anchors) < development_n
        else anchors
    )
    development = select_questions(development, len(development), SEED + ":order")
    excluded = prior_ids | {q["question_id"] for q in development}
    validation = select_questions(
        [q for q in train if q["question_id"] not in excluded],
        validation_n,
        SEED + ":validation",
    )
    evaluation = [q for q in dev if q["question_id"] not in EXPOSED_DEV]
    evaluation = select_questions(evaluation, len(evaluation), SEED + ":evaluation")
    return {
        "development": development,
        "validation": validation,
        "evaluation": evaluation,
        "smoke": development[: min(10, development_n)],
        "timing": select_questions(
            evaluation, min(100, len(evaluation)), SEED + ":timing"
        ),
    }


def weak_ids(question):
    """Automatic answer-span candidates, NOT complete supporting-evidence labels."""
    ids = set()
    for node in question.get("answer-node", []):
        if len(node) != 4:
            raise ValueError("Unexpected HybridQA answer trace")
        if node[3] == "table":
            ids.add(f"row:{question['table_id']}:{node[1][0]}")
        elif node[3] == "passage":
            ids.add(node[2])
        else:
            raise ValueError("Unknown trace kind")
    return sorted(ids)


def build_jobs(bundle, protocol, workspace=None):
    """Reviewable argument lists, not a scheduler or an automatic paid runner."""
    jobs = []
    run_root = REPO / "artifacts/runs" / protocol["bundle_id"][:12]
    cohorts = ["smoke", "development", "validation", "evaluation"]
    for cohort in cohorts + ["timing-1", "timing-2", "timing-3"]:
        timing = cohort.startswith("timing-")
        source = "timing" if timing else cohort
        names = list(CONDITIONS) if not timing else protocol["timing"]["conditions"]
        if timing:
            names = sorted(names, key=lambda name: object_hash([SEED, cohort, name]))
        for name in names:
            condition = CONDITIONS[name]
            args = [
                "--workspace",
                str(workspace or REPO.parent),
                "--questions",
                str(bundle / source / "questions.json"),
                "--references",
                str(bundle / source / "references.json"),
                "--output",
                str(run_root / cohort / name),
                "--evaluation-bundle",
                str(bundle),
                "--cohort",
                source,
                "--condition",
                name,
                "--fusion",
                condition["fusion"],
                "--verify-sources",
                "--continue-on-error",
            ]
            if protocol.get("routing_catalog"):
                args += ["--routing-catalog", str(bundle / "routing-catalog.json")]
            if name == "routed_weighted" and not timing:
                args += ["--decision-records", str(run_root / cohort / "routed_rrf")]
            elif condition["option"] == "decisions":
                # Deliberately omit --allow-decisions-api: user must authorize live work.
                args += [
                    "--max-decision-requests",
                    str(protocol["cohorts"][source]["questions"]),
                ]
            else:
                args += ["--option", condition["option"]]
            jobs.append(
                {
                    "cohort": cohort,
                    "condition": name,
                    "python_module": "routing",
                    "arguments": args,
                    "executed": False,
                    "requires_live_authorization": True,
                    "requires_decisions_opt_in": condition["option"] == "decisions"
                    and not (name == "routed_weighted" and not timing),
                }
            )
    return jobs


def load_bundle(bundle, check_code=False):
    protocol = read_json(bundle / "protocol.json")
    unsigned = {k: v for k, v in protocol.items() if k != "bundle_id"}
    if object_hash(unsigned) != protocol["bundle_id"]:
        raise ValueError("Frozen protocol checksum mismatch")
    for relative, expected in protocol["inputs"].items():
        target = (bundle / relative).resolve()
        if not target.is_relative_to(bundle.resolve()) or sha256(target) != expected:
            raise ValueError("Frozen input checksum mismatch: " + relative)
    if check_code and code_hashes() != protocol["code"]:
        raise ValueError(
            "Code changed since the freeze; create a new version before running"
        )
    return protocol


def validate_run(args, config, questions):
    protocol = load_bundle(args.evaluation_bundle, check_code=True)
    if any(
        version(name) != expected
        for name, expected in protocol.get("dependency_versions", {}).items()
    ):
        raise ValueError(
            "Installed dependency versions differ from the frozen environment"
        )
    if args.cohort not in protocol["cohorts"] or args.condition not in CONDITIONS:
        raise ValueError("Unknown frozen cohort or condition")
    condition = protocol["conditions"][args.condition]
    expected = {**protocol["config"], "fusion": condition["fusion"]}
    if asdict(config) != expected or args.option != condition["option"]:
        raise ValueError("Run settings differ from the frozen condition")
    expected_questions = read_json(
        args.evaluation_bundle / args.cohort / "questions.json"
    )
    if questions != expected_questions:
        raise ValueError("Questions differ from the frozen cohort")
    if args.graph_catalog or not args.verify_sources or not args.continue_on_error:
        raise ValueError(
            "Frozen runs require source audits, no separate graph override and continue-on-error"
        )
    catalog_path = getattr(args, "routing_catalog", None)
    expected_catalog = protocol.get("routing_catalog")
    if expected_catalog:
        if not catalog_path or sha256(catalog_path) != expected_catalog["sha256"]:
            raise ValueError("Routing catalog differs from the frozen public metadata")
    elif catalog_path:
        raise ValueError("This frozen protocol did not supply routing metadata")
    if (
        args.references
        and sha256(args.references)
        != protocol["inputs"][f"{args.cohort}/references.json"]
    ):
        raise ValueError("References differ from the frozen cohort")
    recovery = getattr(args, "smoke_recovery", False)
    if recovery and (
        args.cohort != "smoke"
        or args.condition != "routed_rrf"
        or getattr(args, "allow_decisions_api", False)
    ):
        raise ValueError(
            "Recovery is restricted to the ten-question smoke RRF run with paid calls disabled"
        )
    replay_required = recovery or (
        args.condition == "routed_weighted" and args.cohort != "timing"
    )
    if bool(args.decision_records) != replay_required:
        raise ValueError(
            "Fusion quality uses saved routes; all other frozen runs use live/fixed routes"
        )
    if replay_required:
        source = (
            read_json(args.decision_records / "manifest.json").get("evaluation") or {}
        )
        if recovery:
            if (
                source.get("cohort") != "smoke"
                or source.get("condition") != "routed_rrf"
            ):
                raise ValueError("Recovery requires a previous smoke routed-RRF run")
            if read_json(args.decision_records / "questions.json") != questions:
                raise ValueError(
                    "Recovery questions differ from the original paid smoke"
                )
            records = [
                read_json(p)
                for p in sorted((args.decision_records / "results").glob("*.json"))
            ]
            if (
                len(records) != len(questions)
                or [r["question"] for r in records] != questions
            ):
                raise ValueError(
                    "Recovery requires one saved record per smoke question"
                )
        elif (
            source.get("bundle_id"),
            source.get("cohort"),
            source.get("condition"),
        ) != (
            protocol["bundle_id"],
            args.cohort,
            "routed_rrf",
        ):
            raise ValueError("Replay source is not the matching frozen routed-RRF run")
    return {
        "bundle_id": protocol["bundle_id"],
        "cohort": args.cohort,
        "condition": args.condition,
        "reader_digest": protocol["reader_digest"],
        "recovery_source_bundle_id": source.get("bundle_id") if recovery else None,
        "recovery_source_manifest_sha256": sha256(
            args.decision_records / "manifest.json"
        )
        if recovery
        else None,
    }
