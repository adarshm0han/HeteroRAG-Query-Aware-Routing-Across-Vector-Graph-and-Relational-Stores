"""Scientific evaluation contracts, exercised entirely with synthetic data."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from baseline.common import object_hash, read_json, sha256, write_json
from evaluation.cohorts import (
    CONDITIONS,
    EXPOSED_DEV,
    build_jobs,
    code_hashes,
    load_bundle,
    split_questions,
    validate_run,
    weak_ids,
)
from evaluation.metrics import (
    aggregate,
    compare,
    evidence_metrics,
    fusion_controls,
    load_run,
    paired,
    score_question,
    timing_report,
)
from routing.pipeline import Config
from routing.router import DecisionsRouter, FixedRouter, RecordedRouter

from .fixtures import QUESTION, FakeDecisions
from .test_pipeline import pipeline


def test_splits_reserve_pilots_exclude_exposed_dev_and_ignore_answers():
    train = [
        {**QUESTION, "question_id": f"train-{i}", "answer-text": str(i)}
        for i in range(30)
    ]
    dev = [{**QUESTION, "question_id": f"dev-{i}"} for i in range(12)]
    dev += [{**QUESTION, "question_id": qid} for qid in EXPOSED_DEV]
    prior, reader = {f"train-{i}" for i in range(8)}, {"train-1", "train-2"}
    result = split_questions(train, dev, prior, reader, 10, 5)
    ids = {name: {q["question_id"] for q in rows} for name, rows in result.items()}
    assert reader <= ids["development"]
    assert not ids["validation"] & (prior | ids["development"])
    assert not ids["evaluation"] & set(EXPOSED_DEV)
    assert ids["smoke"] <= ids["development"]
    assert ids["timing"] <= ids["evaluation"]
    changed = [
        {**q, "answer-text": "different", "answer-node": []} for q in reversed(train)
    ]
    again = split_questions(changed, list(reversed(dev)), prior, reader, 10, 5)
    assert {name: [q["question_id"] for q in rows] for name, rows in again.items()} == {
        name: [q["question_id"] for q in rows] for name, rows in result.items()
    }


def test_weak_trace_ids_and_deduplication():
    q = {
        **QUESTION,
        "answer-node": [
            ["name", [0, 0], None, "table"],
            ["name", [0, 1], None, "table"],
            ["name", [0, 1], "/wiki/Ada", "passage"],
        ],
    }
    assert weak_ids(q) == ["/wiki/Ada", "row:synthetic-table:0"]
    assert evidence_metrics(["a", "a", "b"], ["a", "c"])["recall"] == 0.5
    assert evidence_metrics(["a", "a", "b"], ["a", "c"])["precision"] == 0.5
    assert evidence_metrics([], ["a"])["recall"] == 0
    assert evidence_metrics(["a"], [])["recall"] is None


def make_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    reference = {
        QUESTION["question_id"]: {
            "answer": "Rocket",
            "weak_evidence_ids": ["/wiki/Ada", "row:synthetic-table:0"],
        }
    }
    for cohort in ("smoke", "development", "validation", "evaluation", "timing"):
        write_json(bundle / cohort / "questions.json", [QUESTION])
        write_json(bundle / cohort / "references.json", reference)
    write_json(bundle / "graph-table-ids.json", ["synthetic-table"])
    write_json(
        bundle / "manual-audit-template.json",
        {QUESTION["question_id"]: {"complete": False}},
    )
    protocol = {
        "config": asdict(Config()),
        "reader_digest": "synthetic",
        "code": code_hashes(),
        "conditions": CONDITIONS,
        "fusion_weight_method": "route_marginals_v1",
        "primary_pairs": [
            ["fixed_all", "routed_rrf"],
            ["routed_rrf", "routed_weighted"],
        ],
        "cohorts": {
            name: {"questions": 1}
            for name in ("smoke", "development", "validation", "evaluation", "timing")
        },
        "timing": {
            "conditions": ["fixed_all", "all_concat", "routed_rrf", "routed_weighted"]
        },
        "inputs": {
            p.relative_to(bundle).as_posix(): sha256(p) for p in bundle.rglob("*.json")
        },
    }
    protocol["bundle_id"] = object_hash(protocol)
    write_json(bundle / "protocol.json", protocol)
    return bundle, protocol


def write_run(root, protocol, cohort, condition):
    setting = CONDITIONS[condition]
    app, _, _ = pipeline("all")
    app.config = Config(fusion=setting["fusion"])
    if setting["option"] != "decisions":
        app.router = FixedRouter(setting["option"])
    elif condition == "routed_weighted" and cohort != "timing":
        decision = DecisionsRouter(FakeDecisions("all")).choose(QUESTION)
        app.router = RecordedRouter([{"question": QUESTION, "decision": decision}])
    row = app.run(QUESTION)
    assert row["status"] == "completed"
    row["source_audit"] = {"passed": True}
    path = root / condition
    manifest = {
        "evaluation": {
            "bundle_id": protocol["bundle_id"],
            "cohort": cohort,
            "condition": condition,
        },
        "config": asdict(app.config),
        "option": setting["option"],
        "reader": {"digest": "synthetic"},
        "code": protocol["code"],
        "verify_sources": True,
        "startup_seconds": 0.01,
        "decision_records": "synthetic" if row["route_replayed"] else None,
    }
    write_json(path / "manifest.json", manifest)
    write_json(path / "questions.json", [QUESTION])
    write_json(path / "results/00000.json", row)
    return str(path)


def test_frozen_bundle_detects_mutations_and_jobs_do_not_enable_paid_calls(tmp_path):
    bundle, protocol = make_bundle(tmp_path)
    assert load_bundle(bundle, check_code=True)["bundle_id"] == protocol["bundle_id"]
    jobs = build_jobs(bundle, protocol)
    assert len(jobs) == 52
    assert not any("--allow-decisions-api" in j["arguments"] for j in jobs)
    assert not any(j["executed"] for j in jobs)
    write_json(bundle / "evaluation/questions.json", [])
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(bundle)


def test_validate_run_rejects_setting_and_label_changes_before_execution(tmp_path):
    bundle, _ = make_bundle(tmp_path)
    args = SimpleNamespace(
        evaluation_bundle=bundle,
        cohort="validation",
        condition="fixed_all",
        option="all",
        graph_catalog=None,
        verify_sources=True,
        continue_on_error=True,
        references=None,
        decision_records=None,
    )
    validate_run(args, Config(), [QUESTION])
    with pytest.raises(ValueError, match="settings"):
        validate_run(args, Config(row_k=9), [QUESTION])
    with pytest.raises(ValueError, match="Questions"):
        validate_run(args, Config(), [{**QUESTION, "question": "changed"}])


def test_smoke_recovery_reuses_same_questions_without_new_paid_calls(tmp_path):
    bundle, protocol = make_bundle(tmp_path)
    previous = {**protocol, "bundle_id": "previous-smoke"}
    path = write_run(tmp_path / "old", previous, "smoke", "routed_rrf")
    args = SimpleNamespace(
        evaluation_bundle=bundle,
        cohort="smoke",
        condition="routed_rrf",
        option="decisions",
        graph_catalog=None,
        verify_sources=True,
        continue_on_error=True,
        references=None,
        decision_records=Path(path),
        smoke_recovery=True,
        allow_decisions_api=False,
    )
    result = validate_run(args, Config(), [QUESTION])
    assert result["recovery_source_bundle_id"] == "previous-smoke"
    args.allow_decisions_api = True
    with pytest.raises(ValueError, match="paid calls disabled"):
        validate_run(args, Config(), [QUESTION])
    args.allow_decisions_api = False
    args.cohort = "evaluation"
    with pytest.raises(ValueError, match="restricted"):
        validate_run(args, Config(), [QUESTION])


def test_score_failures_empty_labels_and_manual_annotation_denominators():
    app, _, _ = pipeline("all")
    row = app.run(QUESTION)
    reference = {"answer": "Rocket", "weak_evidence_ids": ["/wiki/Ada", "missing"]}
    scores = score_question(row, reference, True)
    assert scores["f1"] == 1
    assert scores["weak_packed_recall"] == 0.5
    assert scores["manual_packed_recall"] is None
    manual = {"complete": True, "reviewer": "reviewer", "evidence_ids": ["/wiki/Ada"]}
    assert score_question(row, reference, True, manual)["manual_packed_recall"] == 1
    failed = {
        "question": QUESTION,
        "status": "failed",
        "total_seconds": 2,
        "database_calls": [],
    }
    failed_score = score_question(failed, {"answer": "", "weak_evidence_ids": []}, True)
    assert failed_score["em"] == 0
    combined = aggregate([scores, failed_score])
    assert combined["f1_percent"] == 50
    assert combined["failed_percent"] == 50
    assert combined["weak_packed_recall_questions"] == 1
    assert combined["latency"]["total_seconds"]["observations"] == 2
    assert combined["latency"]["answer_seconds"]["observations"] == 1
    assert paired([scores], [scores])["f1_right_minus_left_points"] == 0


def test_report_selects_best_fixed_on_validation_and_requires_it_for_evaluation(
    tmp_path,
):
    bundle, protocol = make_bundle(tmp_path)
    runs = {
        name: write_run(tmp_path / "validation", protocol, "validation", name)
        for name in CONDITIONS
    }
    report = compare(bundle, "validation", runs, tmp_path / "validation-report")
    assert report["fusion_controls"]["matched"]
    assert report["results"]["routed_rrf"]["f1_percent"] == 100
    assert not report["results"]["routed_weighted"]["live_latency_eligible"]
    assert report["best_fixed"].startswith("fixed_")
    final_runs = {
        name: write_run(tmp_path / "evaluation", protocol, "evaluation", name)
        for name in CONDITIONS
    }
    with pytest.raises(ValueError, match="requires the best-fixed"):
        compare(bundle, "evaluation", final_runs, tmp_path / "final-report")
    final = compare(
        bundle,
        "evaluation",
        final_runs,
        tmp_path / "final-report",
        tmp_path / "validation-report/best-fixed.json",
    )
    assert final["best_fixed"] == report["best_fixed"]
    assert (tmp_path / "final-report/questions.csv").exists()


def test_missing_results_and_changed_model_are_rejected(tmp_path):
    _bundle, protocol = make_bundle(tmp_path)
    path = tmp_path / "run"
    write_json(path / "manifest.json", {"evaluation": {}})
    with pytest.raises(ValueError, match="frozen bundle"):
        load_run(path, protocol, "validation", "fixed_all", [QUESTION])
    path = tmp_path / "actual/fixed_all"
    write_run(path.parent, protocol, "validation", "fixed_all")
    questions = [QUESTION, {**QUESTION, "question_id": "missing"}]
    write_json(path / "questions.json", questions)
    with pytest.raises(ValueError, match="Incomplete"):
        load_run(
            path,
            protocol,
            "validation",
            "fixed_all",
            questions,
        )
    manifest = read_json(path / "manifest.json")
    manifest["reader"]["digest"] = "changed-model"
    write_json(path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="local reader"):
        load_run(path, protocol, "validation", "fixed_all", questions)


def test_fusion_claims_are_disabled_on_evidence_changes_or_missing_retrieval():
    app, _, _ = pipeline("all")
    left = app.run(QUESTION)
    right = deepcopy(left)
    right["retrieval"]["candidate_evidence_sha256"] = "changed"
    assert not fusion_controls([left], [right])["pure_fusion_claim_allowed"]
    right.pop("retrieval")
    assert fusion_controls([left], [right])["unverifiable_questions"] == [
        QUESTION["question_id"]
    ]


def test_timing_requires_distinct_live_runs_and_reports_all_repeats(tmp_path):
    bundle, protocol = make_bundle(tmp_path)
    runs = {
        str(repeat): {
            name: write_run(tmp_path / str(repeat), protocol, "timing", name)
            for name in protocol["timing"]["conditions"]
        }
        for repeat in (1, 2, 3)
    }
    report = timing_report(bundle, runs, tmp_path / "timing-report")
    assert report["pooled_descriptive"]["routed_rrf"]["questions"] == 3
    runs["2"] = runs["1"]
    with pytest.raises(ValueError, match="distinct"):
        timing_report(bundle, runs, tmp_path / "bad-timing")
