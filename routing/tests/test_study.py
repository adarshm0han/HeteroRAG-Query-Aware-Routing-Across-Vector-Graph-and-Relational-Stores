"""Crash boundaries, paid budgets, blinded labels and controlled evaluation experiments."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from baseline.common import object_hash, read_json, sha256
from evaluation.annotation import validate
from evaluation.checkpoints import (
    Checkpoints,
    InterruptedStage,
    PaidLedger,
    atomic_json,
    load,
    save,
    study_lock,
)
from evaluation.execution import execute, raw_decision, replay_fusion
from evaluation.protocol import CONDITIONS, DEFAULT_PHASES, schedule
from routing.catalog import table_metadata
from routing.pipeline import Config, Pipeline
from routing.router import DecisionsRouter
from routing.stores import LiveStores

from .fixtures import (
    QUESTION,
    FakeDecisions,
    FakeEmbedding,
    FakeReader,
    FakeStores,
    FakeTokenizer,
)


def crash(*args, **kwargs):
    raise KeyboardInterrupt("simulated power/process interruption")


def forbidden(*args, **kwargs):
    raise AssertionError("external work was repeated")


def test_damaged_checkpoint_and_identity_changes_fail_closed(tmp_path):
    path = tmp_path / "saved.json"
    save(path, {"a": 1})
    data = read_json(path)
    data["value"]["a"] = 2
    atomic_json(path, data)
    with pytest.raises(ValueError, match="checksum"):
        load(path)
    Checkpoints(tmp_path / "job", {"q": 1})
    with pytest.raises(ValueError, match="another job"):
        Checkpoints(tmp_path / "job", {"q": 2})


def test_checkpoint_recovery_never_repeats_external_stage(tmp_path):
    saved = Checkpoints(tmp_path, {"q": 1})
    assert saved.step("route", lambda: {"a": 1}) == {"a": 1}
    assert saved.step("route", forbidden) == {"a": 1}
    with pytest.raises(KeyboardInterrupt):
        saved.step("retrieval", crash)
    resumed = Checkpoints(tmp_path, {"q": 1})
    with pytest.raises(InterruptedStage):
        resumed.step("retrieval", forbidden)
    with pytest.raises(KeyboardInterrupt):
        resumed.step("answer", crash, recover=True)
    assert (
        resumed.step("answer", lambda: "local answer", recover=True) == "local answer"
    )
    assert len(load(tmp_path / "stage-answer.json")["history"]) == 1


def test_study_lock_excludes_concurrent_processes_and_releases(tmp_path):
    with study_lock(tmp_path), pytest.raises(RuntimeError, match="Another process"):  # noqa: SIM117 -- test nested lock failure
        with study_lock(tmp_path):
            pass
    with study_lock(tmp_path):
        pass


def transport(mode="complete", counter=None):
    class Transport:
        def __init__(self, key, **kwargs):
            self.journal = kwargs["journal"]
            assert kwargs["capture_response"] and kwargs["max_requests"] == 1

        def create(self, payload):
            if counter is not None:
                counter.append(payload)
            event = {"status": "started"}
            self.journal([event])
            if mode == "interrupt":
                crash()
            response = (
                FakeDecisions("postgres").create(payload)
                if "questions" in payload
                else {"answer": "saved"}
            )
            event.update(status="completed", response=response, seconds=0.1)
            self.journal([event])
            if mode == "after_response":
                crash()
            return response

    return Transport


def test_paid_response_survives_crash_before_decision_checkpoint(tmp_path):
    ledger = PaidLedger(tmp_path, 1, ["a"])
    calls = []
    with pytest.raises(KeyboardInterrupt):
        ledger.client(
            "a", "dummy", allow_paid=True, factory=transport("after_response", calls)
        ).create({"x": 1})
    ledger = PaidLedger(tmp_path, 1, ["a"])
    client = ledger.client("a", "dummy", allow_paid=False, factory=forbidden)
    assert client.create({"x": 1}) == {"answer": "saved"}
    assert client.reused and len(calls) == 1
    with pytest.raises(ValueError, match="payload changed"):
        client.create({"x": 2})


def test_uncertain_paid_attempt_counts_against_budget_and_never_retries(tmp_path):
    ledger = PaidLedger(tmp_path, 1, ["a", "b"])
    with pytest.raises(KeyboardInterrupt):
        ledger.client(
            "a", "dummy", allow_paid=True, factory=transport("interrupt")
        ).create({"x": 1})
    ledger = PaidLedger(tmp_path, 1, ["a", "b"])
    with pytest.raises(InterruptedStage):
        ledger.client("a", "dummy", allow_paid=True, factory=forbidden).create({"x": 1})
    with pytest.raises(ValueError, match="budget exhausted"):
        ledger.client("b", "dummy", allow_paid=True, factory=forbidden).create({"x": 1})
    with pytest.raises(ValueError, match="outside"):
        ledger.client("unknown", "dummy", allow_paid=True, factory=forbidden).create(
            {"x": 1}
        )


def test_live_operation_is_journaled_before_external_side_effect(tmp_path):
    saved = []
    store = LiveStores(
        {}, ["postgres"], journal=lambda events: saved.append(deepcopy(events))
    )
    with store.operation("postgres", "select_table"):
        assert saved[-1][-1]["status"] == "started"
    assert saved[-1][-1]["status"] == "completed"


def dependencies(root, router=None, factory=None, reader=None):
    metadata = table_metadata(
        {
            "title": "Scores",
            "header": [["Player", []], ["Points", []]],
            "data": [[[], []], [[], []]],
        },
        True,
    )
    catalog = SimpleNamespace(digest="catalog", describe=lambda tid: metadata)
    return {
        "router": router or DecisionsRouter(FakeDecisions("postgres")),
        "stores_factory": factory or (lambda allowed, journal: FakeStores(allowed)),
        "embedding": FakeEmbedding(),
        "reader": reader or FakeReader(),
        "tokenizer": FakeTokenizer(),
        "config": Config(),
        "auditor": SimpleNamespace(check=lambda tid, items: {"passed": True}),
        "catalog": catalog,
    }


def selected_job(name="routed_rrf", phase="smoke"):
    return next(j for j in schedule([QUESTION], phase) if j["condition"] == name)


def test_live_output_matches_existing_inference_pipeline(tmp_path):
    kwargs = dependencies(tmp_path)
    result = execute(QUESTION, selected_job(), tmp_path / "job", **kwargs)
    old = Pipeline(
        kwargs["router"],
        lambda allowed: FakeStores(allowed),
        kwargs["embedding"],
        kwargs["reader"],
        kwargs["tokenizer"],
        auditor=kwargs["auditor"],
        catalog=kwargs["catalog"],
    ).run(QUESTION)
    assert result["status"] == old["status"] == "completed"
    for key in (
        "candidate_evidence_sha256",
        "rankings",
        "protected_passages",
        "context",
        "evidence",
        "items",
    ):
        assert result["retrieval"][key] == old["retrieval"][key]
    assert result["answer"] == old["answer"]
    assert result["database_calls"] == old["database_calls"]


def test_fusion_replays_all_candidates_without_store_or_embedding_calls(tmp_path):
    kwargs = dependencies(tmp_path)
    original = execute(QUESTION, selected_job(), tmp_path / "original", **kwargs)
    kwargs.update(
        router=None,
        stores_factory=forbidden,
        embedding=SimpleNamespace(encode=forbidden),
        config=Config(fusion="weighted_rrf"),
    )
    replay = execute(
        QUESTION,
        selected_job("routed_weighted"),
        tmp_path / "weighted",
        source=original,
        **kwargs,
    )
    assert replay["status"] == "completed", replay
    assert not replay["database_calls"] and not replay["decisions_calls"]
    assert (
        replay["retrieval"]["candidate_records"]
        == original["retrieval"]["candidate_records"]
    )
    assert replay["retrieval"]["rankings"] == original["retrieval"]["rankings"]
    assert replay["replay_source"]["sha256"] == object_hash(original)
    assert not replay["timing_eligible"]
    damaged = deepcopy(original["retrieval"])
    next(iter(damaged["candidate_records"].values()))["text"] = "tampered"
    with pytest.raises(ValueError, match="corrupt"):
        replay_fusion(damaged, Config(fusion="concat"), None)


def test_guard_ablation_uses_raw_choice_with_no_paid_call(tmp_path):
    kwargs = dependencies(tmp_path)
    original = execute(QUESTION, selected_job(), tmp_path / "original", **kwargs)
    assert original["decision"]["option"] == "postgres_pinecone"
    kwargs["router"] = None
    raw = execute(
        QUESTION,
        selected_job("routed_raw"),
        tmp_path / "raw",
        source=original,
        **kwargs,
    )
    assert raw["decision"]["option"] == "postgres"
    assert raw["selected_stores"] == ["postgres"]
    assert {e["store"] for e in raw["database_calls"]} == {"postgres"}
    assert raw["decisions_calls"] == []
    assert not raw["timing_eligible"]
    assert (
        raw_decision(original["decision"])["probabilities"]
        == original["decision"]["raw_probabilities"]
    )


def test_resume_after_answer_interruption_reuses_paid_route_and_retrieval(tmp_path):
    job = selected_job()
    ledger = PaidLedger(tmp_path / "paid", 1, [job["id"]])
    paid_calls, stores = [], []

    def factory(allowed, journal):
        stores.append(allowed)
        return FakeStores(allowed)

    kwargs = dependencies(
        tmp_path,
        router=DecisionsRouter(
            ledger.client(
                job["id"],
                "dummy",
                allow_paid=True,
                factory=transport(counter=paid_calls),
            )
        ),
        factory=factory,
        reader=SimpleNamespace(generate=crash),
    )
    with pytest.raises(KeyboardInterrupt):
        execute(QUESTION, job, tmp_path / "job", **kwargs)
    ledger = PaidLedger(tmp_path / "paid", 1, [job["id"]])
    kwargs.update(
        router=DecisionsRouter(
            ledger.client(job["id"], "dummy", allow_paid=True, factory=forbidden)
        ),
        stores_factory=forbidden,
        reader=FakeReader(),
    )
    result = execute(QUESTION, job, tmp_path / "job", **kwargs)
    assert result["status"] == "completed"
    assert len(paid_calls) == len(stores) == len(result["decisions_calls"]) == 1
    assert result["resumed"] and not result["timing_eligible"]
    kwargs["reader"] = SimpleNamespace(generate=forbidden)
    assert execute(QUESTION, job, tmp_path / "job", **kwargs) == result


def test_interrupted_retrieval_is_a_retained_zero_score_failure(tmp_path):
    job = selected_job()
    kwargs = dependencies(tmp_path, factory=crash)
    with pytest.raises(KeyboardInterrupt):
        execute(QUESTION, job, tmp_path, **kwargs)
    kwargs["stores_factory"] = forbidden
    result = execute(QUESTION, job, tmp_path, **kwargs)
    assert result["status"] == "failed" and result["failed_stage"] == "retrieval"
    assert result["failure_details"]["interrupted_external_stage"]


def test_schedule_has_exact_counts_fresh_timing_calls_and_dependencies():
    counts = {"smoke": 10, "validation": 100, "evaluation": 3464, "timing": 100}
    all_jobs = []
    for phase in DEFAULT_PHASES:
        count = counts["timing" if phase.startswith("timing-") else phase]
        questions = [{**QUESTION, "question_id": str(i)} for i in range(count)]
        jobs = schedule(questions, phase)
        done = set()
        for job in jobs:
            if job["dependency"]:
                assert (job["question_id"], job["dependency"]) in done
            done.add((job["question_id"], job["condition"]))
        all_jobs.extend(jobs)
    assert len(CONDITIONS) == 12
    assert len(all_jobs) == 44388
    assert sum(j["paid"] for j in all_jobs) == 4174
    assert len({j["id"] for j in all_jobs}) == len(all_jobs)
    assert all(
        j["mode"] == "live" for j in all_jobs if j["phase"].startswith("timing-")
    )


def test_annotation_validation_rejects_blank_shared_or_invalid_reviewers(tmp_path):
    packet = tmp_path / "packet"
    record = {
        "question": QUESTION,
        "corpus_file": "source.json",
        "reviewer": "",
        "independent": False,
        "complete": False,
        "answer": "",
        "evidence_sets": [],
        "rationale": "",
    }
    atomic_json(
        packet / "source.json",
        {
            "row_ids": ["row:synthetic-table:0"],
            "passages": {"/wiki/Ada": "Ada is Rocket"},
        },
    )
    template = {"synthetic-1": record}
    atomic_json(packet / "reviewer-a-template.json", template)
    manifest = {"inputs": {p.name: sha256(p) for p in packet.glob("*.json")}}
    manifest["packet_id"] = object_hash(manifest)
    atomic_json(packet / "packet.json", manifest)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    atomic_json(a, template)
    atomic_json(b, template)
    with pytest.raises(ValueError, match="Named independent"):
        validate(packet, a, b)
    completed = {
        **record,
        "reviewer": "Reviewer A",
        "independent": True,
        "complete": True,
        "answer": "Rocket",
        "evidence_sets": [["row:synthetic-table:0", "/wiki/Ada"]],
        "rationale": "Identify player then nickname",
    }
    atomic_json(a, {"synthetic-1": completed})
    atomic_json(b, {"synthetic-1": completed})
    with pytest.raises(ValueError, match="share a reviewer"):
        validate(packet, a, b)
    atomic_json(b, {"synthetic-1": {**completed, "reviewer": "Reviewer B"}})
    checked = validate(packet, a, b)
    assert checked["independently_completed_pairs"] == 1
    assert checked["gold_claims_enabled"] is False
    atomic_json(
        b,
        {
            "synthetic-1": {
                **completed,
                "reviewer": "Reviewer B",
                "evidence_sets": [["/wiki/Invalid"]],
            }
        },
    )
    with pytest.raises(ValueError, match="canonical"):
        validate(packet, a, b)


def synthetic_study(
    tmp_path, monkeypatch, phases=("smoke", "validation", "evaluation")
):
    import evaluation.report as reporting
    from routing.router import FixedRouter

    bundle, output = tmp_path / "bundle", tmp_path / "output"
    jobs = [j for phase in phases for j in schedule([QUESTION], phase)]
    protocol = {
        "bundle_id": "synthetic-study",
        "phases": list(phases),
        "config": vars(Config()),
        "conditions": CONDITIONS,
        "routing_catalog": {"sha256": "catalog"},
        "noninferiority_margin_f1_points": 2.0,
        "primary_pairs": [
            ["fixed_all", "routed_rrf"],
            ["fixed_postgres_pinecone", "routed_rrf"],
            ["routed_raw", "routed_rrf"],
            ["routed_rrf", "routed_weighted"],
            ["routed_concat", "routed_weighted"],
            ["all_concat", "fixed_all"],
        ],
    }
    monkeypatch.setattr(reporting, "verify", lambda *a, **k: (protocol, jobs))
    save(output / "study.json", {"bundle_id": protocol["bundle_id"]})
    atomic_json(bundle / "graph-table-ids.json", [QUESTION["table_id"]])
    for phase in phases:
        source_name = "timing" if phase.startswith("timing-") else phase
        atomic_json(bundle / source_name / "questions.json", [QUESTION])
        atomic_json(
            bundle / source_name / "references.json",
            {
                QUESTION["question_id"]: {
                    "answer": "Rocket",
                    "weak_evidence_ids": ["/wiki/Ada"],
                }
            },
        )
    saved = {}
    for job in jobs:
        setting = CONDITIONS[job["condition"]]
        kwargs = dependencies(tmp_path)
        if setting["option"] != "decisions":
            kwargs["router"] = FixedRouter(setting["option"])
        kwargs["config"] = Config(fusion=setting["fusion"])
        if job["mode"] == "fusion_replay":
            kwargs["stores_factory"] = forbidden
        source = saved.get((job["phase"], job["dependency"]))
        result = execute(
            QUESTION, job, output / "jobs" / job["id"], source=source, **kwargs
        )
        assert result["status"] in ("completed", "empty_evidence"), result
        saved[(job["phase"], job["condition"])] = result
    return bundle, output, protocol, jobs


def test_full_reports_seal_validation_and_disable_replay_efficiency_claims(
    tmp_path, monkeypatch
):
    from evaluation.report import phase_results, report

    bundle, output, _, _ = synthetic_study(tmp_path, monkeypatch)
    report(bundle, output, "smoke")
    validation = report(bundle, output, "validation")
    assert load(output / "best-fixed.json")["condition"] == "fixed_neo4j_pinecone"
    final = report(bundle, output, "evaluation")
    assert final["gold_evidence_precision_recall"] is None
    assert final["fusion_controls"]["weighted"]["matched"]
    assert final["fusion_controls"]["concat"]["matched"]
    assert (
        final["paired_comparisons"]["routed_rrf -> routed_weighted"][
            "data_request_reduction_percent"
        ]
        is None
    )
    assert (
        final["results"]["routed_raw"]["latency"]["service_seconds"]["observations"]
        == 0
    )
    assert validation["results"]["fixed_all"]["timing_eligible_questions"] == 1
    raw = phase_results(bundle, output, "evaluation")[2]
    assert "candidate_records" not in raw["fixed_all"][0]["retrieval"]
    selection = load(output / "best-fixed.json")
    selection["condition"] = "fixed_neo4j"
    save(output / "best-fixed.json", selection)
    with pytest.raises(ValueError, match="selection changed"):
        report(bundle, output, "evaluation")


def test_partial_phase_does_not_open_reference_answers(tmp_path, monkeypatch):
    import evaluation.report as reporting

    bundle, output, _, jobs = synthetic_study(tmp_path, monkeypatch, phases=("smoke",))
    missing = output / "jobs" / jobs[-1]["id"] / "result.json"
    missing.unlink()
    original_read = reporting.read_json

    def read(path):
        assert path.name != "references.json", (
            "Reference answers opened before completion"
        )
        return original_read(path)

    monkeypatch.setattr(reporting, "read_json", read)
    with pytest.raises(ValueError, match="incomplete"):
        reporting.report(bundle, output, "smoke")


def test_runner_requires_launch_opt_in_before_initialization(tmp_path, monkeypatch):
    from evaluation import runner

    monkeypatch.setattr(runner, "verify", lambda *a, **k: ({"paid_request_cap": 4}, []))
    with pytest.raises(PermissionError, match="Launch requires"):
        runner.run(
            tmp_path / "bundle",
            tmp_path / "output",
            tmp_path,
            phase="all",
            resume=False,
            allow_live=False,
            allow_paid=False,
            paid_cap=0,
        )
    assert not (tmp_path / "output").exists()
