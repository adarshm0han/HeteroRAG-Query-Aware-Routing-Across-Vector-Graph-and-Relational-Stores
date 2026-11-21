"""Time-bounded study selection and deadline behavior without any paid calls."""

import subprocess
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from baseline.common import object_hash, read_json, sha256
from evaluation.checkpoints import atomic_json, load
from evaluation.protocol import prepare, schedule, select_dev300, verify
from evaluation.runner import wall_deadline

from .fixtures import QUESTION
from .test_study import synthetic_study


def cohorts():
    def rows(prefix, count):
        return [{**QUESTION, "question_id": f"{prefix}-{i}"} for i in range(count)]

    development, evaluation = rows("dev", 200), rows("eval", 350)
    return {
        "development": development,
        "smoke": development[:10],
        "validation": rows("val", 100),
        "evaluation": evaluation,
        "timing": evaluation[:100],
    }


def test_dev300_selection_is_label_independent_and_timing_is_nested():
    source = cohorts()
    selected = select_dev300(source)
    changed = {
        name: [{**q, "answer-text": "irrelevant reference"} for q in reversed(rows)]
        for name, rows in source.items()
    }
    again = select_dev300(changed)
    ids = lambda rows: [q["question_id"] for q in rows]
    for name in ("smoke", "validation", "evaluation", "timing"):
        assert ids(selected[name]) == ids(again[name])
    assert set(ids(selected["timing"])) <= set(ids(selected["evaluation"]))
    assert not set(ids(selected["validation"])) & set(ids(selected["development"]))
    phases = ["smoke", "validation", "evaluation", "timing-1", "timing-2"]
    jobs = [
        j
        for phase in phases
        for j in schedule(
            selected["timing" if phase.startswith("timing-") else phase], phase
        )
    ]
    assert len(jobs) == 4460
    assert sum(j["paid"] for j in jobs) == 435


def test_prepare_and_verify_dev300_bundle_preserves_matching_references(tmp_path):
    source = cohorts()
    base, output = tmp_path / "base", tmp_path / "dev300"
    for name, rows in source.items():
        atomic_json(base / name / "questions.json", rows)
        atomic_json(
            base / name / "references.json",
            {
                q["question_id"]: {"answer": q["question_id"], "weak_evidence_ids": []}
                for q in rows
            },
        )
    atomic_json(base / "routing-catalog.json", {"synthetic": True})
    atomic_json(base / "graph-table-ids.json", [QUESTION["table_id"]])
    original = {
        "inputs": {
            p.relative_to(base).as_posix(): sha256(p) for p in base.rglob("*.json")
        }
    }
    original["bundle_id"] = object_hash(original)
    atomic_json(base / "protocol.json", original)
    protocol = prepare(base, output, profile="dev300")
    assert protocol["wall_time_budget_seconds"] == 21 * 3600
    assert protocol["outcomes"] == 4460 and protocol["paid_request_cap"] == 435
    assert protocol["timing"]["repeats"] == 2
    assert verify(output, check_code=True)[0] == protocol
    for name in source:
        questions = read_json(output / name / "questions.json")
        references = read_json(output / name / "references.json")
        assert set(references) == {q["question_id"] for q in questions}
        assert all(
            references[q["question_id"]]["answer"] == q["question_id"]
            for q in questions
        )
    with pytest.raises(ValueError, match="never overwrite"):
        prepare(base, output, profile="dev300")


def test_deadline_survives_resume_and_cancels_on_exit(tmp_path):
    started = datetime(2026, 11, 19, tzinfo=UTC)
    protocol = {"bundle_id": "synthetic", "wall_time_budget_seconds": 75600}
    timers = []

    def factory(seconds, callback):
        timer = SimpleNamespace(
            seconds=seconds,
            callback=callback,
            start=lambda: None,
            cancel=lambda: setattr(timer, "cancelled", True),
        )
        timers.append(timer)
        return timer

    exits = []
    with wall_deadline(
        tmp_path,
        protocol,
        started.isoformat(),
        now=started,
        timer_factory=factory,
        exit_process=exits.append,
    ):
        assert timers[-1].seconds == 75600
    assert timers[-1].cancelled
    with wall_deadline(
        tmp_path,
        protocol,
        started.isoformat(),
        now=started + timedelta(hours=5),
        timer_factory=factory,
        exit_process=exits.append,
    ):
        assert timers[-1].seconds == 16 * 3600
        timers[-1].callback()
    assert exits == [124]
    assert load(tmp_path / "deadline-expired.json")["status"] == "deadline_reached"
    with (
        pytest.raises(TimeoutError, match="expired"),
        wall_deadline(
            tmp_path,
            protocol,
            started.isoformat(),
            now=started + timedelta(hours=22),
            timer_factory=factory,
            exit_process=exits.append,
        ),
    ):
        pass
    with (
        pytest.raises(ValueError, match="deadline mismatch"),
        wall_deadline(
            tmp_path,
            protocol,
            (started + timedelta(hours=1)).isoformat(),
            now=started,
            timer_factory=factory,
            exit_process=exits.append,
        ),
    ):
        pass


def test_timing_report_accepts_two_frozen_repeats(tmp_path, monkeypatch):
    from evaluation.report import timing_report

    bundle, output, protocol, _ = synthetic_study(
        tmp_path, monkeypatch, phases=("timing-1", "timing-2")
    )
    protocol["timing"] = {"repeats": 2}
    result = timing_report(bundle, output)
    assert set(result["repeats"]) == {"1", "2"}


def test_deadline_stops_a_real_worker_and_preserves_saved_work(tmp_path):
    script = """
from datetime import UTC, datetime
from pathlib import Path
import sys, time
from evaluation.checkpoints import save
from evaluation.runner import wall_deadline
root = Path(sys.argv[1])
save(root / 'completed.json', {'done': True})
with wall_deadline(root, {'bundle_id': 'deadline-test', 'wall_time_budget_seconds': 1}, datetime.now(UTC).isoformat()):
    time.sleep(8)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=6,
        check=False,
    )
    assert result.returncode == 124, result.stderr
    assert load(tmp_path / "completed.json") == {"done": True}
    assert load(tmp_path / "deadline-expired.json")["status"] == "deadline_reached"
