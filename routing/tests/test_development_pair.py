import pytest

from baseline.common import object_hash, read_json, write_json
from evaluation import paired as development_pair
from evaluation.paired import run, schedule, select_fresh


def test_selection_excludes_prior_questions_and_cannot_read_answer_labels():
    questions = [
        {
            "question_id": str(i),
            "table_id": "t",
            "question": f"Question {i}",
            "answer-text": "private",
        }
        for i in range(90)
    ]
    excluded = {str(i) for i in range(30)}
    chosen = select_fresh(questions, excluded)
    assert len(chosen) == 50 and not ({q["question_id"] for q in chosen} & excluded)
    assert all(set(q) == {"question_id", "table_id", "question"} for q in chosen)
    assert chosen == select_fresh(list(reversed(questions)), excluded)
    with pytest.raises(ValueError):
        select_fresh(questions, {str(i) for i in range(50)})


def test_paired_schedule_limits_paid_choices_and_balances_order():
    jobs = schedule(50)
    assert len(jobs) == 100
    assert sum(j["condition"] == "routed" for j in jobs) == 50
    assert sum(jobs[i]["condition"] == "all" for i in range(0, 100, 2)) == 25
    assert jobs[0] == {"index": 0, "condition": "all"}
    assert len({(j["index"], j["condition"]) for j in jobs}) == 100


def test_paid_authorization_gate_precedes_files_and_network(tmp_path):
    with pytest.raises(ValueError, match="allow-decisions-api"):
        run(tmp_path / "absent", tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


@pytest.mark.parametrize("n", [20, 40, 50])
def test_frozen_cohort_and_budget_follow_requested_size(tmp_path, monkeypatch, n):
    monkeypatch.setattr(development_pair, "REPO", tmp_path)
    monkeypatch.setattr(
        development_pair, "code_hashes", lambda: {"example.py": "fixed"}
    )
    monkeypatch.setattr(
        development_pair,
        "load_bundle",
        lambda *args, **kwargs: {
            "bundle_id": "source",
            "exposed_dev_exclusions": [],
            "reader_digest": "model",
        },
    )
    source = tmp_path / "source"
    questions = [
        {"question_id": str(i), "question": f"Question {i}", "table_id": "t"}
        for i in range(60)
    ]
    write_json(source / "smoke/questions.json", questions[:2])
    write_json(source / "development/questions.json", questions)
    write_json(
        source / "development/references.json",
        {q["question_id"]: {"answer": "reference"} for q in questions},
    )
    bundle = tmp_path / "bundle"
    plan = development_pair.prepare(bundle, source, n)
    verified, selected = development_pair.verify(bundle)
    assert verified == plan
    assert len(selected) == plan["questions"] == plan["max_decisions_requests"] == n
    jobs = read_json(bundle / "schedule.json")
    assert len(jobs) == 2 * n
    assert sum(job["condition"] == "routed" for job in jobs) == n
    # Even a rehashed protocol cannot silently expand the paid request budget.
    plan["max_decisions_requests"] = n + 1
    plan["bundle_id"] = object_hash({k: v for k, v in plan.items() if k != "bundle_id"})
    write_json(bundle / "protocol.json", plan)
    with pytest.raises(ValueError, match="Cohort size"):
        development_pair.verify(bundle)


@pytest.mark.parametrize("n", [0, -1, 51, True, 1.5])
def test_invalid_question_count_is_rejected(n):
    with pytest.raises(ValueError, match="Question count"):
        select_fresh([], set(), n)
