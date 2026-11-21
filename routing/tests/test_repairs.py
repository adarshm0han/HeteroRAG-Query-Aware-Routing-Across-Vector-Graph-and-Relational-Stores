"""Regression checks for failures identified in the saved development traces."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from routing.answer_validation import (
    check_explicit_property,
    resolve_quote,
    restore_requested_unit,
)
from routing.fusion import fuse
from routing.graph_query import combine_seeds
from routing.row_selection import reserve_passages, row_order
from routing.sql_plan import LocalSQLPlanner, compile_plan, fallback_plan, plan_guard
from routing.stores import LiveStores

from .fixtures import ROWS
from .test_sql_plan import COLUMNS, plan


def test_unfiltered_selection_and_non_iso_date_order_are_not_execution_strategies():
    assert plan_guard(plan(), COLUMNS) == "unfiltered_unordered_selection"
    dates = [
        {
            "column_index": 0,
            "header": "Date",
            "examples": ["4 October 1957", "3 November 1957"],
        }
    ]
    ordered = plan(order_by={"column": 0, "type": "text", "direction": "desc"}, rank=1)
    assert plan_guard(ordered, dates) == "unsupported_text_date_order"
    dates[0]["examples"] = ["1957-10-04", "1957-11-03"]
    assert plan_guard(ordered, dates) is None


def test_semantic_plan_rejection_retains_proposal_and_counts_review_tokens():
    proposed = plan(
        filters=[{"column": 1, "type": "number", "op": "eq", "value": "376"}]
    )
    responses = iter([proposed, {"valid": False, "reason": "Points are not density."}])
    requests = []

    def request(path, payload):
        requests.append(payload)
        return {
            "done": True,
            "message": {"content": json.dumps(next(responses))},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

    result, metadata = LocalSQLPlanner(
        SimpleNamespace(model="local", metadata={"digest": "test"}, request=request)
    ).plan("Which country has density 376?", {"columns": COLUMNS})
    assert result == fallback_plan()
    assert metadata["proposed_plan"] == proposed
    assert metadata["fallback_reason"] == "semantic_review_rejected"
    assert metadata["usage"] == {"input_tokens": 20, "output_tokens": 10}
    assert len(requests) == 2


def test_empty_sql_selection_falls_back_without_changing_zero_count_semantics():
    p = plan(filters=[{"column": 0, "type": "text", "op": "eq", "value": "unknown"}])
    stores = LiveStores(
        {}, ["postgres"], sql_planner=SimpleNamespace(plan=lambda *a: (p, {}))
    )
    stores.sql_query = lambda sql, params, operation: (
        [{**c, "title": "Scores", "examples": []} for c in COLUMNS]
        if operation == "select_schema"
        else []
    )
    stores.sql_table = lambda tid: {"rows": deepcopy(ROWS), "title": "Scores"}
    result = stores.sql_retrieve("test", "Which player?", 1)
    assert result["sql"]["status"] == "empty_selection_fallback"
    assert len(result["rows"]) == 2 and not result["selection_grounded"]


def test_unordered_filter_compiles_without_early_evidence_limit():
    p = plan(filters=[{"column": 2, "type": "text", "op": "eq", "value": "France"}])
    sql, params = compile_plan("t", p, COLUMNS, None)
    assert "LIMIT" not in sql and params == ("t", "France")
    with pytest.raises(ValueError):
        compile_plan("t", plan(operation="count"), COLUMNS, None)


def test_passage_to_row_lookup_keeps_repeated_entities_beyond_first_eight():
    rows = [
        {
            "row_index": i,
            "text": f"Name: Item {i}",
            "passage_ids": ["/wiki/Target"] if i in (5, 11, 19) else ["/wiki/Other"],
        }
        for i in range(24)
    ]
    assert set(
        row_order(
            "Which entity occurs across decades?",
            {"rows": rows},
            ["/wiki/Target", "/wiki/Other"],
        )[:3]
    ) == {5, 11, 19}
    assert (
        row_order(
            "q", {"rows": list(reversed(rows)), "ordered": True}, ["/wiki/Target"]
        )[0]
        == 23
    )


def test_linkless_occurrences_and_explicit_cell_conditions_survive_selection():
    rows = [
        {
            "row_index": i,
            "text": f"School: {'Target' if i in (5, 11, 19) else 'Other'} | Year: {1980 + i}",
            "passage_ids": [],
            "links": [],
        }
        for i in range(24)
    ]
    rows[5]["passage_ids"] = ["/wiki/Target"]
    rows[5]["links"] = [
        {"column": "School", "cell_text": "Target", "passage_id": "/wiki/Target"}
    ]
    assert set(
        row_order(
            "Which school occurs across decades?", {"rows": rows}, ["/wiki/Target"]
        )[:3]
    ) == {5, 11, 19}
    assert (
        row_order("Which school was listed in 2003?", {"rows": rows}, ["/wiki/Target"])[
            0
        ]
        == 23
    )


def test_explicit_row_passage_is_kept_even_below_semantic_top_k():
    semantic = [f"/wiki/Noise{i}" for i in range(20)] + ["/wiki/Target"]
    row = {
        "text": "Name: Target | Date: 24 March 2011",
        "passage_ids": ["/wiki/Target"],
    }
    kept = reserve_passages(
        "Who was listed on 24 March 2011?", semantic[:12], semantic, [row], 12
    )
    assert len(kept) == 12 and kept[0] == "/wiki/Target"
    assert "/wiki/Noise0" in kept


def test_sql_seed_pool_cannot_remove_an_independent_graph_anchor():
    independent = [{"row_index": i} for i in (15, 9, 12, 10)]
    sql = [{"row_index": i} for i in (0, 1)]
    assert [r["row_index"] for r in combine_seeds(independent, sql)] == [
        15,
        9,
        12,
        10,
        0,
        1,
    ]


@pytest.mark.parametrize("mode", ["rrf", "concat", "weighted_rrf"])
def test_duplicate_votes_cannot_evict_independent_semantic_passages(mode):
    semantic = [f"p{i}" for i in range(12)]
    distracting = [f"d{i}" for i in range(12)]
    records = {pid: {"id": pid, "kind": "passage"} for pid in semantic + distracting}
    rankings = {"postgres": distracting, "neo4j": distracting, "pinecone": semantic}
    result = fuse(
        rankings,
        records,
        mode=mode,
        weights={"postgres": 0.8, "neo4j": 0.1, "pinecone": 0.1},
        protected=semantic,
    )
    assert {r["id"] for r in result} == set(semantic)


def test_quote_validation_allows_tokenization_and_verified_omissions_only():
    text = "Capsule A was launched in 1957 . It was 13 feet tall ."
    assert resolve_quote("launched in 1957.", text) == "launched in 1957 ."
    assert (
        resolve_quote("Capsule A ... 13 feet tall", text)
        == "Capsule A was launched in 1957 . It was 13 feet tall"
    )
    with pytest.raises(ValueError):
        resolve_quote("Capsule A ... 14 feet tall", text)
    answer = {"answer": "13", "support_quotes": [{"quote": text}]}
    assert (
        restore_requested_unit(answer, "How tall in feet was it?")["answer"]
        == "13 feet"
    )
    assert (
        restore_requested_unit(answer, "How tall in meters was it?")["answer"] == "13"
    )


def test_explicit_property_guards_reject_intermediate_names_without_labels():
    evidence = [{"id": "E1", "text": "The Band was founded as The Originals in 1960."}]
    answer = {"answer": "The Band", "citations": ["E1"], "abstain": False}
    with pytest.raises(ValueError, match="founding_name"):
        check_explicit_property("What name was the band founded as?", answer, evidence)
    check_explicit_property(
        "What name was the band founded as?",
        {**answer, "answer": "The Originals"},
        evidence,
    )
    with pytest.raises(ValueError, match="missing_explicit_founding_name_evidence"):
        check_explicit_property(
            "What name was the band founded as?",
            answer,
            [{"id": "E1", "text": "The Band released a record in 1960."}],
        )


def test_explicit_quantity_guard_uses_the_measurement_clause():
    answer = {"answer": "volt", "citations": ["E1"], "abstain": False}
    evidence = [{"id": "E1", "text": "The volt is a unit of electric potential."}]
    with pytest.raises(ValueError, match="measured_quantity"):
        check_explicit_property(
            "What quantity does the unit measure?",
            {**answer, "answer": "volt"},
            evidence,
        )
    check_explicit_property(
        "What quantity does the unit measure?",
        {**answer, "answer": "electric potential"},
        evidence,
    )
