"""Query planning, injection boundaries, execution, aggregate and path provenance."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from routing.graph_query import merge_paths
from routing.sql_audit import check_aggregate, check_paths
from routing.sql_plan import (
    LocalSQLPlanner,
    aggregate_text,
    compile_plan,
    fallback_plan,
    validate_plan,
)
from routing.stores import LiveStores

COLUMNS = [
    {"column_index": i, "header": name}
    for i, name in enumerate(("Player", "Points", "Country"))
]


def plan(**kwargs):
    return {**fallback_plan(), "operation": "select", **kwargs}


def test_predicates_and_table_ids_are_parameters_not_executable_sql():
    malicious = "x'); DROP TABLE hybridqa.wiki_cells; --"
    p = plan(
        filters=[{"column": 0, "type": "text", "op": "contains", "value": malicious}]
    )
    sql, params = compile_plan(malicious, p, COLUMNS, 8)
    assert malicious not in sql
    assert params == (malicious, malicious, 8)
    assert "strpos" in sql  # literal %, _ and quotes have no wildcard/SQL meaning


@pytest.mark.parametrize(
    "change",
    [
        {"column": 999},
        {"order_by": {"column": 1, "type": "number", "direction": "desc;delete"}},
        {"rank": 0},
        {"rank": 1},
        {"rank": True},
        {"operation": "DROP"},
        {"filters": [{"column": 1, "type": "number", "op": "eq", "value": "nan"}]},
        {"filters": [{"column": True, "type": "text", "op": "eq", "value": "x"}]},
        {"operation": "sum"},
        {"operation": "count", "column": 1},
    ],
)
def test_invalid_plans_are_rejected(change):
    with pytest.raises(ValueError):
        validate_plan(plan(**change), COLUMNS)


def test_planner_uses_local_schema_and_marks_invalid_output_as_fallback():
    requests = []
    reader = SimpleNamespace(
        model="qwen3.5:4b",
        metadata={"digest": "test"},
        request=lambda path, payload: (
            requests.append((path, payload))
            or {
                "done": True,
                "message": {"content": json.dumps(plan(operation="sum", column=999))},
            }
        ),
    )
    result, meta = LocalSQLPlanner(reader).plan(
        "Who scored most?", {"columns": COLUMNS}
    )
    assert result == fallback_plan()
    assert meta["fallback_reason"] == "invalid_or_incomplete_local_plan"
    assert meta["api_cost_usd"] == 0 and meta["local_inference"]
    assert requests[0][1]["model"] == "qwen3.5:4b"


def test_adapter_compiles_full_table_plan_without_lexical_reranking():
    p = plan(order_by={"column": 1, "type": "number", "direction": "desc"}, rank=2)
    planner = SimpleNamespace(plan=lambda *args: (p, {"seconds": 0.1}))
    stores = LiveStores({}, ["postgres"], sql_planner=planner)
    calls = []

    def query(sql, params, operation):
        calls.append((sql, params, operation))
        if operation == "select_schema":
            return [{**c, "title": "Scores", "examples": []} for c in COLUMNS]
        return [
            {
                "title": "Scores",
                "row_index": 14,
                "evidence_text": "Player: Target | Points: 100",
                "cells": [],
                "position": 2,
                "result_rows": 1,
            }
        ]

    stores.sql_query = query
    result = stores.sql_retrieve("scores", "Second most points?", 8)
    assert result["ordered"] and result["rows"][0]["row_index"] == 14
    assert "dense_rank()" in calls[1][0]
    assert calls[1][0].index("WHERE position") < calls[1][0].index("LIMIT")
    assert calls[1][1] == ("scores", 2, 8)


def aggregate_item(table, p, result):
    cols = [{"column_index": i, "header": h[0]} for i, h in enumerate(table["header"])]
    return {
        "table_id": "test",
        "kind": "aggregate",
        "plan": p,
        "result": result,
        "text": aggregate_text(p, result, cols),
    }


def test_aggregate_audit_uses_all_rows_and_rejects_wrong_values():
    table = {
        "header": [["Name", []], ["Score", []]],
        "data": [[[str(i), []], [str(i), []]] for i in range(20)],
    }
    p = plan(
        operation="count",
        filters=[{"column": 1, "type": "number", "op": "gte", "value": "10"}],
    )
    result = {"group_value": None, "value": "10", "source_row_count": 10}
    item = aggregate_item(table, p, result)
    check_aggregate("test", table, item)
    item["result"]["value"] = "8"
    with pytest.raises(ValueError, match="differs"):
        check_aggregate("test", table, item)


def test_graph_merge_and_audit_keep_shared_entity_path_not_invented_edges():
    table = {"data": [[["A", ["/wiki/Shared"]]], [["B", ["/wiki/Shared", "/wiki/B"]]]]}
    seeds = [{"row_index": 0, "links": [{"passage_id": "/wiki/Shared"}]}]
    neighbors = [
        {
            "row_index": 1,
            "links": [{"passage_id": "/wiki/B"}],
            "paths": [
                {
                    "seed_row_index": 0,
                    "row_index": 1,
                    "bridge_passage_id": "/wiki/Shared",
                }
            ],
        }
    ]
    rows, truncated = merge_paths("test", seeds, neighbors, 8)
    assert [r["row_index"] for r in rows] == [0, 1] and not truncated
    item = {**rows[1], "graph_paths": rows[1]["paths"]}
    check_paths(table, item)
    bad = deepcopy(item)
    bad["graph_paths"][0]["bridge_passage_id"] = "/wiki/Invented"
    with pytest.raises(ValueError, match="absent"):
        check_paths(table, bad)


def test_structured_selection_order_is_used_for_graph_seeds_and_evidence():
    from routing.retrieval import retrieve

    from .fixtures import QUESTION, ROWS, FakeEmbedding, FakeStores

    stores = FakeStores(["postgres", "neo4j"])
    stores.sql_retrieve = lambda *args: {
        "rows": list(reversed(ROWS)),
        "title": "Scores",
        "ordered": True,
    }
    captured = {}

    def graph(*args, **kwargs):
        captured.update(kwargs)
        return {"rows": [], "title": "Scores", "covered": True, "ordered": True}

    stores.graph_table = graph
    result = retrieve(QUESTION, stores, FakeEmbedding())
    assert captured["seed_rows"] == [1, 0]
    assert result["rankings"]["postgres"] == [
        "row:synthetic-table:1",
        "row:synthetic-table:0",
    ]


def test_computed_evidence_is_packed_audited_and_counted_within_row_budget(tmp_path):
    from routing.audit import EvidenceAudit

    from .test_audit_report import source_archive
    from .test_pipeline import pipeline

    p = plan(operation="count")
    table = {"header": [["Player", []], ["Points", []]], "data": []}
    item = aggregate_item(
        table, p, {"group_value": None, "value": "2", "source_row_count": 2}
    )
    item.update(
        id="sql:synthetic-table:0", table_id="synthetic-table", sources=["postgres"]
    )
    app, _, _ = pipeline("postgres")
    original = app.stores_factory

    def factory(selected):
        stores = original(selected)
        stores.sql_retrieve = lambda *args: {
            "rows": [],
            "aggregates": [item],
            "title": "Scores",
            "ordered": True,
        }
        return stores

    app.stores_factory = factory
    app.auditor = EvidenceAudit(source_archive(tmp_path))
    from .fixtures import QUESTION

    try:
        result = app.run(QUESTION)
        assert result["status"] == "completed", result.get("error")
        assert result["source_audit"]["aggregates"] == 1
        assert result["retrieval"]["evidence"][0]["kind"] == "aggregate"
    finally:
        app.auditor.close()
