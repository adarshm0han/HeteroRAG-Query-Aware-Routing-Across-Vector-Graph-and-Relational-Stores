"""Opt-in execution on a dedicated local PostgreSQL database, never cloud data."""

import os
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from routing.sql_plan import compile_plan
from routing.stores import LiveStores

from .test_sql_plan import COLUMNS, plan

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LOCAL_SQL_TESTS") != "1",
    reason="Dedicated local PostgreSQL integration test",
)
DSN = f"host=127.0.0.1 port={int(os.environ.get('HETERORAG_TEST_PORT', '55433'))} user=postgres dbname=retrieval_tests connect_timeout=5"


@pytest.fixture(scope="module")
def db():
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(DSN, row_factory=dict_row) as conn:
        conn.execute((Path(__file__).parents[2] / "postgres/schema.sql").read_text())
        conn.execute(
            "INSERT INTO hybridqa.wiki_tables VALUES (1,'test','Scores','local','{}','test') ON CONFLICT DO NOTHING"
        )
        for c in COLUMNS:
            conn.execute(
                "INSERT INTO hybridqa.wiki_columns VALUES (1,%s,%s) ON CONFLICT DO NOTHING",
                (c["column_index"], c["header"]),
            )
        for i in range(20):
            conn.execute(
                "INSERT INTO hybridqa.wiki_rows VALUES (1,%s) ON CONFLICT DO NOTHING",
                (i,),
            )
            score = 1000 if i >= 18 else i
            for col, value, number in (
                (0, f"Player {i}", None),
                (1, str(score), score),
                (2, "France" if i % 2 == 0 else "Canada", None),
            ):
                conn.execute(
                    "INSERT INTO hybridqa.wiki_cells VALUES (1,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    (i, col, value, number),
                )
        conn.commit()
        yield conn


def execute(db, p, k=8):
    sql, params = compile_plan("test", p, COLUMNS, k)
    return db.execute(sql, params).fetchall()


def test_sql_numeric_rank_before_limit_preserves_ties_beyond_first_eight(db):
    p = plan(order_by={"column": 1, "type": "number", "direction": "desc"}, rank=1)
    assert [r["row_index"] for r in execute(db, p)] == [18, 19]
    p["rank"] = 2
    assert [r["row_index"] for r in execute(db, p)] == [17]


def test_sql_filters_sort_and_count_entire_table(db):
    filters = [
        {"column": 2, "type": "text", "op": "eq", "value": "FRANCE"},
        {"column": 1, "type": "number", "op": "gte", "value": "10"},
    ]
    rows = execute(
        db,
        plan(
            filters=filters,
            order_by={"column": 1, "type": "number", "direction": "desc"},
        ),
    )
    assert [r["row_index"] for r in rows] == [18, 16, 14, 12, 10]
    rows = execute(db, plan(operation="count", filters=filters), 1)
    assert rows[0]["value"] == "5" and rows[0]["source_row_count"] == 5
    assert execute(db, plan(operation="count"), 1)[0]["value"] == "20"


@pytest.mark.parametrize(
    "operation,value",
    [
        ("sum", "2153"),
        ("avg", "107.65"),
        ("min", "0"),
        ("max", "1000"),
        ("count_distinct", "19"),
    ],
)
def test_numeric_and_distinct_aggregates(db, operation, value):
    result = execute(db, plan(operation=operation, column=1))[0]
    assert Decimal(result["value"]) == Decimal(value)


def test_grouped_count_having_empty_match_and_literal_contains(db):
    grouped = execute(
        db, plan(operation="count", group_by=2, having={"op": "eq", "value": 10})
    )
    assert {(r["group_value"], r["value"]) for r in grouped} == {
        ("france", "10"),
        ("canada", "10"),
    }
    filters = [{"column": 0, "type": "text", "op": "contains", "value": "%"}]
    assert execute(db, plan(filters=filters)) == []
    assert execute(db, plan(operation="count", filters=filters))[0]["value"] == "0"
    assert (
        execute(db, plan(operation="sum", column=1, filters=filters))[0]["value"]
        is None
    )


def test_real_adapter_schema_plan_and_read_only_execution(db):
    p = plan(operation="count")
    stores = LiveStores(
        {"DATABASE_URL": DSN},
        ["postgres"],
        sql_planner=SimpleNamespace(plan=lambda *args: (p, {})),
    )
    result = stores.sql_retrieve("test", "How many players?", 8)
    assert result["aggregates"][0]["result"]["value"] == "20"
    assert [e["operation"] for e in stores.events if e["kind"] == "data"] == [
        "select_schema",
        "execute_plan",
    ]
    import psycopg

    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        stores.sql_query(
            "DELETE FROM hybridqa.wiki_rows WHERE false", (), "test_read_only"
        )


@pytest.mark.skipif(
    os.environ.get("RUN_LOCAL_QWEN_TESTS") != "1",
    reason="Local Qwen planner integration",
)
@pytest.mark.parametrize(
    "question,expected",
    [
        ("How many players are from France?", "10"),
        ("Which player has the second highest Points?", 17),
    ],
)
def test_real_local_planner_compiles_and_executes(db, question, expected):
    from baseline.local_reader import LocalReader
    from routing.sql_plan import LocalSQLPlanner

    stores = LiveStores(
        {"DATABASE_URL": DSN}, ["postgres"], sql_planner=LocalSQLPlanner(LocalReader())
    )
    result = stores.sql_retrieve("test", question, 8)
    assert result["sql"]["status"] == "executed", result["sql"]
    if isinstance(expected, str):
        assert result["aggregates"][0]["result"]["value"] == expected
    else:
        assert [r["row_index"] for r in result["rows"]] == [expected]
