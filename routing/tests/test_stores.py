"""Test the real adapter contracts with stubbed SDK responses and no network."""

from types import SimpleNamespace

import pytest

from routing.stores import LiveStores

from .fixtures import PASSAGES, ROWS, vector


def test_fetch_telemetry_accepts_the_sdk_usage_model_without_failing_retrieval():
    from pinecone.core.openapi.db_data.model.usage import Usage
    from pinecone.db_data.dataclasses import FetchResponse, Vector

    record = Vector(
        id="/wiki/Ada",
        values=vector(1.0),
        metadata={"passage_id": "/wiki/Ada", **PASSAGES["/wiki/Ada"]},
    )
    # Pinecone 7.3 converts vectors to dataclasses but keeps the OpenAPI Usage
    # object; a namespace/dict-only fake does not reproduce this contract.
    response = FetchResponse(
        namespace="", vectors={"/wiki/Ada": record}, usage=Usage(read_units=1)
    )
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return response

    stores = LiveStores({}, ["pinecone"])
    stores.index = SimpleNamespace(fetch=fetch)
    result = stores.passages(["/wiki/Ada"])
    assert len(calls) == 1
    assert result["missing_ids"] == []
    assert result["records"]["/wiki/Ada"]["text"] == record.metadata["text"]
    assert result["records"]["/wiki/Ada"]["values"] == list(record.values)
    assert stores.events[0]["status"] == "completed"
    assert stores.events[0]["read_units"] == 1
    assert stores.events[0]["returned_records"] == 1
    assert stores.events[0]["response_bytes_estimate"] > len(record.metadata["text"])


def test_sql_uses_read_only_parameterized_select_and_counts_actual_attempts(
    monkeypatch,
):
    import psycopg

    observed = {"commands": []}

    class Cursor:
        async def fetchall(self):
            return [
                {
                    "title": "Scores",
                    "row_index": 0,
                    "evidence_text": ROWS[0]["text"],
                    "cells": [
                        {
                            "column_index": 0,
                            "header": "Player",
                            "value": "Ada",
                            "entity_ids": ["/wiki/Ada"],
                        }
                    ],
                }
            ]

    class Connection:
        async def execute(self, sql, params=None):
            observed["commands"].append(sql)
            if sql.startswith("SELECT"):
                observed.update(sql=sql, params=params)
                return Cursor()

        async def close(self):
            observed["closed"] = True

    async def connect(url, **kwargs):
        observed["connection"] = kwargs
        return Connection()

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", connect)
    stores = LiveStores({"DATABASE_URL": "not-a-real-connection"}, ["postgres"])
    table = stores.sql_table("unsafe' table")
    assert observed["params"] == ("unsafe' table",)
    assert "unsafe" not in observed["sql"]
    assert "options" not in observed["connection"]
    assert observed["commands"][:2] == [
        "BEGIN READ ONLY",
        "SET LOCAL statement_timeout = '30s'",
    ]
    assert observed["commands"][-1] == "ROLLBACK"
    assert observed["connection"]["autocommit"] is True
    assert observed["closed"] is True
    assert table["rows"][0]["passage_ids"] == ["/wiki/Ada"]
    assert [(event["operation"], event["status"]) for event in stores.events] == [
        ("connect", "completed"),
        ("begin_read_only", "completed"),
        ("set_timeout", "completed"),
        ("select_table", "completed"),
        ("rollback", "completed"),
    ]


def test_failed_sql_connection_does_not_claim_a_select_happened(monkeypatch):
    import psycopg

    async def connect(*args, **kwargs):
        raise psycopg.OperationalError("Do not expose a secret connection string")

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", connect)
    stores = LiveStores({"DATABASE_URL": "unused"}, ["postgres"])
    with pytest.raises(psycopg.OperationalError):
        stores.sql_table("table")
    assert len(stores.events) == 1
    assert stores.events[0]["operation"] == "connect"
    assert stores.events[0]["error"] == {"type": "OperationalError"}


def test_graph_query_uses_passage_anchors_and_keeps_links(monkeypatch):
    import neo4j

    observed = {}
    queries = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def run(self, query, **params):
            observed.update(query=query.text, params=params)
            queries.append((query.text, params))
            if "UNWIND $seeds" in query.text:
                return SimpleNamespace(data=list)
            return SimpleNamespace(
                data=lambda: [
                    {
                        "title": "Scores",
                        "row_index": 0,
                        "text": ROWS[0]["text"],
                        "links": ROWS[0]["links"],
                        "anchor_hits": 1,
                    }
                ]
            )

    class Driver:
        def session(self, **kwargs):
            observed["session"] = kwargs
            return Session()

        def close(self):
            pass

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *args, **kwargs: Driver())
    stores = LiveStores(
        {"NEO4J_URI": "unused", "NEO4J_USERNAME": "u", "NEO4J_PASSWORD": "p"}, ["neo4j"]
    )
    result = stores.graph_table(
        "table", ["/wiki/Ada"], question="Who has points?", seed_rows=[0]
    )
    assert queries[0][1]["anchors"] == ["/wiki/Ada"]
    assert queries[0][1]["seed_rows"] == [0]
    assert "WHERE r.row_index IN $seed_rows" not in queries[0][0]
    assert "WHERE r.row_index IN $seed_rows" in queries[1][0]
    assert queries[2][1]["seeds"] == [0]
    assert queries[2][1]["neighbor_limit"] == 8
    assert "(r:TableRow)<-[:HAS_ROW]-(t)" in queries[2][0]
    assert observed["session"]["default_access_mode"] == neo4j.READ_ACCESS
    assert result["rows"][0]["anchor_hits"] == 1
    assert result["rows"][0]["passage_ids"] == ["/wiki/Ada"]
    assert [e["operation"] for e in stores.events] == [
        "select_seeds",
        "select_sql_seeds",
        "expand_shared_passages",
    ]
    assert result["rows"][0]["paths"][0]["seed_row_index"] == 0


def test_pinecone_only_uses_metadata_filter_and_tracks_setup_separately(monkeypatch):
    import pinecone

    observed = {}
    record = SimpleNamespace(
        metadata={"passage_id": "/wiki/Ada", **PASSAGES["/wiki/Ada"]},
        values=vector(1.0),
    )

    class Index:
        def query(self, **kwargs):
            observed["query"] = kwargs
            return SimpleNamespace(matches=[record], usage={"read_units": 1})

        def fetch(self, **kwargs):
            observed["fetch"] = kwargs
            return SimpleNamespace(
                vectors={"/wiki/Ada": record}, usage={"read_units": 1}
            )

        def close(self):
            pass

    class Pinecone:
        def __init__(self, **kwargs):
            self._openapi_config = SimpleNamespace(retries=None)
            self.db = SimpleNamespace(
                index=SimpleNamespace(
                    _index_api=SimpleNamespace(
                        describe_index=self.describe_index,
                        api_client=SimpleNamespace(close=lambda: None),
                    )
                )
            )

        def describe_index(self, index_name, **kwargs):
            assert self._openapi_config.retries == 0
            return SimpleNamespace(host="unused", dimension=384, metric="cosine")

        def Index(self, host):
            return Index()

    monkeypatch.setattr(pinecone, "Pinecone", Pinecone)
    stores = LiveStores({"PINECONE_API_KEY": "fake"}, ["pinecone"])
    result = stores.search("provided-table", vector(1.0), 12)
    assert result["/wiki/Ada"]["text"] == PASSAGES["/wiki/Ada"]["text"]
    assert observed["query"]["filter"] == {"table_ids": {"$in": ["provided-table"]}}
    assert observed["query"]["include_values"] is True
    fetched = stores.passages(["/wiki/Ada", "/wiki/Missing"])
    assert fetched["missing_ids"] == ["/wiki/Missing"]
    assert [event["kind"] for event in stores.events] == ["control", "data", "data"]
    assert sum(event.get("read_units", 0) for event in stores.events) == 2


def test_installed_pinecone_endpoint_accepts_exact_adapter_arguments(monkeypatch):
    from pinecone import Pinecone

    pc = Pinecone(api_key="synthetic-no-network")
    endpoint = pc.db.index._index_api.describe_index
    observed = {}

    def intercept(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(host="unused", dimension=384, metric="cosine")

    monkeypatch.setattr(endpoint, "call_with_http_info", intercept)
    response = endpoint(index_name="synthetic", _request_timeout=(10, 30))
    assert response.dimension == 384
    assert observed["index_name"] == "synthetic"
    assert observed["_request_timeout"] == (10, 30)
    pc.db.index._index_api.api_client.close()
