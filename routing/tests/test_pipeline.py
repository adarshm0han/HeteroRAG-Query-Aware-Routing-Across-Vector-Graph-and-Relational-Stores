"""Exercise routing, subset isolation, fusion, packing and reader contracts offline."""

import json
import os

import pytest

from routing.options import OPTIONS
from routing.pipeline import Config, Pipeline
from routing.router import DecisionsRouter, FixedRouter, RecordedRouter
from routing.stores import LiveStores

from .fixtures import (
    QUESTION,
    FakeDecisions,
    FakeEmbedding,
    FakeReader,
    FakeStores,
    FakeTokenizer,
)


def pipeline(option, **store_args):
    client, embedding, reader = FakeDecisions(option), FakeEmbedding(), FakeReader()
    instances = []

    def factory(allowed):
        store = FakeStores(allowed, **store_args)
        instances.append(store)
        return store

    return (
        Pipeline(DecisionsRouter(client), factory, embedding, reader, FakeTokenizer()),
        client,
        instances,
    )


@pytest.mark.parametrize("option", OPTIONS)
def test_each_decisions_choice_executes_only_its_guarded_stores(option):
    app, client, instances = pipeline(option)
    result = app.run(
        {**QUESTION, "answer-text": "SECRET GOLD", "answer-node": ["SECRET TRACE"]}
    )
    assert result["status"] == "completed"
    selected = set(OPTIONS[option]) | {"pinecone"}
    assert set(result["selected_stores"]) == selected
    assert result["decision"]["raw_option"] == option
    assert result["decision"]["evidence_policy"]["stage"] == "before_retrieval"
    assert {call["store"] for call in result["database_calls"]} == selected
    assert instances[0].closed
    assert len(client.payloads) == 1
    assert "SECRET" not in json.dumps(client.payloads)
    assert "SECRET" not in result["retrieval"]["context"]
    assert app.embedding.calls == 1
    assert app.reader.calls == 1
    evidence = result["retrieval"]["evidence"]
    assert all(set(item["sources"]) <= selected for item in evidence)
    assert len({item["source_id"] for item in evidence}) == len(evidence)
    assert set(result["answer"]["citations"]) <= {item["id"] for item in evidence}
    if option in ("postgres_pinecone", "neo4j_pinecone", "all"):
        assert result["answer"]["answer"] == "Rocket"


def test_missing_graph_evidence_abstains_without_querying_other_stores():
    app, _, _ = pipeline("neo4j", graph_covered=False)
    app.router = FixedRouter("neo4j")
    result = app.run(QUESTION)
    assert result["status"] == "empty_evidence"
    assert result["answer"]["abstain"]
    assert app.reader.calls == 0
    assert result["retrieval"]["graph_covered"] is False
    assert {event["store"] for event in result["database_calls"]} == {"neo4j"}


def test_failed_backend_is_not_converted_into_empty_evidence_or_hidden_fallback():
    app, _, instances = pipeline("postgres", fail="postgres")
    result = app.run(QUESTION)
    assert result["status"] == "failed"
    assert result["failed_stage"] == "retrieval"
    assert [
        event["store"]
        for event in result["database_calls"]
        if event["status"] == "failed"
    ] == ["postgres"]
    assert result["error"] == {"type": "TimeoutError"}
    assert app.reader.calls == 0
    assert instances[0].closed


def test_invalid_decision_opens_no_databases():
    app, _, instances = pipeline("not_an_option")
    result = app.run(QUESTION)
    assert result["failed_stage"] == "routing"
    assert result["database_calls"] == []
    assert not instances


def test_failed_recorded_routing_is_preserved_without_new_requests():
    app, _, instances = pipeline("all")
    app.router = RecordedRouter(
        [
            {
                "question": QUESTION,
                "status": "failed",
                "failed_stage": "routing",
                "decisions_calls": [{"status": "failed"}],
            }
        ]
    )
    result = app.run(QUESTION)
    assert result["status"] == "failed" and result["failed_stage"] == "routing"
    assert result["route_replayed"] is True
    assert not instances and result["decisions_calls"] == []


def test_unavailable_graph_is_removed_from_decisions_choices():
    app, client, _ = pipeline("postgres_pinecone")
    result = app.run(QUESTION, graph_available=False)
    assert result["status"] == "completed"
    choices = client.payloads[0]["questions"][0]["choices"]
    assert {c["value"] for c in choices} == {
        "postgres",
        "pinecone",
        "postgres_pinecone",
    }


@pytest.mark.parametrize("option", OPTIONS)
def test_live_adapter_rejects_unselected_stores_before_credentials_or_connections(
    option,
):
    stores = LiveStores({}, OPTIONS[option])
    operations = {
        "postgres": lambda: stores.sql_table("table"),
        "neo4j": lambda: stores.graph_table("table"),
        "pinecone": lambda: stores.search("table", [], 1),
    }
    assert stores.graph is None and stores.index is None and stores.pc is None
    for store in set(OPTIONS["all"]) - set(OPTIONS[option]):
        with pytest.raises(PermissionError):
            operations[store]()
    assert stores.events == []


def test_config_and_fixed_option_validation():
    with pytest.raises(ValueError):
        Config(row_k=0)
    with pytest.raises(ValueError):
        FixedRouter("not_an_option")


def test_weighted_fusion_executes_same_subset_with_bounded_context():
    app, _, _ = pipeline("all")
    app.config = Config(fusion="weighted_rrf", context_tokens=1500)
    result = app.run(QUESTION)
    assert result["status"] == "completed"
    assert result["retrieval"]["fusion"] == "weighted_rrf"
    assert result["retrieval"]["context_tokens"] <= 1500
    assert set(result["selected_stores"]) == {"postgres", "neo4j", "pinecone"}


@pytest.mark.skipif(
    os.environ.get("RUN_LOCAL_QWEN_TESTS") != "1",
    reason="Explicit local Ollama integration test",
)
@pytest.mark.parametrize("option", OPTIONS)
def test_local_qwen_full_flow_with_simulated_decisions_and_stores(option):
    from baseline.local_reader import LocalReader

    app, _, _ = pipeline(option)
    app.reader = LocalReader("qwen3.5:4b")
    result = app.run(QUESTION)
    assert result["status"] == "completed", result.get("error")
    assert result["answer"]["model"] == "qwen3.5:4b"
    assert result["answer"]["local_inference"] is True
    assert result["answer"]["api_cost_usd"] == 0
    assert all(event["status"] == "simulated" for event in result["decisions_calls"])
    if option in ("postgres_pinecone", "neo4j_pinecone", "all"):
        assert result["answer"]["answer"].casefold() == "rocket"
