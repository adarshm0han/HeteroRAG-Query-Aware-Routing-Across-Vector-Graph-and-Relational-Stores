import json
import urllib.error
from copy import deepcopy
from types import SimpleNamespace

import pytest

from routing.catalog import RoutingCatalog, table_metadata, validate_metadata
from routing.evidence_answer import calculate, validate_response
from routing.failures import failure_details
from routing.passage_selection import candidate_links
from routing.reader import CheckedReader
from routing.retrieval import retrieve
from routing.router import DecisionsRouter, RecordedRouter

from .fixtures import QUESTION, FakeDecisions, FakeEmbedding, FakeStores, vector


def test_schema_catalog_does_not_expose_cells_or_answer_fields():
    table = {
        "title": "People",
        "header": [["Name"], ["Medal"]],
        "data": [[["PRIVATE CELL"], ["PRIVATE ANSWER"]]],
        "answer": "SECRET",
    }
    metadata = table_metadata(table, False)
    assert validate_metadata(metadata) == metadata
    assert "PRIVATE" not in json.dumps(metadata) and "SECRET" not in json.dumps(
        metadata
    )
    with pytest.raises(ValueError):
        validate_metadata({**metadata, "answer": "leak"})
    client = FakeDecisions("postgres_pinecone")
    decision = DecisionsRouter(client).choose(QUESTION, table_metadata=metadata)
    payload = json.loads(client.payloads[0]["input"])
    assert payload["table_metadata"] == metadata and payload["graph_available"] is False
    replay = RecordedRouter(
        [{"question": QUESTION, "decision": decision, "routing_metadata": metadata}]
    )
    assert replay.choose(QUESTION, False, metadata)["api_requests"] == 0
    with pytest.raises(ValueError, match="different table metadata"):
        replay.choose(QUESTION, False, {**metadata, "columns": ["Different"]})


def test_catalog_integrity_is_checked_before_lookup(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {"version": "table_schema_catalog_v1", "catalog_id": "wrong", "tables": {}}
        )
    )
    with pytest.raises(ValueError, match="checksum"):
        RoutingCatalog(path)


def test_candidate_discovery_finds_date_before_eight_row_cutoff():
    class Stores(FakeStores):
        def sql_retrieve(self, *args):
            self.record("postgres", "select_table")
            return {
                "title": "People",
                "rows": [
                    {
                        "row_index": i,
                        "text": f"Name: Person{i} | Medal: Gold",
                        "passage_ids": [f"p{i}"],
                        "links": [
                            {
                                "column": "Name",
                                "cell_text": f"Person{i}",
                                "passage_id": f"p{i}",
                            }
                        ],
                    }
                    for i in range(20)
                ],
            }

        def search(self, *args):
            self.record("pinecone", "query")
            return {f"p{i}": self.all_passages[f"p{i}"] for i in range(12)}

        def passages(self, ids):
            assert "pinecone" in self.allowed
            self.fetched.update(ids)
            return {
                "records": {pid: self.all_passages[pid] for pid in ids},
                "missing_ids": [],
            }

    stores = Stores(["postgres", "pinecone"])
    stores.fetched = set()
    stores.all_passages = {
        f"p{i}": {
            "title": f"Person{i}",
            "text": f"Person{i} was born {'April 23' if i == 19 else 'November 19'}, 1965.",
            "values": vector(1.0, i / 20),
        }
        for i in range(20)
    }
    result = retrieve(
        {**QUESTION, "question": "Which medal did the person born April 23, 1965 win?"},
        stores,
        FakeEmbedding(),
    )
    assert "p19" in stores.fetched
    assert 19 in result["row_selection"]["postgres"]["selected_rows"]
    assert any(item["id"] == "p19" for item in result["items"])
    assert result["passage_discovery"]["available"] == 20


def test_discovery_budget_round_robins_links_and_reports_truncation():
    rows = [
        {
            "row_index": i,
            "text": "Name: Person",
            "passage_ids": [f"p{i}-{j}" for j in range(30)],
        }
        for i in range(10)
    ]
    ids, trace = candidate_links("Person", {"postgres": {"rows": rows}}, [], limit=12)
    assert len(ids) == 12 and trace == {"available": 300, "cap": 12, "truncated": True}
    assert {p.split("-")[0] for p in ids[:10]} == {f"p{i}" for i in range(10)}


def test_arithmetic_requires_source_operands_and_rejects_executable_expressions():
    assert calculate("1995 - 1981", ["The term lasted from 1981 to 1995."]) == 14
    for expression in ("1995 - 1980", "__import__('os')", "2 ** 1000000", "1 / 0"):
        with pytest.raises((ValueError, SyntaxError)):
            calculate(expression, ["1995 1981 1 0 2 1000000"])


def test_extractive_answer_cannot_use_uncited_text_or_fabricated_quote():
    excerpts = {
        "E1": {"id": "E1", "evidence_id": "E1", "kind": "row", "text": "Name: One"},
        "E2": {
            "id": "E2",
            "evidence_id": "E2",
            "kind": "passage",
            "text": "Her name was Two.",
        },
    }
    parsed = {
        "reasoning": "It is supported",
        "answer": "Two",
        "excerpts": ["E1"],
        "calculation": "",
    }
    with pytest.raises(ValueError, match="not_in_cited_sources"):
        validate_response(parsed, excerpts, "What name?")
    result = validate_response({**parsed, "excerpts": ["E2"]}, excerpts, "What name?")
    assert result["support_quotes"][0]["quote"] == excerpts["E2"]["text"]
    with pytest.raises(ValueError, match="outside_context"):
        validate_response(
            {**parsed, "answer": "", "excerpts": ["E99"]}, excerpts, "What name?"
        )


def fake_reader(responses):
    replies = iter(responses)
    requests = []

    def request(path, payload):
        requests.append(deepcopy(payload))
        return {
            "done": True,
            "message": {"content": json.dumps(next(replies))},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

    return SimpleNamespace(
        model="local", metadata={"digest": "fixture"}, request=request
    ), requests


def test_independent_review_may_reject_a_validly_quoted_but_wrong_date():
    draft = {
        "reasoning": "A matching person",
        "answer": "Gold",
        "excerpts": ["E1"],
        "calculation": "",
    }
    rejected = {
        **draft,
        "reasoning": "Birth date differs",
        "answer": "",
        "excerpts": [],
    }
    base, calls = fake_reader([draft, rejected])
    evidence = [{"id": "E1", "text": "Ada was born on May 10, 1985 and won Gold."}]
    result = CheckedReader(base).generate(
        "What medal did the person born May 12, 1985 win?", evidence
    )
    assert result["abstain"] and result["draft"]["answer"] == "Gold"
    assert result["usage"] == {"input_tokens": 20, "output_tokens": 10}
    assert len(calls) == 2 and calls[0]["messages"][0] != calls[1]["messages"][0]


def test_reader_does_not_force_answer_into_intermediate_table_column():
    draft = {
        "reasoning": "Return original name",
        "answer": "The Originals",
        "excerpts": ["E2.S1"],
        "calculation": "",
    }
    base, calls = fake_reader([draft, draft])
    evidence = [
        {"id": "E1", "kind": "row", "text": "Performer: The Band"},
        {
            "id": "E2",
            "kind": "passage",
            "text": "The Band was founded as The Originals in 1960.",
        },
    ]
    result = CheckedReader(base).generate(
        "What name was the group founded as?", evidence
    )
    assert result["answer"] == "The Originals"
    assert "enum" not in calls[0]["format"]["properties"]["answer"]


def test_invalid_review_fails_closed_after_one_correction():
    bad = {
        "reasoning": "A claim",
        "answer": "Missing",
        "excerpts": ["E1"],
        "calculation": "",
    }
    base, calls = fake_reader([bad, bad, bad])
    result = CheckedReader(base).generate(
        "Which name?", [{"id": "E1", "text": "Name: One"}]
    )
    assert result["abstain"] and result["answer_check"]["status"] == "rejected"
    assert len(calls) == 3


def test_failure_classification_records_no_messages_or_credentials():
    error = urllib.error.URLError(
        PermissionError(10013, "SECRET connection credential")
    )
    details = failure_details(error)
    assert details["infrastructure"] and details["reason_errno"] == 10013
    assert "SECRET" not in json.dumps(details)
    assert not failure_details(ValueError("invalid plan"))["infrastructure"]


def test_frozen_evaluation_requires_exact_catalog(tmp_path):
    from baseline.common import object_hash, read_json, sha256, write_json
    from evaluation.cohorts import code_hashes, validate_run
    from routing.pipeline import Config

    from .test_evaluation import make_bundle

    bundle, protocol = make_bundle(tmp_path)
    catalog = bundle / "routing-catalog.json"
    write_json(catalog, {"public": "metadata"})
    protocol["routing_catalog"] = {"sha256": sha256(catalog)}
    protocol["inputs"]["routing-catalog.json"] = sha256(catalog)
    protocol["code"] = code_hashes()
    protocol["bundle_id"] = object_hash(
        {k: v for k, v in protocol.items() if k != "bundle_id"}
    )
    write_json(bundle / "protocol.json", protocol)
    args = SimpleNamespace(
        evaluation_bundle=bundle,
        cohort="smoke",
        condition="fixed_all",
        option="all",
        graph_catalog=None,
        routing_catalog=None,
        verify_sources=True,
        continue_on_error=True,
        references=None,
        decision_records=None,
    )
    questions = read_json(bundle / "smoke/questions.json")
    with pytest.raises(ValueError, match="Routing catalog"):
        validate_run(args, Config(), questions)
    args.routing_catalog = catalog
    assert validate_run(args, Config(), questions)["bundle_id"] == protocol["bundle_id"]


def test_spelled_counts_allow_equivalent_digits_but_not_partial_number_phrases():
    def check(answer, text, question="How many teams qualified?"):
        excerpts = {
            "E1": {"id": "E1", "evidence_id": "E1", "kind": "passage", "text": text}
        }
        return validate_response(
            {
                "reasoning": "Count stated directly",
                "answer": answer,
                "excerpts": ["E1"],
                "calculation": "",
            },
            excerpts,
            question,
        )

    assert check("1", "One team qualified.")["answer"] == "1"
    assert check("21", "Twenty-one teams qualified.")["answer"] == "21"
    for answer, source in (
        ("1", "One hundred teams qualified."),
        ("3", "One and two teams."),
        ("1", "Minus one points."),
    ):
        with pytest.raises(ValueError, match="not_in_cited_sources"):
            check(answer, source)
    with pytest.raises(ValueError, match="not_in_cited_sources"):
        check("1", "Her name is One.", "What is her name?")


def test_non_object_draft_can_be_repaired_without_runtime_error():
    valid = {
        "reasoning": "The source states it",
        "answer": "Ada",
        "excerpts": ["E1"],
        "calculation": "",
    }
    base, calls = fake_reader([[], valid])
    result = CheckedReader(base).generate(
        "What name?", [{"id": "E1", "text": "Name: Ada"}]
    )
    assert result["answer"] == "Ada" and len(calls) == 2
