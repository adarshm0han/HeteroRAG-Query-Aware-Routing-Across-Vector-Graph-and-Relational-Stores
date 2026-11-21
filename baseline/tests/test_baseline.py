import json
import zipfile

import pytest

from baseline.common import (
    answer_scores,
    public_question,
    select_questions,
)
from baseline.reader import Reader
from baseline.report import cluster_interval, evidence_diagnostics
from baseline.retrieval import build_plans, pack_context, rrf, vector_rank


def test_sampling_order_independent_and_answer_blind():
    questions = [
        {
            "question_id": str(i),
            "question": "q",
            "table_id": "t",
            "answer-text": "secret",
        }
        for i in range(100)
    ]
    chosen = select_questions(questions, 20, "seed")
    changed = [{**q, "answer-text": "different"} for q in reversed(questions)]
    assert [q["question_id"] for q in chosen] == [
        q["question_id"] for q in select_questions(changed, 20, "seed")
    ]
    assert all(
        set(public_question(q)) == {"question_id", "question", "table_id"}
        for q in chosen
    )


@pytest.mark.parametrize(
    "gold,pred,em,f1",
    [
        ("The Ghurids", "Ghurids", 1, 1),
        ("New York", "York", 0, 2 / 3),
        ("January 1, 1950", "January 1 1950", 1, 1),
        ("", "", 1, 1),
        ("a cat", "", 0, 0),
        ("red red blue", "red blue blue", 0, 2 / 3),
    ],
)
def test_official_metrics(gold, pred, em, f1):
    result = answer_scores(gold, pred)
    assert result["em"] == em
    assert result["f1"] == pytest.approx(f1)


def test_duplicate_source_votes_do_not_inflate_rrf():
    assert rrf([["x", "x", "y"], ["y"]]) == ["y", "x"]


def fixtures():
    rows = [
        {
            "table_id": "t",
            "row_index": i,
            "text": name,
            "links": [],
            "passage_ids": [pid],
        }
        for i, name, pid in [
            (0, "Alpha author", "/wiki/Alpha"),
            (1, "Beta winner", "/wiki/Beta"),
        ]
    ]
    passages = {
        pid: {
            "title": pid,
            "text": text,
            "values": [1.0 if j == i else 0.0 for j in range(384)],
        }
        for i, pid, text in [
            (0, "/wiki/Alpha", "Born 1900"),
            (1, "/wiki/Beta", "Born 2000"),
        ]
    }
    return rows, passages


def test_vector_ranking_and_absent_graph_are_valid():
    rows, passages = fixtures()
    query = [0.0, 1.0] + [0.0] * 382
    assert vector_rank(query, passages)[0] == "/wiki/Beta"
    plans = build_plans(
        "Which author?", rows, [], passages, query, row_k=1, passage_k=1
    )
    assert plans["sql_only"][0]["row_index"] == 0
    assert plans["vector_only"][0]["passage_id"] == "/wiki/Beta"
    assert plans["sql_links"][-1]["passage_id"] == "/wiki/Alpha"
    assert plans["all_stores"]


def test_context_budget_keeps_whole_evidence():
    class Encoder:
        def encode(self, s):
            return list(s)

    items = [
        {"id": "small", "kind": "passage", "text": "short"},
        {"id": "big", "kind": "passage", "text": "x" * 1000},
    ]
    text, selected, count = pack_context("q", "t", items, Encoder(), 200)
    assert [e["id"] for e in selected] == ["E1"]
    assert selected[0]["source_id"] == "small"
    assert count == len(text) <= 200


def test_weak_trace_diagnostic_does_not_require_all_answer_candidates():
    ref = {
        "answer": "1900",
        "trace_candidates": [
            ["a", [0, 0], "/wiki/Alpha", "passage"],
            ["b", [1, 0], "/wiki/Beta", "passage"],
        ],
    }
    d = evidence_diagnostics(
        ref, [{"kind": "passage", "passage_id": "/wiki/Alpha", "text": "born 1900"}]
    )
    assert d == {"weak_trace_any_hit": True, "answer_string_in_context": True}


def test_cluster_bootstrap_handles_repeated_tables():
    assert cluster_interval([0.5, 0.5, 0.5], ["a", "a", "b"], 100) == [0.5, 0.5]


def test_cost_uses_included_cached_and_write_tokens_once():
    cost = Reader.price(
        {
            "input_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 300},
            "output_tokens": 100,
        }
    )
    assert cost == pytest.approx(
        (500 * 0.10 + 200 * 0.01 + 300 * 0.125 + 100 * 0.50) / 1e6
    )


def test_source_audit_rejects_corruption(tmp_path):
    from baseline.audit import SourceAudit
    from baseline.common import CORPUS_REVISION

    path = tmp_path / "corpus.zip"
    with zipfile.ZipFile(path, "w") as z:
        prefix = f"WikiTables-WithLinks-{CORPUS_REVISION}/"
        z.writestr(
            prefix + "tables_tok/t.json",
            json.dumps(
                {"header": [["Name", []]], "data": [[["Alpha", ["/wiki/Alpha"]]]]}
            ),
        )
        z.writestr(
            prefix + "request_tok/t.json",
            json.dumps({"/wiki/Alpha": "Original passage"}),
        )
    bundle = {
        "table_id": "t",
        "sql": {
            "title": "",
            "rows": [
                {
                    "row_index": 0,
                    "text": "Name: Alpha",
                    "cells": [
                        {
                            "column_index": 0,
                            "header": "Name",
                            "value": "Alpha",
                            "entity_ids": ["/wiki/Alpha"],
                        }
                    ],
                }
            ],
        },
        "passages": {
            "missing_ids": [],
            "records": {"/wiki/Alpha": {"text": "Corrupted passage"}},
        },
    }
    audit = SourceAudit(path)
    with pytest.raises(ValueError, match="text mismatch"):
        audit.check(bundle)
    bundle["passages"]["records"]["/wiki/Alpha"] = {
        "title": "Alpha",
        "text": "Original passage",
    }
    assert audit.check(bundle)["passed"]
    bundle["sql"]["rows"][0]["text"] += " altered"
    with pytest.raises(ValueError, match="rendered row text mismatch"):
        audit.check(bundle)
    audit.close()


def test_reader_never_receives_labels_and_validates_citations(tmp_path, monkeypatch):
    from baseline import reader

    class Encoder:
        def encode(self, s):
            return list(s)

    seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(
                {
                    "status": "completed",
                    "model": "gpt-6-luna",
                    "id": "test",
                    "usage": {},
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json.dumps(
                                        {"answer": "Alpha", "citations": ["invented"]}
                                    ),
                                }
                            ],
                        }
                    ],
                }
            ).encode()

    def send(request, timeout):
        seen.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(reader.urllib.request, "urlopen", send)
    r = Reader("not-a-real-key", "gpt-6-luna", Encoder(), tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError, match="citation_outside_context"):
        r.generate(
            "qid", "all_stores", "question and source evidence", [{"id": "known"}]
        )
    assert seen[0]["store"] is False
    assert set(seen[0]) == {
        "model",
        "instructions",
        "input",
        "store",
        "reasoning",
        "max_output_tokens",
        "text",
    }
    assert "not-a-real-key" not in (tmp_path / "ledger.jsonl").read_text()
