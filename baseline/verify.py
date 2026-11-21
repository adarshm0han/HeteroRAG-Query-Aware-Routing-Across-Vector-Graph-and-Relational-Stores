"""Independently audit saved local runs against the pinned public source and evaluator."""

import argparse
import ast
import collections
import json
import re
import string
import zipfile
from pathlib import Path

from .common import CORPUS_REVISION, REPO, read_json, sha256, write_json


def official_metrics(archive):
    name = next(n for n in archive.namelist() if n.endswith("/evaluate_script.py"))
    tree = ast.parse(archive.read(name).decode())
    wanted = {"normalize_answer", "get_tokens", "compute_exact", "compute_f1"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    scope = {"collections": collections, "re": re, "string": string}
    exec(  # noqa: S102 - Only selected functions from the checksum-verified archive.
        compile(
            ast.Module(body=functions, type_ignores=[]), "pinned_evaluator", "exec"
        ),
        scope,
    )
    return scope["compute_exact"], scope["compute_f1"]


def check_context(question, context, table, passages):
    """Check every content-bearing field, including rendered row text and graph links."""
    prefix, serialized = context["context"].split("\nEvidence:\n")
    assert json.loads(prefix) == {
        "question": question["question"],
        "table_title": table["title"],
    }
    assert json.loads(serialized) == context["evidence"]
    allowed_passages = {
        pid for row in table["data"] for _, links in row for pid in links
    }
    allowed_passages.update(pid for _, links in table["header"] for pid in links)
    for item in context["evidence"]:
        if item["kind"] == "row":
            assert item["table_id"] == question["table_id"]
            row = table["data"][item["row_index"]]
            assert item["text"] == " | ".join(
                f"{table['header'][i][0]}: {cell[0]}" for i, cell in enumerate(row)
            )
            expected = {
                (i, table["header"][i][0], cell[0], pid)
                for i, cell in enumerate(row)
                for pid in cell[1]
            }
            assert {
                (
                    link["column_index"],
                    link["column"],
                    link["cell_text"],
                    link["passage_id"],
                )
                for link in item["links"]
            } == expected
            assert all(
                (link["column"], link["cell_text"], link["passage_id"])
                in {v[1:] for v in expected}
                for link in item.get("graph_links", [])
            )
            assert (
                item["source_id"] == f"row:{question['table_id']}:{item['row_index']}"
            )
        else:
            assert item["kind"] == "passage"
            pid = item["passage_id"]
            assert pid in allowed_passages
            assert item["text"].strip() == passages[pid].strip()
            assert item["title"] == pid.removeprefix("/wiki/").replace("_", " ")
            assert item["source_id"] == pid


def verify_run(run, workspace):
    manifest = read_json(run / "manifest.json")
    summary = read_json(run / "summary.json")
    questions = read_json(run / "questions.json")
    references = read_json(run / "references.json")
    assert manifest["config"]["reader"] == "local_ollama"
    assert summary["complete"] and summary["api_requests"] == 0
    assert all(count == 0 for count in summary["physical_retrieval_calls"].values())
    for name, digest in manifest["config"]["code"].items():
        assert sha256(REPO / "baseline" / name) == digest, (
            f"Code changed after run: {name}"
        )
    lock = read_json(REPO / "postgres/sources.lock.json")
    for name in ("questions", "corpus"):
        assert (
            sha256(workspace / "data" / (name + ".zip")) == lock[name]["archive_sha256"]
        )
    scores = collections.defaultdict(list)
    valid, invalid, citations = 0, 0, 0
    with (
        zipfile.ZipFile(workspace / "data/questions.zip") as qarchive,
        zipfile.ZipFile(workspace / "data/corpus.zip") as corpus,
    ):
        exact, f1 = official_metrics(qarchive)
        names = qarchive.namelist()
        originals = {
            q["question_id"]: q
            for q in json.loads(
                qarchive.read(next(n for n in names if n.endswith("/train.json")))
            )
        }
        traces = {
            q["question_id"]: q
            for q in json.loads(
                qarchive.read(
                    next(n for n in names if n.endswith("/train.traced.json"))
                )
            )
        }
        for question in questions:
            qid, tid = question["question_id"], question["table_id"]
            assert set(question) == {"question_id", "question", "table_id"}
            assert question == {key: originals[qid][key] for key in question}
            assert references[qid]["answer"] == originals[qid]["answer-text"]
            assert references[qid]["trace_candidates"] == traces[qid].get(
                "answer-node", []
            )
            prefix = f"WikiTables-WithLinks-{CORPUS_REVISION}/"
            table = json.loads(corpus.read(prefix + "tables_tok/" + tid + ".json"))
            passages = json.loads(corpus.read(prefix + "request_tok/" + tid + ".json"))
            retrieval = read_json(run / "retrieval" / (qid + ".json"))
            assert (
                retrieval["question"] == question
                and retrieval["source_audit"]["passed"]
            )
            for plan in manifest["config"]["plans"]:
                context = retrieval["contexts"][plan]
                check_context(question, context, table, passages)
                assert context["tokens"] <= manifest["config"]["context_tokens"]
                answer = read_json(run / "answers" / f"{qid}.{plan}.json")
                assert answer["question_id"] == qid and answer["plan"] == plan
                assert answer["local_inference"] and answer["api_cost_usd"] == 0
                assert answer["model_digest"] == manifest["config"]["model_digest"]
                if answer["ok"]:
                    assert set(answer["citations"]) <= {
                        e["id"] for e in context["evidence"]
                    }
                    assert (
                        answer["abstain"]
                        == (not answer["answer"].strip())
                        == (not answer["citations"])
                    )
                    scores[plan].append(
                        (
                            exact(references[qid]["answer"], answer["answer"]),
                            f1(references[qid]["answer"], answer["answer"]),
                        )
                    )
                    valid += 1
                    citations += len(answer["citations"])
                else:
                    scores[plan].append((0.0, 0.0))
                    invalid += 1
    for plan, values in scores.items():
        for index, key in enumerate(("em_percent", "f1_percent")):
            expected = 100 * sum(v[index] for v in values) / len(values)
            assert abs(expected - summary["results"][plan][key]) < 1e-9
    result = {
        "passed": True,
        "questions": len(questions),
        "checked_contexts": valid + invalid,
        "valid_answer_contracts": valid,
        "invalid_answer_contracts": invalid,
        "validated_citation_ids": citations,
        "official_metric_parity": True,
        "public_source_context_parity": True,
        "code_hashes_match": True,
        "paid_api_calls": 0,
        "new_database_calls": 0,
    }
    write_json(run / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--workspace", type=Path, default=REPO.parent)
    args = parser.parse_args()
    for run in args.runs:
        print(
            json.dumps(
                {"run": str(run), **verify_run(run.resolve(), args.workspace.resolve())}
            )
        )


if __name__ == "__main__":
    main()
