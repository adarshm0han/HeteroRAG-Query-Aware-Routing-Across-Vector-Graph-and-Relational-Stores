import json
import zipfile

import pytest

from baseline.common import CORPUS_REVISION
from routing.audit import EvidenceAudit
from routing.report import summarize

from .fixtures import QUESTION
from .test_pipeline import pipeline


def source_archive(tmp_path):
    path = tmp_path / "corpus.zip"
    prefix = f"WikiTables-WithLinks-{CORPUS_REVISION}/"
    table = {
        "header": [["Player", []], ["Points", []]],
        "data": [
            [["Ada", ["/wiki/Ada"]], ["7", []]],
            [["Bea", ["/wiki/Bea"]], ["3", []]],
        ],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(prefix + "tables_tok/synthetic-table.json", json.dumps(table))
        archive.writestr(
            prefix + "request_tok/synthetic-table.json",
            json.dumps(
                {
                    "/wiki/Ada": "Ada is a player nicknamed Rocket.",
                    "/wiki/Bea": "Bea is a player nicknamed Comet.",
                }
            ),
        )
    return path


def test_source_audit_checks_retrieved_text_and_rejects_corruption(tmp_path):
    app, _, _ = pipeline("all")
    result = app.run(QUESTION)
    audit = EvidenceAudit(source_archive(tmp_path))
    try:
        items = result["retrieval"]["items"]
        assert audit.check(QUESTION["table_id"], items) == {
            "passed": True,
            "rows": 2,
            "passages": 2,
        }
        passage = next(item for item in items if item["kind"] == "passage")
        passage["text"] = "Altered text"
        with pytest.raises(ValueError, match="Passage text"):
            audit.check(QUESTION["table_id"], items)
    finally:
        audit.close()


def test_failed_source_audit_prevents_answer_generation():
    class Auditor:
        def check(self, *args):
            raise ValueError("Synthetic source mismatch")

    app, _, _ = pipeline("all")
    app.auditor = Auditor()
    result = app.run(QUESTION)
    assert result["failed_stage"] == "source_audit"
    assert app.reader.calls == 0


def test_summary_counts_execution_failures_in_answer_scores():
    app, _, _ = pipeline("all")
    completed = app.run(QUESTION)
    failed = {
        "question": {**QUESTION, "question_id": "failed"},
        "status": "failed",
        "database_calls": [],
        "decisions_calls": [],
        "total_seconds": 1.0,
    }
    references = {
        qid: {"answer": "Rocket"} for qid in (QUESTION["question_id"], "failed")
    }
    summary = summarize([completed, failed], references)
    assert summary["scored_questions"] == 2
    assert summary["em"] == 50
    assert summary["f1"] == 50
    assert summary["statuses"] == {"completed": 1, "failed": 1}
