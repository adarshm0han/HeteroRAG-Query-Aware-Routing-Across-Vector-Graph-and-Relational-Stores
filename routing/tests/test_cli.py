"""Exercise the complete command, output artifacts and saved-route reuse offline."""

import pytest

from baseline.common import read_json, write_json
from routing import cli

from .fixtures import (
    QUESTION,
    FakeDecisions,
    FakeEmbedding,
    FakeReader,
    FakeStores,
    FakeTokenizer,
)


@pytest.mark.parametrize("first_failure", [False, "timeout", "protocol"])
def test_cli_runs_and_replays_without_paid_requests_or_label_leakage(
    tmp_path, monkeypatch, first_failure
):
    import baseline.local_reader
    import routing.decisions
    import routing.embedding
    import routing.reader
    import routing.stores
    import routing.tokenizer

    reader = FakeReader()
    reader.metadata = {"name": "qwen3.5:4b", "digest": "synthetic"}
    reader.runtime = {"version": "synthetic"}
    client = FakeDecisions("all")
    monkeypatch.setattr(baseline.local_reader, "LocalReader", lambda model: reader)
    monkeypatch.setattr(routing.reader, "CheckedReader", lambda base: base)
    monkeypatch.setattr(
        routing.decisions, "DecisionsClient", lambda *args, **kwargs: client
    )
    monkeypatch.setattr(
        routing.embedding, "LocalEmbedding", lambda cache: FakeEmbedding()
    )
    instances = []

    def stores(cfg, allowed, **kwargs):
        class ProtocolFailureStores(FakeStores):
            def record(self, store, operation, **extra):
                from urllib3.exceptions import ProtocolError

                if self.fail == store:
                    raise ProtocolError("Synthetic disconnect")
                return super().record(store, operation, **extra)

        store_type = (
            ProtocolFailureStores if first_failure == "protocol" else FakeStores
        )
        instance = store_type(
            allowed, fail="postgres" if first_failure and not instances else None
        )
        instances.append(instance)
        return instance

    monkeypatch.setattr(routing.stores, "LiveStores", stores)
    monkeypatch.setattr(
        routing.tokenizer, "local_tokenizer", lambda workspace: FakeTokenizer()
    )
    monkeypatch.setattr(cli, "database_settings", lambda workspace: {})
    monkeypatch.setattr(cli, "decisions_key", lambda workspace: "fake")
    questions, references = tmp_path / "questions.json", tmp_path / "references.json"
    public = (
        [QUESTION, {**QUESTION, "question_id": "synthetic-2"}]
        if first_failure
        else [QUESTION]
    )
    write_json(
        questions, [{**q, "answer-text": "DO NOT SEND THIS LABEL"} for q in public]
    )
    write_json(references, {q["question_id"]: {"answer": "Rocket"} for q in public})
    first, replay = tmp_path / "first", tmp_path / "replay"
    arguments = [
        "--questions",
        str(questions),
        "--references",
        str(references),
        "--output",
        str(first),
        "--allow-decisions-api",
        "--collect-fusion-weights",
        "--continue-on-error",
        "--max-decision-requests",
        "2",
    ]
    if first_failure:
        with pytest.raises(SystemExit, match="stopped after a failure"):
            cli.main(arguments)
        summary = read_json(first / "summary.json")
        assert not summary["run_complete"]
        assert len(client.payloads) == 1
        assert len(instances) == 1
        result = read_json(first / "results/00000.json")
        assert result["failure_details"]["infrastructure"]
        assert not (first / "results/00001.json").exists()
        return
    cli.main(arguments)
    assert read_json(first / "questions.json") == public
    summary = read_json(first / "summary.json")
    assert summary["run_complete"] is True
    assert summary["em"] == (50 if first_failure else 100)
    assert summary["all_calls_successful"] is not first_failure
    assert len(client.payloads) == len(public)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Replaying a decision must not load API credentials or create an API client"
        )

    monkeypatch.setattr(cli, "decisions_key", forbidden)
    monkeypatch.setattr(routing.decisions, "DecisionsClient", forbidden)
    cli.main(
        [
            "--questions",
            str(questions),
            "--output",
            str(replay),
            "--decision-records",
            str(first),
            "--fusion",
            "weighted_rrf",
        ]
    )
    summary = read_json(replay / "summary.json")
    assert summary["run_complete"] is True
    assert summary["replayed_decisions"] == len(public)
    assert summary["decisions_requests"] == 0
