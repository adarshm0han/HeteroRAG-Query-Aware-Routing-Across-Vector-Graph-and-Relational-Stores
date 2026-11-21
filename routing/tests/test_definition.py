"""Portable reconstruction from pinned archives without machine-local run history."""

import json
import zipfile

import pytest

from baseline.common import read_json, sha256
from evaluation import cohorts, protocol
from evaluation.checkpoints import atomic_json
from routing import catalog


@pytest.fixture
def definition(tmp_path, monkeypatch):
    root, workspace = tmp_path / "repo", tmp_path / "workspace"
    folder = root / "experiments/example"
    folder.mkdir(parents=True)
    (workspace / "data").mkdir(parents=True)
    rows = {
        split: [
            {
                "question_id": f"{split}-{i}",
                "question": "What is the value?",
                "table_id": "table",
                "answer-text": f"value-{i}",
                "answer-node": [],
            }
            for i in range(3)
        ]
        for split in ("train", "dev")
    }
    with zipfile.ZipFile(workspace / "data/questions.zip", "w") as archive:
        for split, records in rows.items():
            archive.writestr(
                f"source/released_data/{split}.traced.json", json.dumps(records)
            )
    (workspace / "data/corpus.zip").write_bytes(b"synthetic pinned corpus")
    lock = {
        name: {"archive_sha256": sha256(workspace / f"data/{name}.zip")}
        for name in ("questions", "corpus")
    }
    atomic_json(root / "postgres/sources.lock.json", lock)
    ids = {
        "development": ["train-0", "train-1"],
        "smoke": ["train-0"],
        "validation": ["train-2"],
        "evaluation": ["dev-0", "dev-1"],
        "timing": ["dev-1"],
    }
    source = {q["question_id"]: q for records in rows.values() for q in records}
    phases = protocol.DEFAULT_PHASES
    jobs = [
        j
        for phase in phases
        for j in protocol.schedule(
            [
                source[qid]
                for qid in ids["timing" if phase.startswith("timing-") else phase]
            ],
            phase,
        )
    ]
    expected, formats = {}, {}
    golden = tmp_path / "golden"
    serialization = {"indent": 2, "newline": "\r\n", "trailing_newline": False}
    for name, qids in ids.items():
        values = {
            "questions.json": [
                {k: source[qid][k] for k in ("question_id", "question", "table_id")}
                for qid in qids
            ],
            "references.json": {
                qid: {"answer": source[qid]["answer-text"], "weak_evidence_ids": []}
                for qid in qids
            },
        }
        for filename, value in values.items():
            relative = f"{name}/{filename}"
            protocol.write_input(golden / relative, value, serialization)
            expected[relative], formats[relative] = (
                sha256(golden / relative),
                serialization,
            )
    protocol.write_input(folder / "graph-table-ids.json", ["table"], serialization)
    expected["graph-table-ids.json"] = sha256(folder / "graph-table-ids.json")
    protocol.write_input(
        golden / "routing-catalog.json", {"synthetic": True}, serialization
    )
    expected["routing-catalog.json"], formats["routing-catalog.json"] = (
        sha256(golden / "routing-catalog.json"),
        serialization,
    )
    config = {
        "experiment": "example",
        "cohort_ids": ids,
        "expected_inputs": expected,
        "input_serialization": formats,
        "protocol": {
            "study": "hybridqa",
            "profile": "full",
            "sources": lock,
            "conditions": protocol.CONDITIONS,
            "phases": phases,
            "outcomes": len(jobs),
            "paid_request_cap": sum(j["paid"] for j in jobs),
            "wall_time_budget_seconds": 75600,
            "cohorts": {
                name: {"questions": len(qids), "tables": 1, "graph_covered": len(qids)}
                for name, qids in ids.items()
            },
        },
    }
    atomic_json(folder / "config.json", config)
    monkeypatch.setattr(protocol, "REPO", root)
    monkeypatch.setattr(protocol, "code_hashes", dict)
    monkeypatch.setattr(cohorts, "code_hashes", dict)

    def fake_catalog(workspace, output, coverage):
        atomic_json(output, {"synthetic": True})
        return {"synthetic": True}

    monkeypatch.setattr(catalog, "prepare", fake_catalog)
    return folder, workspace, config, golden


def test_definition_rebuild_preserves_reference_bytes_and_custom_runtime(
    definition, tmp_path
):
    folder, workspace, config, golden = definition
    output = tmp_path / "bundle"
    result = protocol.prepare_definition(
        folder, workspace, output, max_runtime_hours=2.5
    )
    assert result["wall_time_budget_seconds"] == 9000
    assert protocol.verify(output, check_code=True)[0] == result
    for relative, digest in config["expected_inputs"].items():
        assert sha256(output / relative) == digest
    assert (output / "evaluation/references.json").read_bytes() == (
        golden / "evaluation/references.json"
    ).read_bytes()
    assert all(
        "answer-text" not in q for q in read_json(output / "evaluation/questions.json")
    )
    with pytest.raises(ValueError, match="never overwrite"):
        protocol.prepare_definition(folder, workspace, output)


def test_definition_rejects_cross_split_ids_before_reconstructing(definition, tmp_path):
    folder, workspace, config, _ = definition
    config["cohort_ids"]["validation"] = ["dev-2"]
    atomic_json(folder / "config.json", config)
    with pytest.raises(ValueError, match="intended source split"):
        protocol.prepare_definition(folder, workspace, tmp_path / "invalid-bundle")


@pytest.mark.parametrize("hours", [0, -1, float("inf"), float("nan")])
def test_runtime_limit_must_be_finite_and_positive(hours):
    with pytest.raises(ValueError, match="finite positive"):
        protocol.runtime_seconds(hours, 75600)
