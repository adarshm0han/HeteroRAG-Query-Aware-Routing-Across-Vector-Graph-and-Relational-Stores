import pytest

from routing.fusion import fuse


def test_rrf_deduplicates_each_source_and_respects_evidence_type_limits():
    records = {
        "a": {"id": "a", "kind": "row"},
        "b": {"id": "b", "kind": "row"},
        "p": {"id": "p", "kind": "passage"},
    }
    result = fuse(
        {"postgres": ["a", "a", "b"], "neo4j": ["b", "a"], "pinecone": ["p"]},
        records,
        row_k=1,
        passage_k=1,
    )
    assert [item["id"] for item in result] == ["a", "p"]
    assert result[0]["ranked_by"] == ["postgres", "neo4j"]


def test_concat_preserves_source_order_and_deduplicates():
    records = {key: {"id": key, "kind": "row"} for key in "abc"}
    result = fuse({"postgres": ["b", "a"], "neo4j": ["a", "c"]}, records, mode="concat")
    assert [item["id"] for item in result] == ["b", "a", "c"]


def test_weighted_rrf_changes_order_without_changing_candidate_set():
    records = {key: {"id": key, "kind": "row"} for key in "ab"}
    rankings = {"postgres": ["a", "b"], "neo4j": ["b", "a"]}
    unweighted = fuse(rankings, records)
    weighted = fuse(
        rankings, records, mode="weighted_rrf", weights={"postgres": 0.1, "neo4j": 0.9}
    )
    assert [item["id"] for item in unweighted] == ["a", "b"]
    assert [item["id"] for item in weighted] == ["b", "a"]
    with pytest.raises(ValueError, match="positive weights"):
        fuse(rankings, records, mode="weighted_rrf", weights={"postgres": 1.0})
