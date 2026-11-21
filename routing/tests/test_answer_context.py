from routing.answer_context import frame_context


def test_framing_keeps_only_links_to_evidence_in_the_context():
    evidence = [
        {
            "id": "E1",
            "kind": "row",
            "text": "Name: Example",
            "links": [{"passage_id": "/wiki/Example"}, {"passage_id": "/wiki/Absent"}],
        },
        {
            "id": "E2",
            "kind": "passage",
            "passage_id": "/wiki/Example",
            "text": "Example has the required attribute.",
        },
    ]
    *_, excerpts = frame_context("Which name?", evidence)
    assert excerpts["E1"]["linked_evidence"] == ["E2"]
