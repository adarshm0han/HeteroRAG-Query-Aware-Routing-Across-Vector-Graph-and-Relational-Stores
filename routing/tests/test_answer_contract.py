import json

import pytest
from urllib3.exceptions import ProtocolError

from routing.answer_context import frame_context
from routing.evidence_answer import validate_response
from routing.failures import failure_details
from routing.reader import CheckedReader

from .test_improvements import fake_reader


def checked(answer, question, evidence, ids=None):
    _, _, _, excerpts = frame_context(question, evidence)
    return validate_response(
        {
            "answer": answer,
            "reasoning": "Source evidence",
            "calculation": "",
            "excerpts": ids or list(excerpts),
        },
        excerpts,
        question,
    )


@pytest.mark.parametrize(
    "text",
    [
        "Value: 6.1",
        "Value: 19",
        "Value: -1",
        "Value: - 1",
        "Value: minus 1",
        "Value: \u22121",
    ],
)
def test_numeric_answer_cannot_be_a_decimal_signed_or_integer_fragment(text):
    with pytest.raises(ValueError, match="not_in_cited_sources"):
        checked("1", "What value?", [{"id": "E1", "text": text}])


@pytest.mark.parametrize(
    "answer,text", [("1000", "1,000"), ("1", "1.0"), ("-1", "- 1"), ("0.1", "0.10")]
)
def test_equivalent_whole_numeric_atoms_are_supported(answer, text):
    assert (
        checked(answer, "What value?", [{"id": "E1", "text": text}])["answer"] == answer
    )


def test_cited_row_determines_explicit_field_even_when_another_row_matches():
    evidence = [
        {"id": "E1", "kind": "row", "text": "Name: Alder | Rank: 19 | Percentage: 6.1"},
        {"id": "E2", "kind": "row", "text": "Name: Birch | Rank: 1"},
    ]
    with pytest.raises(ValueError):
        checked("1", "What is the rank of Alder?", evidence, ["E1"])
    assert (
        checked("19", "What is the rank of Alder?", evidence, ["E1"])["answer"] == "19"
    )
    with pytest.raises(ValueError, match="other_field"):
        checked("Alder", "What is the rank?", evidence, ["E1"])


def test_final_column_guard_does_not_bind_intermediate_or_ambiguous_names():
    evidence = [
        {
            "id": "E1",
            "kind": "row",
            "text": "Name: New Band | English name: The New Band",
        },
        {
            "id": "E2",
            "kind": "passage",
            "text": "New Band was founded as The Originals.",
        },
    ]
    assert (
        checked("The Originals", "What name was the group founded as?", evidence)[
            "answer"
        ]
        == "The Originals"
    )
    with pytest.raises(ValueError, match="other_field"):
        checked(
            "New Band",
            "What English name is listed?",
            [
                {
                    "id": "E1",
                    "kind": "row",
                    "text": "Name: New Band | English name: Originals",
                }
            ],
        )


def test_quantity_and_time_guards_reject_explicit_wrong_types():
    with pytest.raises(ValueError, match="quantity"):
        checked(
            "Rae", "How many members?", [{"id": "E1", "text": "Members: Rae and Kay"}]
        )
    for text, kind, answer in [
        ("Date: 2017 | Location: Oakville", "row", "Oakville"),
        (
            "The first event was held on 3 March 1991 in Oakville.",
            "passage",
            "Oakville",
        ),
    ]:
        with pytest.raises(ValueError, match="time_not_location"):
            checked(
                answer,
                "The event first happened when?",
                [{"id": "E1", "kind": kind, "text": text}],
            )
    for answer in ("the coronation", "Independence", "March", "1991"):
        assert (
            checked(
                answer,
                "When did it happen?",
                [{"id": "E1", "text": "It happened during " + answer}],
            )["answer"]
            == answer
        )


def test_release_year_is_not_a_work_title_but_numeric_titles_remain_valid():
    with pytest.raises(ValueError, match="work_title"):
        checked(
            "2012",
            "What was the most recent movie by the director?",
            [
                {
                    "id": "E1",
                    "kind": "passage",
                    "text": "Her most recent film, Old Times, was released in 2012.",
                }
            ],
        )
    for text in (
        "1984 is a film released in 1984.",
        "Title: 1984 | Year: 1984",
        "The film titled 1984 was released in 1984.",
    ):
        assert (
            checked("1984", "Which movie was released?", [{"id": "E1", "text": text}])[
                "answer"
            ]
            == "1984"
        )


def test_pronoun_support_adds_actual_name_quote_only_from_same_passage():
    evidence = [
        {
            "id": "E1",
            "kind": "passage",
            "title": "Alder",
            "text": "Alder is a tree. It is the provincial emblem.",
        }
    ]
    result = checked("Alder", "Which tree is the emblem?", evidence, ["E1.S2"])
    assert [q["excerpt_id"] for q in result["support_quotes"]] == ["E1.S2", "E1.S1"]
    assert result["support_quotes"][-1]["quote"] == "Alder is a tree."
    evidence[0]["text"] = "Birch is a tree. It is the provincial emblem."
    evidence.append(
        {"id": "E2", "kind": "passage", "title": "Alder", "text": "Alder is a tree."}
    )
    with pytest.raises(ValueError, match="not_in_cited_sources"):
        checked("Alder", "Which tree is the emblem?", evidence, ["E1.S2"])


def test_invalid_final_type_is_repaired_by_bounded_reader_review():
    bad = {"reasoning": "Name", "answer": "Rae", "excerpts": ["E1"], "calculation": ""}
    good = {**bad, "answer": "2"}
    base, calls = fake_reader([bad, bad, good])
    result = CheckedReader(base).generate(
        "How many members?", [{"id": "E1", "text": "There are 2 members: Rae and Kay."}]
    )
    assert result["answer"] == "2" and len(calls) == 3
    assert "question_requires_quantity_not_entity" in calls[2]["messages"][1]["content"]


def test_protocol_disconnect_is_infrastructure_without_logging_sensitive_message():
    details = failure_details(
        ProtocolError("SECRET URL", ConnectionResetError("SECRET"))
    )
    assert details["infrastructure"] is True
    assert "SECRET" not in json.dumps(details)
