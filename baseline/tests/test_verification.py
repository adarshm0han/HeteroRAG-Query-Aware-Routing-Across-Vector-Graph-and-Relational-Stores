import json

import pytest

from baseline.verify import check_context


@pytest.mark.parametrize("corrupt", [False, True])
def test_cached_context_is_checked_against_source_text(corrupt):
    question = {"question_id": "q", "question": "Who?", "table_id": "t"}
    table = {"title": "Example", "header": [["Name", []]], "data": [[["Alpha", []]]]}
    evidence = [
        {
            "id": "E1",
            "source_id": "row:t:0",
            "kind": "row",
            "table_id": "t",
            "row_index": 0,
            "text": "Name: Beta" if corrupt else "Name: Alpha",
            "links": [],
        }
    ]
    context = {
        "evidence": evidence,
        "context": json.dumps({"question": "Who?", "table_title": "Example"})
        + "\nEvidence:\n"
        + json.dumps(evidence),
    }
    if corrupt:
        with pytest.raises(AssertionError):
            check_context(question, context, table, {})
    else:
        check_context(question, context, table, {})
