"""Small synthetic store responses for offline contract tests, never benchmark data."""

import copy

from baseline.answers import validate_answer

QUESTION = {
    "question_id": "synthetic-1",
    "table_id": "synthetic-table",
    "question": "What is the nickname of the player with the most points?",
}


class FakeTokenizer:
    def encode(self, text):
        return list(text.encode("utf-8"))


def vector(first, second=0.0):
    return [first, second] + [0.0] * 382


ROWS = [
    {
        "table_id": "synthetic-table",
        "row_index": index,
        "text": f"Player: {name} | Points: {points}",
        "links": [
            {"column": "Player", "cell_text": name, "passage_id": f"/wiki/{name}"}
        ],
        "passage_ids": [f"/wiki/{name}"],
    }
    for index, name, points in ((0, "Ada", 7), (1, "Bea", 3))
]
PASSAGES = {
    "/wiki/Ada": {
        "title": "Ada",
        "text": "Ada is a player nicknamed Rocket.",
        "values": vector(1.0),
    },
    "/wiki/Bea": {
        "title": "Bea",
        "text": "Bea is a player nicknamed Comet.",
        "values": vector(0.0, 1.0),
    },
}


class FakeEmbedding:
    def __init__(self):
        self.calls = 0

    def encode(self, question):
        self.calls += 1
        return vector(1.0), 0.001


class FakeStores:
    def __init__(self, allowed, graph_covered=True, fail=None, empty=False):
        self.allowed = frozenset(allowed)
        self.graph_covered = graph_covered
        self.fail = fail
        self.empty = empty
        self.events = []
        self.closed = False

    def record(self, store, operation):
        assert store in self.allowed, "Unselected store used"
        event = {
            "store": store,
            "operation": operation,
            "kind": "data",
            "status": "completed",
            "seconds": 0.001,
        }
        self.events.append(event)
        if self.fail == store:
            event["status"] = "failed"
            raise TimeoutError("Synthetic database timeout")

    def sql_table(self, table_id):
        self.record("postgres", "select_table")
        return {"rows": [] if self.empty else copy.deepcopy(ROWS), "title": "Scores"}

    def sql_retrieve(self, table_id, question, limit):
        return self.sql_table(table_id)

    def graph_table(self, table_id, anchors=(), **kwargs):
        self.record("neo4j", "select_rows_and_links")
        rows = copy.deepcopy(ROWS) if self.graph_covered and not self.empty else []
        for row in rows:
            row["anchor_hits"] = len(set(row["passage_ids"]) & set(anchors))
        return {"rows": rows, "title": "Scores", "covered": bool(rows)}

    def search(self, table_id, query_vector, k):
        self.record("pinecone", "query")
        return {} if self.empty else {"/wiki/Ada": copy.deepcopy(PASSAGES["/wiki/Ada"])}

    def passages(self, ids):
        assert "pinecone" in self.allowed
        if ids:
            self.record("pinecone", "fetch")
        return {
            "records": {
                pid: copy.deepcopy(PASSAGES[pid]) for pid in ids if pid in PASSAGES
            },
            "missing_ids": sorted(set(ids) - set(PASSAGES)),
        }

    def close(self):
        self.closed = True


class FakeReader:
    def __init__(self):
        self.calls = 0

    def generate(self, context, evidence):
        self.calls += 1
        supporting = [
            item
            for item in evidence
            if "Rocket" in item["text"] or "Points: 7" in item["text"]
        ]
        has_rows = any(item["kind"] == "row" for item in supporting)
        has_passages = any(item["kind"] == "passage" for item in supporting)
        parsed = {
            "answer": "Rocket" if has_rows and has_passages else "",
            "citations": [item["id"] for item in supporting]
            if has_rows and has_passages
            else [],
        }
        return {
            **validate_answer(parsed, evidence),
            "seconds": 0.001,
            "api_cost_usd": 0,
            "local_inference": True,
        }


class FakeDecisions:
    def __init__(self, option):
        self.option = option
        self.payloads = []
        self.events = []

    def create(self, payload):
        self.payloads.append(payload)
        self.events.append(
            {"endpoint": "decisions", "status": "simulated", "seconds": 0.001}
        )
        choices = payload["questions"][0]["choices"]
        response = {
            "model": "gpt-6-luna",
            "usage": {"input_tokens": 100, "output_tokens": 0},
            "answers": [
                {
                    "type": "choice",
                    "name": "retrieval_option",
                    "choice": self.option,
                    "probabilities": [
                        {
                            "value": c["value"],
                            "probability": float(c["value"] == self.option),
                        }
                        for c in choices
                    ],
                    "confidence": 1.0,
                }
            ],
        }
        for question in payload["questions"][1:]:
            response["answers"].append(
                {
                    "type": "score",
                    "name": question["name"],
                    "score": 1.5,
                }
            )
        return response
