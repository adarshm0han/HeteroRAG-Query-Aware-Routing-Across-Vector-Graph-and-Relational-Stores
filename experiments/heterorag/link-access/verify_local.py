"""Offline acceptance checks for link masking, retained graph access and raw replay."""

import importlib.util
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
RUNTIME = REPO / "artifacts/link-access-runtime"
sys.path.insert(0, str(RUNTIME))

from routing.link_access import LinkAccessRouter, LinkAccessStores, mask_postgres_links

from evaluation.execution import execute
from evaluation.protocol import verify
from routing.catalog import table_metadata
from routing.pipeline import Config
from routing.retrieval import retrieve
from routing.router import DecisionsRouter
from routing.stores import LiveStores

spec = importlib.util.spec_from_file_location(
    "fixtures", REPO / "routing/tests/fixtures.py"
)
fx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fx)
metadata = table_metadata(
    {"title": "Scores", "header": [["Player"], ["Points"]], "data": [[], []]}, True
)
table = {"title": "Scores", "rows": deepcopy(fx.ROWS)}
table["rows"][0]["cells"] = [
    {"header": "Player", "value": "Ada", "entity_ids": ["/wiki/Ada"]}
]
before = deepcopy(table)
masked = mask_postgres_links(table)
assert table == before
assert all(not r["links"] and not r["passage_ids"] for r in masked["rows"])
assert masked["rows"][0]["cells"][0] == {
    "header": "Player",
    "value": "Ada",
    "entity_ids": [],
}
assert [r["text"] for r in masked["rows"]] == [r["text"] for r in table["rows"]]
with patch.object(LiveStores, "sql_retrieve", return_value=table):
    assert (
        LinkAccessStores({}, ["postgres"], postgres_links=False).sql_retrieve(
            "x", "q", 8
        )
        == masked
    )
    assert (
        LinkAccessStores({}, ["postgres"], postgres_links=True).sql_retrieve(
            "x", "q", 8
        )
        == table
    )

clients = [fx.FakeDecisions("postgres_pinecone") for _ in range(3)]
DecisionsRouter(clients[0]).choose(fx.QUESTION, table_metadata=metadata)
LinkAccessRouter(clients[1], True).choose(fx.QUESTION, table_metadata=metadata)
LinkAccessRouter(clients[2], False).choose(fx.QUESTION, table_metadata=metadata)
assert clients[0].payloads == clients[1].payloads
instructions = clients[2].payloads[0]["questions"][0]["instructions"]
assert "PostgreSQL rows expose no hyperlink identifiers" in instructions
assert "follows direct row hyperlinks without graph access" not in instructions
assert "Original hyperlinks remain" not in instructions
assert clients[0].payloads[0]["input"] == clients[2].payloads[0]["input"]
assert (
    clients[0].payloads[0]["questions"][0]["choices"][0]
    == clients[2].payloads[0]["questions"][0]["choices"][0]
)


class Stores(fx.FakeStores):
    def __init__(self, allowed, postgres_links, journal=None):
        super().__init__(allowed)
        self.postgres_links = postgres_links

    def sql_retrieve(self, *args):
        result = super().sql_retrieve(*args)
        return result if self.postgres_links else mask_postgres_links(result)


on = retrieve(fx.QUESTION, Stores(["postgres", "pinecone"], True), fx.FakeEmbedding())
off = retrieve(fx.QUESTION, Stores(["postgres", "pinecone"], False), fx.FakeEmbedding())
graph = retrieve(
    fx.QUESTION, Stores(["postgres", "neo4j", "pinecone"], False), fx.FakeEmbedding()
)
assert on["passage_discovery"]["available"] == 2
assert off["passage_discovery"]["available"] == 0
assert graph["passage_discovery"]["available"] == 2
assert (
    "/wiki/Bea" in on["candidate_records"]
    and "/wiki/Bea" not in off["candidate_records"]
)
assert "/wiki/Bea" in graph["candidate_records"]
assert all(not r["links"] for r in off["items"] if r["kind"] == "row")
assert all(
    r["links"] and "neo4j" in r["sources"] for r in graph["items"] if r["kind"] == "row"
)

catalog = SimpleNamespace(describe=lambda tid: metadata.copy(), digest="synthetic")
auditor = SimpleNamespace(check=lambda tid, items: {"passed": True})
with tempfile.TemporaryDirectory(
    prefix="link-access-check-", dir=REPO.parent / "tmp"
) as temp:
    root = Path(temp)
    client = fx.FakeDecisions("postgres")
    shared = {
        "stores_factory": lambda allowed, journal: Stores(allowed, False, journal),
        "embedding": fx.FakeEmbedding(),
        "reader": fx.FakeReader(),
        "tokenizer": fx.FakeTokenizer(),
        "config": Config(),
        "auditor": auditor,
        "catalog": catalog,
    }
    guarded = execute(
        fx.QUESTION,
        {"id": "guarded", "mode": "live", "postgres_links": False},
        root / "guarded",
        router=LinkAccessRouter(client, False),
        **shared,
    )
    assert guarded["status"] == "completed", guarded.get("error")
    assert guarded["decision"]["raw_option"] == "postgres"
    assert guarded["decision"]["option"] == "postgres_pinecone"
    raw = execute(
        fx.QUESTION,
        {"id": "raw", "mode": "raw_route", "postgres_links": False},
        root / "raw",
        router=None,
        source=guarded,
        **shared,
    )
    assert raw["status"] == "completed", raw.get("error")
    assert raw["decision"]["option"] == "postgres"
    assert raw["decision"]["api_requests"] == 0 and len(client.payloads) == 1
    assert raw["answer"]["abstain"]
protocol, jobs = verify(
    REPO / "artifacts/bundles/hybridqa-link-access-v1", check_code=True
)
assert len(jobs) == 2440 and sum(j["paid"] for j in jobs) == 610
assert all(
    j["mode"] == "raw_route" for j in jobs if j["condition"].startswith("routed_raw")
)
assert all(
    j["mode"] == "live" for j in jobs if not j["condition"].startswith("routed_raw")
)
print(
    json.dumps(
        {
            "passed": True,
            "checks": [
                "nonmutating mask",
                "adapter boundary",
                "original on-arm payload",
                "corrected off-arm capability descriptions",
                "link expansion blocked",
                "graph expansion retained",
                "graph links retained in reader items",
                "guard/raw end-to-end replay",
                "frozen identity and cap",
            ],
            "bundle_id": protocol["bundle_id"],
        },
        indent=2,
    )
)
