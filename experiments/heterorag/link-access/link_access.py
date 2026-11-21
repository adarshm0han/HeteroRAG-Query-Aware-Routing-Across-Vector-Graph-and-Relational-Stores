"""Application-boundary PostgreSQL hyperlink masking and matching API descriptions."""

from copy import deepcopy

from routing.router import DecisionsRouter
from routing.stores import LiveStores


def mask_postgres_links(table):
    table = deepcopy(table)
    for row in table["rows"]:
        row["links"] = []
        row["passage_ids"] = []
        for cell in row.get("cells", []):
            cell["entity_ids"] = []
    table.setdefault("sql", {})["postgres_link_access"] = "hidden"
    return table


class LinkAccessStores(LiveStores):
    def __init__(self, *args, postgres_links=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.postgres_links = postgres_links

    def sql_retrieve(self, table_id, question, limit):
        table = super().sql_retrieve(table_id, question, limit)
        return table if self.postgres_links else mask_postgres_links(table)


class LinkAccessClient:
    def __init__(self, client, postgres_links):
        self.client, self.postgres_links = client, postgres_links

    def __getattr__(self, name):
        return getattr(self.client, name)

    def create(self, payload):
        if self.postgres_links:
            return self.client.create(payload)
        payload = deepcopy(payload)
        q = payload["questions"][0]
        replacements = {
            "If schema is missing/truncated or need is uncertain,\nprefer PostgreSQL+Pinecone to preserve both table and passage evidence.": "If schema is missing/truncated or need is uncertain, preserve both table and\npassage evidence. PostgreSQL+Pinecone can combine table rows with independently\nsearched passages, but only Neo4j exposes explicit row-to-passage links in this condition.",
            "PostgreSQL+Pinecone follows direct row hyperlinks without graph access. Add Neo4j\nfor necessary shared-passage connections between rows or multi-step hyperlink paths,\nnot merely because the question combines a row and its directly linked biography.": "PostgreSQL rows expose no hyperlink identifiers in this condition.\nPostgreSQL+Pinecone supports SQL operations and independent semantic passage search,\nbut cannot expand a table row to its linked passages. When explicit row-to-passage\nlinks or shared-passage paths are required, Neo4j supplies them if the table is covered.",
            "Original hyperlinks remain and\ncan fetch direct linked bodies only if Pinecone is selected.": "Original hyperlink identifiers are hidden from the PostgreSQL adapter.\nIts rows cannot supply passage identifiers for link expansion or passage-to-row ranking.",
            "A direct row link may suffice; graph traversal adds explicit shared-link connections.": "Neo4j is the only source of explicit row-to-passage links in this condition.\nPinecone semantic search remains available without Neo4j.",
        }
        for old, new in replacements.items():
            if q["instructions"].count(old) != 1:
                raise ValueError(
                    "Router capability patch no longer matches frozen instructions"
                )
            q["instructions"] = q["instructions"].replace(old, new)
        for choice in q["choices"]:
            if choice["value"] == "postgres_pinecone":
                choice["description"] = (
                    "Full-table SQL selection/aggregation plus independent semantic passage search; no PostgreSQL row-to-passage hyperlinks."
                )
        return self.client.create(payload)


class LinkAccessRouter(DecisionsRouter):
    def __init__(self, client, postgres_links=True):
        super().__init__(LinkAccessClient(client, postgres_links))
