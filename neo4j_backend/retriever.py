"""Small graph retriever returning HeteroRAG-compatible evidence dictionaries."""

from __future__ import annotations

import re
from typing import Any

from neo4j import GraphDatabase

from neo4j_backend.config import Neo4jConfig


def tokenize(query: str) -> list[str]:
    # Keep informative tokens and avoid common English stop words.
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "what", "which",
        "who", "when", "where", "why", "how", "of", "to", "in", "on",
        "for", "and", "or", "with", "from", "by", "does", "do", "did",
        "that", "this", "it", "as", "at", "be", "than", "then",
    }
    tokens = [t.casefold() for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", query)]
    return list(dict.fromkeys(t for t in tokens if len(t) >= 2 and t not in stop))[:12]


class Neo4jRetriever:
    """Graph-based retrieval over table rows and shared cell entities.

    It finds rows whose serialized header/value text matches query terms, then
    expands one hop through shared cell entities. It is deliberately independent
    of an LLM so it can be benchmarked before the router/generator are ready.
    """

    def __init__(self, uri: str | None = None, user: str | None = None,
                 password: str | None = None, database: str | None = None):
        cfg = Neo4jConfig()
        self.database = database or cfg.database
        self.driver = GraphDatabase.driver(
            uri or cfg.uri,
            auth=(user or cfg.user, password if password is not None else cfg.password),
        )
        self.driver.verify_connectivity()

    def close(self) -> None:
        self.driver.close()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return ranked evidence in a common format used by other backends."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        terms = tokenize(query)
        if not terms:
            return []

        cypher = """
        // Find seed rows by lexical overlap with the complete row text, including headers.
        UNWIND $terms AS term
        MATCH (seed:TableRow)
        WHERE toLower(seed.text) CONTAINS term
        WITH seed, collect(DISTINCT term) AS seed_terms

        // Expand one hop through shared cell entities to discover related rows.
        OPTIONAL MATCH (seed)-[:HAS_ENTITY]->(:Entity)<-[:HAS_ENTITY]-(related:TableRow)
        WITH seed, seed_terms,
             [r IN collect(DISTINCT related) WHERE r IS NOT NULL][0..10] AS related_rows
        WITH seed, seed_terms, [seed] + related_rows AS candidates
        UNWIND candidates AS r
        MATCH (t:Table)-[:HAS_ROW]->(r)
        WITH r, t, seed, seed_terms,
             CASE WHEN r.row_id = seed.row_id THEN 1.0 ELSE 0.5 END AS graph_factor
        OPTIONAL MATCH (r)-[:HAS_ENTITY]->(e:Entity)
        WITH r, t, max(size(seed_terms) * graph_factor) AS overlap,
             collect(DISTINCT e.value) AS matched_entities,
             collect(DISTINCT seed.row_id)[0..5] AS seed_row_ids
        RETURN r.row_id AS row_id,
               r.row_index AS row_index,
               r.text AS text,
               t.table_id AS table_id,
               t.title AS table_title,
               matched_entities,
               seed_row_ids,
               overlap
        ORDER BY overlap DESC, table_title ASC, row_index ASC
        LIMIT $top_k
        """
        with self.driver.session(database=self.database) as session:
            rows = session.run(cypher, terms=terms, top_k=top_k).data()

        results = []
        for rank, row in enumerate(rows, start=1):
            # The score is a simple lexical-overlap score, not a calibrated probability.
            score = float(row["overlap"] / max(len(terms), 1))
            text = row.get("text") or ""
            title = row.get("table_title") or ""
            results.append({
                "id": f"neo4j:{row['row_id']}",
                "source": "neo4j",
                "text": f"Table: {title}\nRow: {text}",
                "score": score,
                "rank": rank,
                "metadata": {
                    "table_id": row["table_id"],
                    "table_title": title,
                    "row_id": row["row_id"],
                    "row_index": row["row_index"],
                    "matched_entities": row["matched_entities"],
                    "seed_row_ids": row["seed_row_ids"],
                    "matched_terms": row["overlap"],
                },
            })
        return results


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Query the HeteroRAG Neo4j backend")
    parser.add_argument("query", help="Natural-language query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    retriever = Neo4jRetriever()
    try:
        print(json.dumps(retriever.retrieve(args.query, args.top_k), indent=2))
    finally:
        retriever.close()
