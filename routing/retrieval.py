"""Execute exactly one selected subset. Source labels describe the text's origin."""

from time import perf_counter

from baseline.common import object_hash
from baseline.retrieval import link_set

from .fusion import fuse, interleave
from .passage_selection import VERSION as PASSAGE_POLICY
from .passage_selection import candidate_links, passage_order
from .row_selection import reserve_passages, row_order


def retrieve(
    question, stores, embedding, row_k=8, passage_k=12, fusion="rrf", weights=None
):
    started = perf_counter()
    selected = stores.allowed
    text, tid = question["question"], question["table_id"]
    records, rankings, tables = {}, {}, {}
    vector, passages, embedding_seconds = None, {}, 0.0
    anchors = []
    if "pinecone" in selected:
        vector, embedding_seconds = embedding.encode(text)
        passages = stores.search(tid, vector, passage_k)
        anchors = passage_order(text, vector, passages)[:passage_k]
    if "postgres" in selected:
        tables["postgres"] = stores.sql_retrieve(tid, text, row_k)
    if "neo4j" in selected:
        sql = tables.get("postgres", {})
        # A successful SQL operation has already selected/ranked the full table.
        seeds = (
            row_order(text, sql, anchors)[:row_k]
            if sql.get("selection_grounded", sql.get("ordered", False))
            else []
        )
        tables["neo4j"] = stores.graph_table(
            tid, anchors=anchors, question=text, seed_rows=seeds, limit=row_k
        )

    # The initial semantic search may miss a rare date or relation. Discover
    # linked passages from the complete returned row candidates BEFORE keeping
    # only row_k rows; then reverse-join the stronger passage anchors to rows.
    discovery = {"available": 0, "cap": 96, "truncated": False}
    discovery_missing = []
    initial_anchors = list(anchors)
    if "pinecone" in selected and tables:
        ids, discovery = candidate_links(text, tables, anchors)
        expanded = stores.passages(set(ids) - set(passages))
        passages.update(expanded["records"])
        discovery_missing = expanded["missing_ids"]
        anchors = passage_order(text, vector, passages)[:passage_k]

    row_choices, linked = {}, {}
    for store, table in tables.items():
        rows = table["rows"]
        order = row_order(text, table, anchors)
        row_choices[store] = order[:row_k]
        linked[store] = link_set(rows, row_choices[store])
        for row in rows:
            if row["row_index"] not in row_choices[store]:
                continue
            key = f"row:{tid}:{row['row_index']}"
            item = {
                "id": key,
                "kind": "row",
                "table_id": tid,
                "row_index": row["row_index"],
                "text": row["text"],
                "links": row["links"],
                "sources": [store],
            }
            if row.get("paths"):
                item["graph_paths"] = row["paths"]
            if key in records:
                if records[key]["text"] != item["text"]:
                    raise ValueError("Conflicting row text across selected stores")
                records[key]["sources"].append(store)
                if row.get("paths"):
                    records[key]["graph_paths"] = row["paths"]
            else:
                records[key] = item
        for item in table.get("aggregates", []):
            records[item["id"]] = item

    missing = []
    if "pinecone" in selected:
        ids = set().union(*linked.values()) if linked else set()
        fetched = stores.passages(ids - set(passages) - set(discovery_missing))
        passages.update(fetched["records"])
        missing = sorted(set(discovery_missing) | set(fetched["missing_ids"]))
        semantic_order = passage_order(text, vector, passages)
        for pid, passage in passages.items():
            records[pid] = {
                "id": pid,
                "kind": "passage",
                "passage_id": pid,
                "title": passage["title"],
                "text": passage["text"],
                "sources": ["pinecone"],
            }
        rankings["pinecone"] = semantic_order[:passage_k]
    else:
        semantic_order = []

    for store, table in tables.items():
        row_ids = [f"row:{tid}:{i}" for i in row_choices[store]]
        passage_ids = [pid for pid in semantic_order if pid in linked[store]][
            :passage_k
        ]
        rankings[store] = [
            item["id"] for item in table.get("aggregates", [])
        ] + interleave(row_ids, passage_ids)

    # Use a stable store order for both RRF tie-breaking and concatenation.
    rankings = {
        store: rankings[store]
        for store in ("postgres", "neo4j", "pinecone")
        if store in rankings
    }
    fusion_started = perf_counter()
    # Protect a balanced set of independent anchors and row-linked passages.
    # Repeated structural votes cannot consume the entire passage budget.
    selected_rows = []
    for store, table in tables.items():
        by_id = {row["row_index"]: row for row in table["rows"]}
        selected_rows.extend(by_id[i] for i in row_choices[store])
    protected = reserve_passages(
        text, anchors, semantic_order, selected_rows, passage_k
    )
    # Reserved linked passages may lie below a store's original top-k ranking.
    # Retain them at their real semantic rank for fusion, rather than inventing votes.
    if "pinecone" in rankings:
        rankings["pinecone"] = semantic_order
    items = fuse(
        rankings, records, row_k, passage_k, fusion, weights, protected=protected
    )
    fusion_seconds = perf_counter() - fusion_started
    return {
        "items": items,
        "candidate_records": records,
        "rankings": rankings,
        "semantic_anchors": anchors,
        "initial_search_anchors": initial_anchors,
        "passage_policy": PASSAGE_POLICY,
        "passage_ranking": "equal_lexical_semantic_rrf_k10",
        "passage_discovery": discovery,
        "candidate_passage_count": len(passages),
        "candidate_passage_text_bytes": sum(
            len(p["text"].encode("utf-8")) for p in passages.values()
        ),
        "protected_passages": protected,
        "row_selection": {
            store: {
                "selected_rows": row_choices[store],
                "candidate_rows": len(table["rows"]),
                "truncated": len(table["rows"]) > row_k,
            }
            for store, table in tables.items()
        },
        "candidate_evidence_sha256": object_hash(
            [records[key] for key in sorted(records)]
        ),
        "rankings_sha256": object_hash(rankings),
        "title": next(
            (table["title"] for table in tables.values() if table["title"]), ""
        ),
        "graph_covered": tables["neo4j"]["covered"] if "neo4j" in tables else None,
        "sql": tables.get("postgres", {}).get("sql"),
        "graph": tables.get("neo4j", {}).get("graph"),
        "missing_passage_ids": missing,
        "empty_stores": [store for store, ranking in rankings.items() if not ranking],
        "seconds": perf_counter() - started,
        "embedding_seconds": embedding_seconds,
        "fusion": fusion,
        "fusion_seconds": fusion_seconds,
        "fusion_weights": weights if fusion == "weighted_rrf" else None,
        "limitations": ["pinecone_table_membership_capped"]
        if "pinecone" in selected
        else [],
    }
