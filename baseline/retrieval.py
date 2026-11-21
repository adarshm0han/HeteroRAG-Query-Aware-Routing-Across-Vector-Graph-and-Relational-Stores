"""Fixed retrieval and fusion. No reference answers, traced nodes, or decision models."""

import collections
import math
import re

import numpy as np

STOP = {
    "the",
    "has",
    "or",
    "at",
    "do",
    "from",
    "it",
    "and",
    "does",
    "been",
    "were",
    "for",
    "where",
    "are",
    "which",
    "did",
    "as",
    "have",
    "by",
    "who",
    "this",
    "that",
    "in",
    "was",
    "an",
    "had",
    "a",
    "on",
    "its",
    "be",
    "what",
    "with",
    "when",
    "of",
    "is",
    "to",
    "how",
}


def tokens(text):
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOP]


def lexical_rank(question, rows):
    """BM25 over this question's provided table rows, stable row-index tie breaking."""
    corpus = [tokens(row["text"]) for row in rows]
    n = len(corpus)
    if not n:
        return []
    frequencies = collections.Counter(t for doc in corpus for t in set(doc))
    average = sum(map(len, corpus)) / n or 1
    terms = set(tokens(question))
    scored = []
    for row, doc in zip(rows, corpus):
        counts = collections.Counter(doc)
        score = sum(
            math.log(1 + (n - frequencies[t] + 0.5) / (frequencies[t] + 0.5))
            * (counts[t] * 2.5)
            / (counts[t] + 1.5 * (0.25 + 0.75 * len(doc) / average))
            for t in terms
            if counts[t]
        )
        scored.append((row["row_index"], score))
    return [key for key, _ in sorted(scored, key=lambda x: (-x[1], x[0]))]


def rrf(rankings, k=60):
    scores = collections.defaultdict(float)
    for ranking in rankings:
        # A source contributes at most one vote per canonical evidence ID.
        for rank, key in enumerate(dict.fromkeys(ranking), 1):
            scores[key] += 1 / (k + rank)
    return sorted(scores, key=lambda key: (-scores[key], str(key)))


def vector_rank(query_vector, passages):
    if not passages:
        return []
    keys = sorted(passages)
    matrix = np.array([passages[key]["values"] for key in keys], dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != 384 or query.shape != (384,):
        raise ValueError(
            "Embedding dimension differs from the existing 384-dimensional index"
        )
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    if np.any(norms == 0) or not np.all(np.isfinite(matrix)):
        raise ValueError("Invalid passage vector")
    scores = dict(zip(keys, (matrix @ query / norms).tolist()))
    return sorted(keys, key=lambda key: (-scores[key], key))


def link_set(rows, row_ids):
    wanted = set(row_ids)
    return {p for row in rows if row["row_index"] in wanted for p in row["passage_ids"]}


def build_plans(
    question, sql_rows, graph_rows, passages, query_vector, row_k=8, passage_k=12
):
    lexical = lexical_rank(question, sql_rows)
    vector = vector_rank(query_vector, passages)
    positions = {key: rank for rank, key in enumerate(vector)}
    sql_linked = link_set(sql_rows, lexical[:row_k])
    sql_passages = [key for key in vector if key in sql_linked][:passage_k]
    # Fixed graph plan: seed with semantic passage anchors; traverse back to rows,
    # then expand to their other passages. No query-conditioned learned routing.
    anchors = set(vector[:passage_k])
    reached = [row for row in graph_rows if anchors.intersection(row["passage_ids"])]
    reached.sort(
        key=lambda row: (
            min(positions.get(p, 10**9) for p in row["passage_ids"]),
            row["row_index"],
        )
    )
    graph_lexical = lexical_rank(question, graph_rows)
    graph_ranked = rrf([[r["row_index"] for r in reached], graph_lexical])[:row_k]
    graph_linked = link_set(graph_rows, graph_ranked)
    graph_passages = [key for key in vector if key in graph_linked][:passage_k]
    combined_rows = rrf([lexical[:row_k], graph_ranked])[:row_k]
    combined_passages = rrf([vector[:passage_k], sql_passages, graph_passages])[
        :passage_k
    ]

    def evidence(row_ids, passage_ids, graph=False):
        rows_by_id = {r["row_index"]: r for r in sql_rows}
        graph_by_id = {r["row_index"]: r for r in graph_rows}
        items = []
        for index in row_ids:
            row = rows_by_id[index]
            items.append(
                {
                    "id": f"row:{row['table_id']}:{index}",
                    "kind": "row",
                    "row_index": index,
                    "table_id": row["table_id"],
                    "text": row["text"],
                    "links": row["links"],
                    "sources": ["postgres"],
                }
            )
            if graph and index in graph_by_id:
                items[-1]["sources"].append("neo4j")
                items[-1]["graph_links"] = graph_by_id[index]["links"]
        for pid in passage_ids:
            p = passages[pid]
            items.append(
                {
                    "id": pid,
                    "kind": "passage",
                    "passage_id": pid,
                    "title": p["title"],
                    "text": p["text"],
                    "sources": ["pinecone"],
                }
            )
        return items

    return {
        "sql_only": evidence(lexical[:row_k], []),
        "vector_only": evidence([], vector[:passage_k]),
        "sql_links": evidence(lexical[:row_k], sql_passages),
        "all_stores": evidence(combined_rows, combined_passages, graph=True),
    }


def pack_context(question, title, items, encoder, budget):
    """Bound serialized evidence tokens; retain whole items to preserve citations/provenance."""
    import json

    prefix = json.dumps(
        {"question": question, "table_title": title}, ensure_ascii=False
    )
    used = len(encoder.encode(prefix))
    selected = []
    for index, original in enumerate(items, 1):
        # Short ASCII labels avoid model spelling/Unicode changes to source IDs.
        item = {**original, "source_id": original["id"], "id": f"E{index}"}
        cost = len(encoder.encode(json.dumps(item, ensure_ascii=False))) + 2
        if used + cost <= budget:
            selected.append(item)
            used += cost
    text = prefix + "\nEvidence:\n" + json.dumps(selected, ensure_ascii=False)
    # Count the final serialization too, not just the estimated per-item totals.
    while selected and len(encoder.encode(text)) > budget:
        selected.pop()
        text = prefix + "\nEvidence:\n" + json.dumps(selected, ensure_ascii=False)
    if len(encoder.encode(text)) > budget:
        raise ValueError("Question exceeds context budget")
    return text, selected, len(encoder.encode(text))
