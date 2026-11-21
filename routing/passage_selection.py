"""Bounded passage discovery before row truncation, with lexical/semantic ranking."""

from itertools import zip_longest

from baseline.retrieval import lexical_rank, tokens, vector_rank

from .row_selection import row_order

CANDIDATE_PASSAGES = 96
VERSION = "bounded_link_expansion_hybrid_rank_v1"


def candidate_links(question, tables, anchors, limit=CANDIDATE_PASSAGES):
    """Use only rows returned by selected stores, never an auxiliary corpus lookup.

    Round-robin links across relevant rows prevents one link-heavy row consuming
    the pool. Ranking/truncating the final rows happens after these passages arrive.
    """
    chains = []
    for table in tables.values():
        rows = {row["row_index"]: row for row in table["rows"]}
        chains.extend(
            sorted(set(rows[index]["passage_ids"]))
            for index in row_order(question, table, anchors)
        )
    ids = list(
        dict.fromkeys(
            pid for group in zip_longest(*chains) for pid in group if pid is not None
        )
    )
    return ids[:limit], {
        "available": len(ids),
        "cap": limit,
        "truncated": len(ids) > limit,
    }


def passage_order(question, vector, passages):
    """Combine independent lexical and vector ranks; preserve rare literal facts."""
    semantic = vector_rank(vector, passages)
    if not passages:
        return []
    keys = sorted(passages)
    query = set(tokens(question))
    rows = [
        {"row_index": i, "text": passages[pid]["title"] + ". " + passages[pid]["text"]}
        for i, pid in enumerate(keys)
    ]
    lexical = [
        keys[i]
        for i in lexical_rank(question, rows)
        if query & set(tokens(rows[i]["text"]))
    ]
    scores = {}
    for ranking in (semantic, lexical):
        for rank, pid in enumerate(ranking, 1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (10 + rank)
    return sorted(scores, key=lambda pid: (-scores[pid], pid))
