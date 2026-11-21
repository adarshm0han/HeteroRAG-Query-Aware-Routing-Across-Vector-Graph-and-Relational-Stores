"""Rank unstructured table candidates using independent passage-to-row links."""

import re
from itertools import zip_longest

from baseline.retrieval import lexical_rank

from .fusion import interleave
from .graph_query import query_terms


def normalized(value):
    return " ".join(re.findall(r"\w+", re.sub(r"(?<=\d),(?=\d)", "", value.casefold())))


def cell_values(row):
    if "cells" in row:
        return [(c["header"], c["value"]) for c in row["cells"]]
    # Graph rows retain the same canonical column-labelled text as SQL rows.
    return [
        (part.partition(": ")[0], part.partition(": ")[2])
        for part in row["text"].split(" | ")
        if ": " in part
    ]


def literal_match(question, row):
    query = " " + normalized(question) + " "
    return any(
        len(normalized(value)) >= 3
        and query_terms(value)
        and " " + normalized(value) + " " in query
        for _, value in cell_values(row)
    )


def row_order(question, table, anchors=()):
    rows = table["rows"]
    if table.get("ordered"):
        return [row["row_index"] for row in rows]
    lexical = lexical_rank(question, rows)
    positions = {row_id: rank for rank, row_id in enumerate(lexical)}
    anchor_positions = {pid: rank for rank, pid in enumerate(anchors)}
    aliases = {}
    for row in rows:
        for link in row.get("links", []):
            if link["passage_id"] in anchor_positions and normalized(link["cell_text"]):
                key = (link["column"], normalized(link["cell_text"]))
                aliases[key] = min(
                    aliases.get(key, len(anchors)), anchor_positions[link["passage_id"]]
                )
    # All occurrences of a high-ranked entity remain candidates. No first-N-row
    # truncation is allowed before this reverse lookup over the complete table.
    return [
        row["row_index"]
        for row in sorted(
            rows,
            key=lambda row: (
                not literal_match(question, row),
                min(
                    [
                        anchor_positions[p]
                        for p in row["passage_ids"]
                        if p in anchor_positions
                    ]
                    + [
                        aliases[(column, normalized(value))]
                        for column, value in cell_values(row)
                        if (column, normalized(value)) in aliases
                    ],
                    default=len(anchor_positions),
                ),
                positions[row["row_index"]],
                row["row_index"],
            ),
        )
    ]


def reserve_passages(question, anchors, semantic_order, rows, limit):
    """Balance independent passage search with the selected rows' evidence chains."""
    available = set(semantic_order)
    # Explicit cell conditions (names, dates, categories) deserve their linked text
    # even if a generic semantic result outranks it. Reserve at most half the cap.
    explicit = list(
        dict.fromkeys(
            pid
            for row in rows
            if literal_match(question, row)
            for pid in semantic_order
            if pid in row["passage_ids"]
        )
    )[: max(1, limit // 2)]
    chains = [
        [pid for pid in semantic_order if pid in row["passage_ids"]] for row in rows
    ]
    linked = list(
        dict.fromkeys(
            pid for group in zip_longest(*chains) for pid in group if pid is not None
        )
    )
    independent = [pid for pid in anchors if pid in available]
    return list(
        dict.fromkeys([*explicit, *interleave(independent, linked), *semantic_order])
    )[:limit]
