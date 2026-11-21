"""Canonical evidence IDs, one ranking per selected store, and bounded fusion."""

import collections
import math
from itertools import zip_longest

from baseline.retrieval import rrf


def interleave(left, right):
    return [
        value
        for pair in zip_longest(left, right)
        for value in pair
        if value is not None
    ]


def fuse(
    rankings, records, row_k=8, passage_k=12, mode="rrf", weights=None, *, protected=()
):
    if mode == "rrf":
        ordered = rrf(list(rankings.values()))
    elif mode == "weighted_rrf":
        if weights is None or any(
            store not in weights
            or not math.isfinite(weights[store])
            or weights[store] <= 0
            for store in rankings
        ):
            raise ValueError(
                "Weighted RRF requires positive weights for every selected store"
            )
        scores = collections.defaultdict(float)
        for store, ranking in rankings.items():
            for rank, key in enumerate(dict.fromkeys(ranking), 1):
                scores[key] += weights[store] / (60 + rank)
        ordered = sorted(scores, key=lambda key: (-scores[key], key))
    elif mode == "concat":
        ordered = list(
            dict.fromkeys(key for ranking in rankings.values() for key in ranking)
        )
    else:
        raise ValueError("Unknown fusion method")
    limits = {"row": row_k, "passage": passage_k}
    reserved = list(dict.fromkeys(protected))
    if len(reserved) > passage_k or any(
        key not in ordered or records[key]["kind"] != "passage" for key in reserved
    ):
        raise ValueError("Invalid protected passage reservation")
    reserved = set(reserved)
    selected = []
    for key in ordered:
        record = records[key]
        kind = "row" if record["kind"] == "aggregate" else record["kind"]
        if kind == "passage" and key not in reserved and limits[kind] <= len(reserved):
            continue
        if limits[kind] <= 0:
            continue
        limits[kind] -= 1
        reserved.discard(key)
        selected.append(
            {
                **record,
                "ranked_by": [
                    store for store, ranking in rankings.items() if key in ranking
                ],
            }
        )
    return selected
