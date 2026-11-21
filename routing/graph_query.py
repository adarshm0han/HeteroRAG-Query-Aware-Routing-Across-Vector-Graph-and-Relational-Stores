"""Bounded traversal of existing Table -> TableRow -> Passage relationships."""

import re

SEEDS = """MATCH (t:Table {table_id:$table_id})
OPTIONAL MATCH (t)-[:HAS_ROW]->(r:TableRow)
OPTIONAL MATCH (r)-[l:LINKS_TO]->(p:Passage)
WITH t, r, l, p ORDER BY r.row_index, p.page_id, l.column, l.cell_text
WITH t, r, sum(CASE WHEN p.page_id IN $anchors THEN 1 ELSE 0 END) AS anchor_hits,
 collect(CASE WHEN p IS NULL THEN null ELSE
  {passage_id:p.page_id, column:l.column, cell_text:l.cell_text} END) AS all_links
WITH t, r, anchor_hits, all_links,
 reduce(best=size($anchors), i IN range(0,size($anchors)-1) |
  CASE WHEN any(link IN all_links WHERE link.passage_id=$anchors[i]) AND i<best THEN i ELSE best END) AS anchor_rank,
 reduce(score=0, term IN $terms | score + CASE WHEN toLower(r.text) CONTAINS term THEN 1 ELSE 0 END) AS lexical_score
ORDER BY anchor_rank, anchor_hits DESC, lexical_score DESC, r.row_index
LIMIT $seed_limit
RETURN t.title AS title, r.row_index AS row_index, r.text AS text,
 anchor_hits, all_links[..$link_limit] AS links, size(all_links)>$link_limit AS links_truncated"""

# The independent search always runs over the table. SQL seeds are a second pool,
# never a WHERE restriction on passage/lexical discovery.
SQL_SEEDS = SEEDS.replace(
    "OPTIONAL MATCH (t)-[:HAS_ROW]->(r:TableRow)",
    "OPTIONAL MATCH (t)-[:HAS_ROW]->(r:TableRow)\nWHERE r.row_index IN $seed_rows",
).replace(
    "ORDER BY anchor_rank, anchor_hits DESC, lexical_score DESC, r.row_index",
    "ORDER BY [i IN range(0,size($seed_rows)-1) WHERE $seed_rows[i]=r.row_index][0], r.row_index",
)


def combine_seeds(independent, sql):
    """Preserve every independent seed, append distinct SQL seeds within its quota."""
    records = {r["row_index"]: r for r in independent if r["row_index"] is not None}
    for record in sql:
        if record["row_index"] is not None:
            records.setdefault(record["row_index"], record)
    return list(records.values())


EXPAND = """UNWIND $seeds AS seed_index
MATCH (t:Table {table_id:$table_id})-[:HAS_ROW]->(s:TableRow {row_index:seed_index})
CALL {
 WITH t, s
 MATCH (s)-[:LINKS_TO]->(bridge:Passage)<-[:LINKS_TO]-(r:TableRow)<-[:HAS_ROW]-(t)
 WHERE r <> s
 WITH DISTINCT r, bridge
 ORDER BY r.row_index, bridge.page_id
 LIMIT $neighbor_limit
 RETURN r, bridge.page_id AS bridge_id
}
WITH DISTINCT r, seed_index, bridge_id ORDER BY r.row_index, seed_index, bridge_id
WITH r, collect({seed_row_index:seed_index, bridge_passage_id:bridge_id,
 row_index:r.row_index}) AS paths
OPTIONAL MATCH (r)-[l:LINKS_TO]->(p:Passage)
WITH r, paths, l, p ORDER BY r.row_index, p.page_id, l.column, l.cell_text
WITH r, paths, collect(CASE WHEN p IS NULL THEN null ELSE
 {passage_id:p.page_id, column:l.column, cell_text:l.cell_text} END) AS all_links
RETURN r.row_index AS row_index, r.text AS text, paths,
 all_links[..$link_limit] AS links, size(all_links)>$link_limit AS links_truncated
ORDER BY row_index"""


def query_terms(question):
    stop = {
        "what",
        "which",
        "who",
        "when",
        "where",
        "how",
        "is",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "and",
        "that",
        "this",
        "did",
        "does",
        "from",
        "by",
        "as",
        "it",
    }
    return list(
        dict.fromkeys(
            t
            for t in re.findall(r"\w+", question.lower())
            if len(t) > 1 and t not in stop
        )
    )[:24]


def merge_paths(table_id, seeds, neighbors, limit):
    """Keep seeds first, then bounded shared-passage neighbors; retain real paths."""
    records = {}
    for record in seeds:
        records[record["row_index"]] = {
            **record,
            "paths": [
                {
                    "seed_row_index": record["row_index"],
                    "bridge_passage_id": None,
                    "row_index": record["row_index"],
                }
            ],
        }
    for record in neighbors:
        index = record["row_index"]
        if index in records:
            records[index]["paths"].extend(record["paths"])
        else:
            records[index] = {**record, "anchor_hits": 0}
    for record in records.values():
        record["links"] = sorted(
            record["links"],
            key=lambda link: (
                link["passage_id"],
                link.get("column", ""),
                link.get("cell_text", ""),
            ),
        )
        record["paths"] = sorted(
            record["paths"],
            key=lambda path: (
                path["seed_row_index"],
                path["bridge_passage_id"] or "",
                path["row_index"],
            ),
        )
    rows = [
        {
            **r,
            "table_id": table_id,
            "passage_ids": sorted({l["passage_id"] for l in r["links"]}),
        }
        for r in records.values()
    ]
    return rows[:limit], len(rows) > limit
