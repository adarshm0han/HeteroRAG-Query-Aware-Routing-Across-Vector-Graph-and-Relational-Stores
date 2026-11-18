"""Build a small evidence graph from the team's PostgreSQL rag_rows view.

This script intentionally inspects the source schema first. It auto-detects
common column names, but requires a clear table ID and row content/value field.
Use --print-schema before the first ingestion run.

Expected graph model:
  (:Table {table_id, title})-[:HAS_ROW]->(:TableRow {row_id, row_index, text})
  (:TableRow)-[:HAS_ENTITY {column}]->(:Entity {key, value})

Entities are normalized cell values shared across rows/tables. This creates
cross-row links without inventing semantic relationships that are not present
in the source. It is a baseline graph; explicit hyperlink edges can be added
once we verify which link fields the HybridQA release provides.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

import psycopg
from psycopg import sql
from neo4j import GraphDatabase

from neo4j_backend.config import Neo4jConfig, PostgresConfig

CANDIDATES = {
    "table_id": ["table_id", "tableid", "table", "id_table"],
    "row_index": ["row_index", "row_idx", "row_number", "row_num", "index"],
    "row_id": ["row_id", "id", "evidence_id"],
    "title": ["table_title", "title", "table_name"],
    "headers": ["headers", "table_headers", "header", "columns"],
    "row_values": ["row_values", "values", "row_data", "row_json", "row_content", "content", "cells"],
}


def normalize_key(value: Any) -> str:
    """Stable key for joining equivalent cell values across rows."""
    return re.sub(r"\\s+", " ", str(value).strip()).casefold()


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        # Preserve field names for dict-shaped rows.
        return [f"{k}: {v}" for k, v in value.items() if v is not None and str(v).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, (list, dict)):
                return as_list(parsed)
        except (json.JSONDecodeError, TypeError):
            pass
        return [stripped]
    return [value]


def detect_columns(columns: list[str]) -> dict[str, str | None]:
    by_lower = {c.lower(): c for c in columns}
    detected: dict[str, str | None] = {}
    for logical, candidates in CANDIDATES.items():
        detected[logical] = next((by_lower[name] for name in candidates if name in by_lower), None)
    return detected


def get_source_rows() -> tuple[list[str], list[dict[str, Any]]]:
    pg = PostgresConfig()
    # The source view name is configurable but must be a schema-qualified identifier.
    parts = pg.source_view.split(".")
    if len(parts) != 2 or not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p) for p in parts):
        raise ValueError("POSTGRES_SOURCE_VIEW must be a schema-qualified name, e.g. hybridqa.rag_rows")

    with psycopg.connect(
        host=pg.host, port=pg.port, dbname=pg.dbname,
        user=pg.user, password=pg.password,
        connect_timeout=10,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
                (parts[0], parts[1]),
            )
            columns = [r[0] for r in cur.fetchall()]
            if not columns:
                raise RuntimeError(f"Could not find source view {pg.source_view!r} or it has no visible columns")
            cur.execute(sql.SQL("SELECT * FROM {}.{} LIMIT 0").format(sql.Identifier(parts[0]), sql.Identifier(parts[1])))
            names = [d.name for d in cur.description]
            cur.execute(sql.SQL("SELECT * FROM {}.{}").format(sql.Identifier(parts[0]), sql.Identifier(parts[1])))
            rows = [dict(zip(names, row)) for row in cur.fetchall()]
    return names, rows


def prepare_records(rows: list[dict[str, Any]], columns: dict[str, str | None]) -> list[dict[str, Any]]:
    table_col = columns.get("table_id")
    if not table_col:
        raise ValueError("Could not auto-detect a table ID column. Use --print-schema and update CANDIDATES.")
    value_col = columns.get("row_values")
    if not value_col:
        raise ValueError("Could not auto-detect a row content/value column. Use --print-schema and update CANDIDATES.")

    records = []
    for ordinal, row in enumerate(rows):
        table_id = row.get(table_col)
        if table_id is None:
            continue
        row_index_col = columns.get("row_index")
        row_id_col = columns.get("row_id")
        title_col = columns.get("title")
        headers_col = columns.get("headers")
        raw_values = row.get(value_col)
        values = as_list(raw_values)
        headers = as_list(row.get(headers_col)) if headers_col else []
        title = str(row.get(title_col) or "") if title_col else ""
        row_index = row.get(row_index_col) if row_index_col else ordinal
        row_id = row.get(row_id_col) if row_id_col else f"{table_id}:{row_index}"

        # If values are a dictionary, preserve header/value associations.
        cell_pairs = []
        if isinstance(raw_values, dict):
            cell_pairs = [(str(k), v) for k, v in raw_values.items()]
        elif isinstance(raw_values, str):
            try:
                parsed = json.loads(raw_values)
                if isinstance(parsed, dict):
                    cell_pairs = [(str(k), v) for k, v in parsed.items()]
                elif isinstance(parsed, list):
                    cell_pairs = [
                        (str(headers[i]) if i < len(headers) else f"column_{i}", v)
                        for i, v in enumerate(parsed)
                    ]
            except json.JSONDecodeError:
                pass
        elif isinstance(raw_values, (list, tuple)):
            cell_pairs = [
                (str(headers[i]) if i < len(headers) else f"column_{i}", v)
                for i, v in enumerate(raw_values)
            ]

        if not cell_pairs:
            cell_pairs = [(f"value_{i}", v) for i, v in enumerate(values)]

        cell_pairs = [(h, v) for h, v in cell_pairs if v is not None and str(v).strip()]
        row_text = " | ".join(f"{header}: {value}" for header, value in cell_pairs)
        records.append({
            "table_id": str(table_id),
            "table_title": title,
            "row_id": str(row_id),
            "row_index": str(row_index),
            "row_text": row_text,
            "cells": [{"column": h, "value": str(v).strip()} for h, v in cell_pairs],
        })
    return records


def ingest(records: list[dict[str, Any]]) -> None:
    cfg = Neo4jConfig()
    if not cfg.password:
        raise RuntimeError("Set NEO4J_PASSWORD before running ingestion.")
    driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
    try:
        driver.verify_connectivity()
        with driver.session(database=cfg.database) as session:
            session.run("CREATE CONSTRAINT heterorag_table_id IF NOT EXISTS FOR (n:Table) REQUIRE n.table_id IS UNIQUE")
            session.run("CREATE CONSTRAINT heterorag_row_id IF NOT EXISTS FOR (n:TableRow) REQUIRE n.row_id IS UNIQUE")
            session.run("CREATE CONSTRAINT heterorag_entity_key IF NOT EXISTS FOR (n:Entity) REQUIRE n.key IS UNIQUE")
            session.run("CREATE FULLTEXT INDEX heterorag_table_row_text IF NOT EXISTS FOR (n:TableRow) ON EACH [n.text]")

            batch_size = 500
            for start in range(0, len(records), batch_size):
                batch = records[start:start + batch_size]
                session.execute_write(_write_batch, batch)
                print(f"Ingested {min(start + len(batch), len(records))}/{len(records)} rows")

            counts = session.run(
                "MATCH (t:Table) WITH count(t) AS tables "
                "MATCH (r:TableRow) WITH tables, count(r) AS rows "
                "MATCH (e:Entity) RETURN tables, rows, count(e) AS entities"
            ).single()
            print(f"Graph ready: {counts['tables']} tables, {counts['rows']} rows, {counts['entities']} entities")
    finally:
        driver.close()


def _write_batch(tx, records: list[dict[str, Any]]) -> None:
    tx.run(
        "UNWIND $records AS rec "
        "MERGE (t:Table {table_id: rec.table_id}) "
        "ON CREATE SET t.title = rec.table_title "
        "ON MATCH SET t.title = CASE WHEN rec.table_title <> '' THEN rec.table_title ELSE t.title END "
        "MERGE (r:TableRow {row_id: rec.row_id}) "
        "SET r.table_id = rec.table_id, r.row_index = rec.row_index, r.text = rec.row_text "
        "MERGE (t)-[:HAS_ROW]->(r) "
        "WITH r, rec UNWIND rec.cells AS cell "
        "WITH r, cell, toLower(trim(cell.value)) AS normalized "
        "WHERE normalized <> '' AND size(normalized) <= 300 "
        "MERGE (e:Entity {key: normalized}) "
        "ON CREATE SET e.value = trim(cell.value) "
        "MERGE (r)-[rel:HAS_ENTITY {column: cell.column}]->(e)",
        records=records,
    ).consume()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-schema", action="store_true", help="Print source view columns and sample record; do not write to Neo4j")
    parser.add_argument("--dry-run", action="store_true", help="Prepare records and print one example without writing to Neo4j")
    args = parser.parse_args()

    columns, rows = get_source_rows()
    print(f"Source view columns: {columns}")
    print(f"Source rows: {len(rows)}")
    if rows:
        print("Sample source row:")
        print(json.dumps(rows[0], indent=2, default=str)[:5000])

    if args.print_schema:
        return 0

    detected = detect_columns(columns)
    print("Detected column mapping:")
    print(json.dumps(detected, indent=2))
    records = prepare_records(rows, detected)
    print(f"Prepared {len(records)} graph row records")
    if records:
        print("Prepared sample:")
        print(json.dumps(records[0], indent=2)[:3000])
    if args.dry_run:
        print("Dry run only; no Neo4j writes performed.")
        return 0
    if not records:
        print("No records prepared; refusing to create an empty graph.", file=sys.stderr)
        return 2
    ingest(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
