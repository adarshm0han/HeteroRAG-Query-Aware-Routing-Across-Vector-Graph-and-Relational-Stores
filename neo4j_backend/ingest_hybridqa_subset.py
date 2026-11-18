"""Batch-ingest the selected HybridQA subset into Neo4j."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

from neo4j_backend.config import Neo4jConfig


DEFAULT_DATASET = Path.home() / "datasets" / "WikiTables-WithLinks"
DEFAULT_MANIFEST = Path("data/processed/neo4j_subset_manifest.json")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def prepare_records(dataset_dir: Path, table_id: str):
    table_path = dataset_dir / "tables_tok" / f"{table_id}.json"
    passage_path = dataset_dir / "request_tok" / f"{table_id}.json"

    if not table_path.exists():
        raise FileNotFoundError(table_path)

    if not passage_path.exists():
        raise FileNotFoundError(passage_path)

    table_data = load_json(table_path)
    passage_data = load_json(passage_path)

    headers = [str(cell[0]) for cell in table_data["header"]]

    rows = []
    links = []
    linked_pages = set()

    for row_index, cells in enumerate(table_data["data"]):
        row_values = []

        for column_index, cell in enumerate(cells):
            cell_text, cell_links = cell
            column = headers[column_index]

            row_values.append(f"{column}: {cell_text}")

            for page_id in cell_links:
                linked_pages.add(page_id)

                links.append({
                    "row_id": f"{table_id}:{row_index}",
                    "page_id": page_id,
                    "column": column,
                    "cell_text": str(cell_text),
                })

        rows.append({
            "row_id": f"{table_id}:{row_index}",
            "row_index": row_index,
            "text": " | ".join(row_values),
        })

    missing = linked_pages - passage_data.keys()

    if missing:
        raise ValueError(
            f"{table_id}: {len(missing)} linked pages lack passage data. "
            f"Examples: {sorted(missing)[:5]}"
        )

    passages = []

    for page_id in sorted(linked_pages):
        text = passage_data[page_id]

        passages.append({
            "page_id": page_id,
            "title": page_id.rsplit("/", 1)[-1].replace("_", " "),
            "text_length": len(text),
        })

    table = {
        "table_id": table_id,
        "title": table_data["title"],
        "url": table_data["url"],
        "headers": headers,
    }

    return table, rows, passages, links


def create_constraints(session):
    queries = [
        """
        CREATE CONSTRAINT heterorag_table_id IF NOT EXISTS
        FOR (t:Table) REQUIRE t.table_id IS UNIQUE
        """,
        """
        CREATE CONSTRAINT heterorag_row_id IF NOT EXISTS
        FOR (r:TableRow) REQUIRE r.row_id IS UNIQUE
        """,
        """
        CREATE CONSTRAINT heterorag_passage_id IF NOT EXISTS
        FOR (p:Passage) REQUIRE p.page_id IS UNIQUE
        """,
    ]

    for query in queries:
        session.run(query).consume()


def write_batch(session, query, records, batch_size, **params):
    for batch in chunks(records, batch_size):
        def write_transaction(tx):
            tx.run(
                query,
                records=batch,
                **params,
            ).consume()

        session.execute_write(write_transaction)


def write_table(session, table):
    def write_transaction(tx):
        tx.run(
            """
            MERGE (t:Table {table_id: $table_id})
            SET t.title = $title,
                t.url = $url,
                t.headers = $headers
            """,
            **table,
        ).consume()

    session.execute_write(write_transaction)


def write_records(session, table, rows, passages, links, batch_size):
    write_table(session, table)

    write_batch(
        session,
        """
        UNWIND $records AS rec
        MATCH (t:Table {table_id: $table_id})
        MERGE (r:TableRow {row_id: rec.row_id})
        SET r.row_index = rec.row_index,
            r.text = rec.text,
            r.table_id = $table_id
        MERGE (t)-[:HAS_ROW]->(r)
        """,
        rows,
        batch_size,
        table_id=table["table_id"],
    )

    write_batch(
        session,
        """
        UNWIND $records AS rec
        MERGE (p:Passage {page_id: rec.page_id})
        SET p.title = rec.title,
            p.text_length = rec.text_length
        """,
        passages,
        batch_size,
    )

    write_batch(
        session,
        """
        UNWIND $records AS rec
        MATCH (r:TableRow {row_id: rec.row_id})
        MATCH (p:Passage {page_id: rec.page_id})
        MERGE (r)-[rel:LINKS_TO {
            column: rec.column,
            page_id: rec.page_id
        }]->(p)
        SET rel.cell_text = rec.cell_text
        """,
        links,
        batch_size,
    )


def verify_graph(session):
    result = session.run(
        """
        MATCH (t:Table)
        WITH count(t) AS tables

        MATCH (r:TableRow)
        WITH tables, count(r) AS rows

        MATCH (p:Passage)
        WITH tables, rows, count(p) AS passages

        MATCH ()-[hr:HAS_ROW]->()
        WITH tables, rows, passages, count(hr) AS has_row

        MATCH ()-[lt:LINKS_TO]->()
        RETURN tables, rows, passages,
               has_row, count(lt) AS links
        """
    ).single()

    print("\n========== NEO4J GRAPH COUNTS ==========")

    for key in ["tables", "rows", "passages", "has_row", "links"]:
        print(f"{key}: {result[key]:,}")

    print(f"Total nodes: {result['tables'] + result['rows'] + result['passages']:,}")
    print(f"Total relationships: {result['has_row'] + result['links']:,}")

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing with the first N tables.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    manifest = load_json(args.manifest)
    selected = manifest["selected_tables"]

    if args.limit is not None:
        selected = selected[:args.limit]

    print(f"Selected tables to process: {len(selected):,}")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Batch size: {args.batch_size}")

    expected = {
        "tables": 0,
        "rows": 0,
        "passages": 0,
        "links": 0,
    }

    cfg = Neo4jConfig()

    if not args.dry_run and not cfg.password:
        raise RuntimeError("Set NEO4J_PASSWORD before ingestion.")

    driver = None

    if not args.dry_run:
        driver = GraphDatabase.driver(
            cfg.uri,
            auth=(cfg.user, cfg.password),
        )
        driver.verify_connectivity()

    try:
        session_context = (
            driver.session(database=cfg.database)
            if driver is not None
            else None
        )

        if session_context is not None:
            session = session_context
        else:
            session = None

        for index, item in enumerate(selected, 1):
            table_id = item["table_id"]

            table, rows, passages, links = prepare_records(
                args.dataset_dir,
                table_id,
            )

            expected["tables"] += 1
            expected["rows"] += len(rows)
            expected["passages"] += len(passages)
            expected["links"] += len(links)

            if session is not None:
                if index == 1:
                    create_constraints(session)

                write_records(
                    session,
                    table,
                    rows,
                    passages,
                    links,
                    args.batch_size,
                )

            if index % 100 == 0 or index == len(selected):
                print(
                    f"Processed {index:,}/{len(selected):,} tables | "
                    f"rows: {expected['rows']:,} | "
                    f"passage records: {expected['passages']:,} | "
                    f"links: {expected['links']:,}"
                )

        print("\n========== INGESTION TOTALS ==========")
        print(f"Tables: {expected['tables']:,}")
        print(f"Rows: {expected['rows']:,}")
        print(f"Passage records processed: {expected['passages']:,}")
        print(f"Hyperlink records: {expected['links']:,}")

        if args.dry_run:
            print("\nDry run completed. No Neo4j writes performed.")
            return 0

        actual = verify_graph(session)

        print("\nExpected unique passage nodes from manifest:")
        print(f"{manifest['totals']['passage_nodes']:,}")

        print("\nExpected graph totals from manifest:")
        print(f"Nodes: {manifest['totals']['total_nodes']:,}")
        print(f"Relationships: {manifest['totals']['total_relationships']:,}")

        print("\nCompare actual counts above with the manifest.")
        print("Shared passages are deduplicated by page_id.")

        return 0

    finally:
        if driver is not None:
            driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
