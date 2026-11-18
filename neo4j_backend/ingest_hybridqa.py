
"""Ingest one original HybridQA table and its linked passages into Neo4j."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

from neo4j_backend.config import Neo4jConfig


DEFAULT_DATASET = (
    r"\\wsl.localhost\Ubuntu\home\appadmin\datasets\WikiTables-WithLinks"
)

DEFAULT_TABLE_ID = "List_of_mosques_in_Afghanistan_0"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_records(dataset_dir: Path, table_id: str):
    table_path = dataset_dir / "tables_tok" / f"{table_id}.json"
    passage_path = dataset_dir / "request_tok" / f"{table_id}.json"

    if not table_path.exists():
        raise FileNotFoundError(f"Table file not found: {table_path}")
    if not passage_path.exists():
        raise FileNotFoundError(f"Passage file not found: {passage_path}")

    table = load_json(table_path)
    passages = load_json(passage_path)

    headers = [str(cell[0]) for cell in table["header"]]

    rows = []
    links = []
    all_linked_pages = set()

    for row_index, cells in enumerate(table["data"]):
        row_values = []
        for column_index, cell in enumerate(cells):
            cell_text, cell_links = cell
            column = headers[column_index]

            row_values.append(f"{column}: {cell_text}")

            for page_id in cell_links:
                all_linked_pages.add(page_id)
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

    missing_passages = all_linked_pages - passages.keys()
    if missing_passages:
        raise ValueError(
            f"{len(missing_passages)} linked pages have no passage text. "
            f"Examples: {sorted(missing_passages)[:5]}"
        )

    passage_records = []
    for page_id in sorted(all_linked_pages):
        text = passages[page_id]
        title = page_id.rsplit("/", 1)[-1].replace("_", " ")

        passage_records.append({
            "page_id": page_id,
            "title": title,
            "text_length": len(text),
        })

    table_record = {
        "table_id": table_id,
        "title": table["title"],
        "url": table["url"],
        "headers": headers,
    }

    return table_record, rows, passage_records, links



def write_to_neo4j(table, rows, passages, links):
    cfg = Neo4jConfig()

    if not cfg.password:
        raise RuntimeError("Set NEO4J_PASSWORD before ingestion.")

    driver = GraphDatabase.driver(
        cfg.uri,
        auth=(cfg.user, cfg.password),
    )

    try:
        driver.verify_connectivity()

        with driver.session(database=cfg.database) as session:

            # Step 1: Create constraints separately from data writes.
            constraints = [
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

            for query in constraints:
                session.run(query).consume()

            print("Neo4j constraints verified.")

            # Step 2: Write graph data in a separate transaction.
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

                tx.run(
                    """
                    UNWIND $rows AS rec
                    MATCH (t:Table {table_id: $table_id})
                    MERGE (r:TableRow {row_id: rec.row_id})
                    SET r.row_index = rec.row_index,
                        r.text = rec.text,
                        r.table_id = $table_id
                    MERGE (t)-[:HAS_ROW]->(r)
                    """,
                    rows=rows,
                    table_id=table["table_id"],
                ).consume()

                tx.run(
                    """
                    UNWIND $passages AS rec
                    MERGE (p:Passage {page_id: rec.page_id})
                    SET p.title = rec.title,
                        p.text_length = rec.text_length
                    """,
                    passages=passages,
                ).consume()

                tx.run(
                    """
                    UNWIND $links AS rec
                    MATCH (r:TableRow {row_id: rec.row_id})
                    MATCH (p:Passage {page_id: rec.page_id})
                    MERGE (r)-[rel:LINKS_TO {
                        column: rec.column,
                        page_id: rec.page_id
                    }]->(p)
                    SET rel.cell_text = rec.cell_text
                    """,
                    links=links,
                ).consume()

            session.execute_write(write_transaction)

            # Step 3: Verify the resulting graph.
            result = session.run(
                """
                MATCH (t:Table {table_id: $table_id})
                OPTIONAL MATCH (t)-[:HAS_ROW]->(r:TableRow)
                OPTIONAL MATCH (r)-[rel:LINKS_TO]->(p:Passage)
                RETURN count(DISTINCT t) AS tables,
                       count(DISTINCT r) AS rows,
                       count(DISTINCT p) AS passages,
                       count(rel) AS links
                """,
                table_id=table["table_id"],
            ).single()

            print("\nNeo4j verification:")
            print(f"Tables: {result['tables']}")
            print(f"Rows: {result['rows']}")
            print(f"Passages: {result['passages']}")
            print(f"Links: {result['links']}")

    finally:
        driver.close()



def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET,
        help="Root directory of WikiTables-WithLinks",
    )
    parser.add_argument(
        "--table-id",
        default=DEFAULT_TABLE_ID,
        help="HybridQA table ID to ingest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and validate records without writing to Neo4j",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    table, rows, passages, links = prepare_records(
        dataset_dir,
        args.table_id,
    )

    print(f"Table: {table['title']}")
    print(f"Table ID: {table['table_id']}")
    print(f"Columns: {table['headers']}")
    print(f"Rows prepared: {len(rows)}")
    print(f"Passages prepared: {len(passages)}")
    print(f"Hyperlink relationships prepared: {len(links)}")

    if rows:
        print("\nSample row:")
        print(json.dumps(rows[3], indent=2))

    if links:
        print("\nSample hyperlink:")
        print(json.dumps(links[0], indent=2))

    if args.dry_run:
        print("\nDry run only. No Neo4j writes performed.")
        return 0

    if not rows or not passages or not links:
        print("Refusing to ingest empty or incomplete data.", file=sys.stderr)
        return 1

    write_to_neo4j(table, rows, passages, links)
    print("\nHybridQA ingestion completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
