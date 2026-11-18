# HeteroRAG Neo4j backend (initial implementation)

This is an initial, read-only-from-PostgreSQL ingestion path and graph retriever designed to fit the existing HeteroRAG project. It deliberately does **not** create or modify PostgreSQL objects. It reads the team's `hybridqa.rag_rows` view, then builds a Neo4j graph of tables, rows, and normalized cell-value entities.

## Graph model

- `(:Table {table_id, title})-[:HAS_ROW]->(:TableRow {row_id, row_index, text})`
- `(:TableRow)-[:HAS_ENTITY {column}]->(:Entity {key, value})`
- `Entity.key` is a case-normalized cell value. Rows that share values connect through the same entity node.

This is a useful baseline graph for discovering related rows across tables. It does not yet ingest passage hyperlinks because we must first verify the exact link fields available in the team's HybridQA source. Do not claim that these inferred shared-cell links are equivalent to the original Wikipedia hyperlinks.

## 1. Install dependency

From the repository root, in your active Python environment:

```powershell
pip install neo4j "psycopg[binary]"
```

## 2. Configure environment variables

Set these in PowerShell for the current terminal. Use the PostgreSQL connection details supplied by your team; do not commit credentials.

```powershell
$env:POSTGRES_HOST = "<team-postgres-host>"
$env:POSTGRES_PORT = "5432"
$env:POSTGRES_DB = "hybridqa"
$env:POSTGRES_USER = "<read-only-user>"
$env:POSTGRES_PASSWORD = "<password>"
$env:POSTGRES_SOURCE_VIEW = "hybridqa.rag_rows"

$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASSWORD = "<your-local-neo4j-password>"
$env:NEO4J_DATABASE = "neo4j"
```

If the shared PostgreSQL database is accessible only from campus/VPN, connect through the approved ASU method first. Do not expose PostgreSQL or Neo4j to the public internet.

## 3. Inspect the source schema first

From the repository root:

```powershell
python -m neo4j_backend.ingest_from_postgres --print-schema
```

The script prints the view columns, one sample row, and then exits without writing to Neo4j. Check that the detected columns match the real view. If the source uses different column names, add them to `CANDIDATES` in `ingest_from_postgres.py` before proceeding.

Then prepare a dry run:

```powershell
python -m neo4j_backend.ingest_from_postgres --dry-run
```

Review the prepared sample. In particular, verify that the row text preserves table headers and that `row_id` is stable and unique. If it is not, update the row ID construction before ingestion.

## 4. Ingest

Only after reviewing the schema and dry-run output:

```powershell
python -m neo4j_backend.ingest_from_postgres
```

Ingestion uses `MERGE` and uniqueness constraints so reruns update existing nodes instead of intentionally duplicating them. The graph writes only to Neo4j.

## 5. Query the graph

```powershell
python -m neo4j_backend.retriever "Which table rows mention Arizona and population?" --top-k 5
```

The retriever returns evidence records with `id`, `source`, `text`, `score`, `rank`, and `metadata`, suitable for an adapter to a shared HeteroRAG result contract.

Example Python usage:

```python
from neo4j_backend.retriever import Neo4jRetriever

retriever = Neo4jRetriever()
try:
    evidence = retriever.retrieve("question text here", top_k=5)
finally:
    retriever.close()
```

## Integration contract

The retriever returns a list of dictionaries shaped like:

```python
{
    "id": "neo4j:<stable-row-id>",
    "source": "neo4j",
    "text": "Table: ...\\\\nRow: ...",
    "score": 0.5,
    "rank": 1,
    "metadata": {
        "table_id": "...",
        "row_id": "...",
        "row_index": "..."
    }
}
```

`score` is a lexical-overlap score, **not** a probability and not directly comparable to vector cosine similarity. Normalize or rank-fuse scores in the HeteroRAG fusion layer rather than comparing raw values across backends.

## Current limitations / next steps

1. Verify the actual `rag_rows` schema before ingestion; automatic column detection is intentionally conservative.
2. Add a source-specific loader for original HybridQA table-to-passage hyperlinks, then represent those explicit links in Neo4j.
3. Add tests against a small fixture and integration tests against a disposable Neo4j instance.
4. Add graph retrieval latency and recall measurements to the common benchmark.
5. Align the exact evidence schema with the existing PostgreSQL and vector adapters before integrating the router.
