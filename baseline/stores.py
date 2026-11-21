"""Read-only adapters for the already populated stores. No seeding or schema changes."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter, sleep
from urllib.parse import quote

import psycopg
from neo4j import READ_ACCESS, GraphDatabase, Query
from pinecone import Pinecone
from psycopg.rows import dict_row


class Stores:
    def __init__(self, cfg):
        self.pg_url = cfg["DATABASE_URL"]
        self.graph = GraphDatabase.driver(
            cfg["NEO4J_URI"],
            auth=(cfg["NEO4J_USERNAME"], cfg["NEO4J_PASSWORD"]),
            connection_timeout=30,
            connection_acquisition_timeout=30,
            max_transaction_retry_time=15,
        )
        self.graph_database = cfg.get("NEO4J_DATABASE", "neo4j")
        self.pc = Pinecone(api_key=cfg["PINECONE_API_KEY"])
        info = self.pc.describe_index(cfg.get("PINECONE_INDEX", "hybridqa-passages"))
        if info.dimension != 384 or info.metric != "cosine":
            raise ValueError("Unexpected Pinecone index configuration")
        self.index = self.pc.Index(host=info.host)
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.metadata = {
            "pinecone_index": info.name,
            "dimension": info.dimension,
            "metric": info.metric,
            "graph_database": self.graph_database,
        }

    def close(self):
        self.pool.shutdown()
        self.graph.close()
        self.index.close()

    def graph_tables(self):
        with self.graph.session(
            database=self.graph_database, default_access_mode=READ_ACCESS
        ) as session:
            return sorted(
                r["table_id"]
                for r in session.run(
                    Query("MATCH (t:Table) RETURN t.table_id AS table_id", timeout=30)
                )
            )

    def sql_table(self, table_id):
        start = perf_counter()
        for attempt in range(1, 4):
            try:
                # psycopg on Windows needs a selector loop. A fresh pooled connection
                # per table prevents a dead socket from stalling the entire corpus.
                with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
                    records, header_links = runner.run(self._sql_records(table_id))
                break
            except (psycopg.OperationalError, TimeoutError):
                if attempt == 3:
                    raise
                sleep(attempt)
        if not records:
            raise ValueError("Supplied table was not found in PostgreSQL")
        rows = []
        for r in records:
            links = [
                {
                    "column_index": c["column_index"],
                    "column": c["header"],
                    "cell_text": c["value"],
                    "passage_id": pid,
                }
                for c in r["cells"]
                for pid in c["entity_ids"]
            ]
            rows.append(
                {
                    "table_id": table_id,
                    "row_index": r["row_index"],
                    "text": r["evidence_text"],
                    "cells": r["cells"],
                    "links": links,
                    "passage_ids": sorted({p["passage_id"] for p in links}),
                }
            )
        return {
            "rows": rows,
            "title": records[0]["title"],
            "url": records[0]["url"],
            "header_passage_ids": [r["wiki_id"] for r in header_links],
            "seconds": perf_counter() - start,
            "queries": 2 * attempt,
            "connection_attempts": attempt,
        }

    async def _sql_records(self, table_id):
        connection = await psycopg.AsyncConnection.connect(
            self.pg_url,
            connect_timeout=15,
            row_factory=dict_row,
            options="-c default_transaction_read_only=on",
        )

        async def query():
            async with connection.transaction():
                await connection.execute("SET LOCAL statement_timeout = '30s'")
                cursor = await connection.execute(
                    """SELECT table_id, title, url, row_index, evidence_text, cells
                    FROM hybridqa.rag_rows WHERE table_id = %s ORDER BY row_index""",
                    (table_id,),
                )
                rows = await cursor.fetchall()
                cursor = await connection.execute(
                    """SELECT DISTINCT e.wiki_id
                    FROM hybridqa.wiki_tables t JOIN hybridqa.column_links l ON l.table_pk=t.id
                    JOIN hybridqa.entities e ON e.id=l.entity_pk WHERE t.table_id=%s""",
                    (table_id,),
                )
                return rows, await cursor.fetchall()

        task = asyncio.create_task(query())
        try:
            done, _ = await asyncio.wait({task}, timeout=40)
            if not done:
                # Close the socket before cancellation, so rollback/cancel cannot
                # itself wait indefinitely on the disconnected network path.
                await connection.close()
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, psycopg.Error):
                    pass
                raise TimeoutError("PostgreSQL client deadline exceeded")
            return task.result()
        finally:
            await connection.close()

    def graph_table(self, table_id):
        start = perf_counter()
        with self.graph.session(
            database=self.graph_database, default_access_mode=READ_ACCESS
        ) as session:
            records = session.run(
                Query(
                    """MATCH (:Table {table_id:$table_id})-[:HAS_ROW]->(r:TableRow)
                OPTIONAL MATCH (r)-[l:LINKS_TO]->(p:Passage)
                RETURN r.row_index AS row_index, r.text AS text,
                  collect(CASE WHEN p IS NULL THEN null ELSE
                    {passage_id:p.page_id, column:l.column, cell_text:l.cell_text} END) AS links
                ORDER BY row_index""",
                    timeout=30,
                ),
                table_id=table_id,
            ).data()
        rows = [
            {
                "table_id": table_id,
                **r,
                "passage_ids": sorted({p["passage_id"] for p in r["links"]}),
            }
            for r in records
        ]
        return {
            "rows": rows,
            "covered": bool(rows),
            "seconds": perf_counter() - start,
            "queries": 1,
        }

    def passages(self, ids):
        start = perf_counter()
        encoded = {quote(pid, safe="/_-.,()'!:"): pid for pid in sorted(set(ids))}
        keys = list(encoded)
        passages = {}
        read_units, calls = 0, 0
        for offset in range(0, len(keys), 500):
            response = self.index.fetch(
                ids=keys[offset : offset + 500], namespace="", _request_timeout=(10, 30)
            )
            calls += 1
            read_units += (response.usage or {}).get("read_units", 0)
            for key, record in response.vectors.items():
                meta = record.metadata
                pid = meta["passage_id"]
                if pid != encoded[key]:
                    raise ValueError("Pinecone passage identifier mismatch")
                passages[pid] = {
                    "title": meta["title"],
                    "text": meta["text"],
                    "values": list(record.values),
                }
        return {
            "records": passages,
            "missing_ids": sorted(set(ids) - set(passages)),
            "seconds": perf_counter() - start,
            "fetch_calls": calls,
            "read_units": read_units,
        }

    def table_bundle(self, table_id):
        start = perf_counter()
        graph_future = self.pool.submit(self.graph_table, table_id)
        sql = self.sql_table(table_id)
        ids = set(sql["header_passage_ids"]) | {
            p for row in sql["rows"] for p in row["passage_ids"]
        }
        passages = self.passages(ids)
        graph = graph_future.result()

        def topology(rows):
            return {r["row_index"]: set(r["passage_ids"]) for r in rows}

        equivalent = (
            topology(sql["rows"]) == topology(graph["rows"])
            if graph["covered"]
            else None
        )
        return {
            "table_id": table_id,
            "sql": sql,
            "graph": graph,
            "passages": passages,
            "graph_sql_topology_equal": equivalent,
            "seconds": perf_counter() - start,
        }
