"""Read-only, lazy database adapters with enforced store boundaries and call logs."""

import asyncio
import json
from contextlib import contextmanager
from time import perf_counter
from urllib.parse import quote

from baseline.common import safe_error

from .options import OPTIONS
from .sql_plan import aggregate_text, compile_plan, fallback_plan, plan_guard


def response_bytes_estimate(response):
    """Serialized SDK payload size, not an HTTP capture or billing measurement."""
    if hasattr(response, "to_dict"):
        value = response.to_dict()
    else:

        def record(item):
            return {
                "id": getattr(item, "id", ""),
                "values": list(item.values),
                "metadata": item.metadata,
                "score": getattr(item, "score", None),
            }

        usage = response.usage
        value = {"usage": usage.to_dict() if hasattr(usage, "to_dict") else usage}
        if hasattr(response, "matches"):
            value["matches"] = [record(item) for item in response.matches]
        if hasattr(response, "vectors"):
            value["vectors"] = {
                key: record(item) for key, item in response.vectors.items()
            }
    return len(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode())


class LiveStores:
    def __init__(self, cfg, allowed, sql_planner=None, journal=None):
        if not set(allowed) or not set(allowed) <= set(OPTIONS["all"]):
            raise ValueError("Invalid database allowlist")
        self.cfg = cfg
        self.allowed = frozenset(allowed)
        self.events = []
        self.journal = journal
        self.graph = self.index = self.pc = None
        self.sql_planner = sql_planner

    def require(self, store):
        if store not in self.allowed:
            raise PermissionError(
                "Retrieval tried to access an unselected database: " + store
            )

    @contextmanager
    def operation(self, store, operation, kind="data", **details):
        self.require(store)
        event = {"store": store, "operation": operation, "kind": kind, **details}
        self.events.append(event)
        event["status"] = "started"
        if self.journal is not None:
            self.journal(self.events)
        start = perf_counter()
        try:
            yield event
            event["status"] = "completed"
        except Exception as exc:
            event.update(status="failed", error=safe_error(exc))
            raise
        finally:
            event["seconds"] = perf_counter() - start
            if self.journal is not None:
                self.journal(self.events)

    def sql_table(self, table_id):
        self.require("postgres")
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            records = runner.run(self._sql_records(table_id))
        rows = []
        for record in records:
            links = [
                {
                    "column_index": cell["column_index"],
                    "column": cell["header"],
                    "cell_text": cell["value"],
                    "passage_id": pid,
                }
                for cell in record["cells"]
                for pid in cell["entity_ids"]
            ]
            rows.append(
                {
                    "table_id": table_id,
                    "row_index": record["row_index"],
                    "text": record["evidence_text"],
                    "cells": record["cells"],
                    "links": links,
                    "passage_ids": sorted({link["passage_id"] for link in links}),
                }
            )
        return {"rows": rows, "title": records[0]["title"] if records else ""}

    def sql_retrieve(self, table_id, question, limit):
        self.require("postgres")
        if self.sql_planner is None:
            raise ValueError("Structured SQL retrieval requires a local planner")
        schema_rows = self.sql_query(
            """SELECT t.title, c.column_index, c.header,
                ARRAY(SELECT value_text FROM hybridqa.wiki_cells v
                  WHERE v.table_pk=t.id AND v.column_index=c.column_index
                  ORDER BY row_index LIMIT 3) AS examples
                FROM hybridqa.wiki_tables t JOIN hybridqa.wiki_columns c ON c.table_pk=t.id
                WHERE t.table_id=%s ORDER BY c.column_index""",
            (table_id,),
            "select_schema",
        )
        schema = {
            "title": schema_rows[0]["title"] if schema_rows else "",
            "columns": [
                {k: r[k] for k in ("column_index", "header", "examples")}
                for r in schema_rows
            ],
        }
        if not schema_rows:
            return {
                "rows": [],
                "title": "",
                "ordered": True,
                "sql": {"status": "missing_table"},
            }
        plan, metadata = self.sql_planner.plan(question, schema)
        # Validate again at the execution boundary, including fallback plans.
        from .sql_plan import validate_plan

        validate_plan(plan, schema["columns"])
        guard = plan_guard(plan, schema["columns"])
        if guard:
            metadata = {**metadata, "proposed_plan": plan, "fallback_reason": guard}
            plan = fallback_plan()
        if plan["operation"] == "fallback":
            table = self.sql_table(table_id)
            table["selection_grounded"] = False
            table["sql"] = {
                "plan": plan,
                "planner": metadata,
                "status": "lexical_fallback",
                "candidate_rows": len(table["rows"]),
            }
            return table
        # Unordered matching rows must be ranked for relevance before the evidence
        # limit, rather than truncating by row_index inside SQL.
        sql_limit = (
            None if plan["operation"] == "select" and not plan["order_by"] else limit
        )
        query, params = compile_plan(table_id, plan, schema["columns"], sql_limit)
        records = self.sql_query(query, params, "execute_plan")
        trace = {
            "plan": plan,
            "planner": metadata,
            "status": "executed",
            "statement": query,
            "parameters": list(params),
        }
        table = {
            "rows": [],
            "title": schema["title"],
            "ordered": True,
            "selection_grounded": True,
            "sql": trace,
        }
        if plan["operation"] == "select":
            if not records:
                # Zero is meaningful for an aggregate. An empty entity selection
                # instead triggers an explicitly labelled retrieval fallback.
                table = self.sql_table(table_id)
                table.update(
                    selection_grounded=False,
                    sql={
                        **trace,
                        "status": "empty_selection_fallback",
                        "result_rows": 0,
                        "candidate_rows": len(table["rows"]),
                    },
                )
                return table
            table["rows"] = [self.sql_row(table_id, r) for r in records]
            table["ordered"] = plan["order_by"] is not None
            trace["result_rows"] = records[0]["result_rows"] if records else 0
            trace["truncated"] = trace["result_rows"] > limit
        else:
            trace["result_groups"] = records[0]["result_groups"] if records else 0
            trace["truncated"] = trace["result_groups"] > limit
            for record in records:
                record.pop("result_groups")
            table["aggregates"] = [
                {
                    "id": f"sql:{table_id}:{i}",
                    "kind": "aggregate",
                    "table_id": table_id,
                    "text": aggregate_text(plan, r, schema["columns"]),
                    "plan": plan,
                    "result": r,
                    "sources": ["postgres"],
                }
                for i, r in enumerate(records)
            ]
        return table

    @staticmethod
    def sql_row(table_id, record):
        links = [
            {
                "column_index": c["column_index"],
                "column": c["header"],
                "cell_text": c["value"],
                "passage_id": pid,
            }
            for c in record["cells"]
            for pid in c["entity_ids"]
        ]
        return {
            "table_id": table_id,
            "row_index": record["row_index"],
            "text": record["evidence_text"],
            "cells": record["cells"],
            "links": links,
            "passage_ids": sorted({x["passage_id"] for x in links}),
        }

    def sql_query(self, query, params, operation):
        self.require("postgres")
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            return runner.run(self._sql_query(query, params, operation))

    async def _sql_records(self, table_id):
        return await self._sql_query(
            "SELECT title, row_index, evidence_text, cells FROM hybridqa.rag_rows WHERE table_id = %s ORDER BY row_index",
            (table_id,),
            "select_table",
        )

    async def _sql_query(self, query, params, operation):
        import psycopg
        from psycopg.rows import dict_row

        with self.operation("postgres", "connect", kind="connection"):
            connection = await psycopg.AsyncConnection.connect(
                self.cfg["DATABASE_URL"],
                autocommit=True,
                connect_timeout=15,
                row_factory=dict_row,
            )

        async def select():
            # PgBouncer rejects statement_timeout in startup options. Keep both
            # protections transaction-local so they also work with pooled URLs.
            with self.operation("postgres", "begin_read_only", kind="control"):
                await connection.execute("BEGIN READ ONLY")
            with self.operation("postgres", "set_timeout", kind="control"):
                await connection.execute("SET LOCAL statement_timeout = '30s'")
            with self.operation("postgres", operation):
                cursor = await connection.execute(query, params)
                records = await cursor.fetchall()
            with self.operation("postgres", "rollback", kind="control"):
                await connection.execute("ROLLBACK")
            return records

        try:
            task = asyncio.create_task(select())
            done, _ = await asyncio.wait({task}, timeout=40)
            if not done:
                # Closing first prevents cancellation/rollback hanging on a dead socket.
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

    def graph_table(self, table_id, anchors=(), *, question="", seed_rows=(), limit=8):
        self.require("neo4j")
        from neo4j import READ_ACCESS, GraphDatabase, Query

        from .graph_query import (
            EXPAND,
            SEEDS,
            SQL_SEEDS,
            combine_seeds,
            merge_paths,
            query_terms,
        )

        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Invalid graph result limit")
        if any(type(i) is not int or i < 0 for i in seed_rows):
            raise ValueError("Invalid graph seed rows")
        if self.graph is None:
            self.graph = GraphDatabase.driver(
                self.cfg["NEO4J_URI"],
                auth=(self.cfg["NEO4J_USERNAME"], self.cfg["NEO4J_PASSWORD"]),
                connection_timeout=15,
                connection_acquisition_timeout=30,
                max_transaction_retry_time=0,
            )
        # Reserve space for expansion, instead of filling the whole budget with seeds.
        seed_limit = min(max(1, limit // 2), 4)
        sql_seed_limit = min(2, max(0, limit - seed_limit - 1))
        params = {
            "table_id": table_id,
            "anchors": list(anchors)[:24],
            "terms": query_terms(question),
            "seed_rows": list(seed_rows)[:limit],
            "seed_limit": seed_limit,
            "link_limit": 64,
        }
        with self.graph.session(
            database=self.cfg.get("NEO4J_DATABASE", "neo4j"),
            default_access_mode=READ_ACCESS,
        ) as session:
            with self.operation(
                "neo4j",
                "select_seeds",
                seed_row_count=len(seed_rows),
                anchor_count=len(params["anchors"]),
            ):
                seed_records = session.run(Query(SEEDS, timeout=30), **params).data()
                seeds = [r for r in seed_records if r["row_index"] is not None]
                independent_rows = [r["row_index"] for r in seeds]
            if seed_rows and sql_seed_limit:
                with self.operation(
                    "neo4j", "select_sql_seeds", seed_row_count=len(seed_rows)
                ):
                    sql_seeds = session.run(
                        Query(SQL_SEEDS, timeout=30),
                        **{**params, "seed_limit": sql_seed_limit},
                    ).data()
                seeds = combine_seeds(seeds, sql_seeds)
            neighbors = []
            if seeds:
                with self.operation(
                    "neo4j", "expand_shared_passages", seed_count=len(seeds)
                ):
                    neighbors = session.run(
                        Query(EXPAND, timeout=30),
                        table_id=table_id,
                        seeds=[r["row_index"] for r in seeds],
                        neighbor_limit=8,
                        link_limit=64,
                    ).data()
        rows, truncated = merge_paths(table_id, seeds, neighbors, limit)
        return {
            "rows": rows,
            "title": (seed_records[0]["title"] or "") if seed_records else "",
            "covered": bool(seed_records),
            "ordered": True,
            "graph": {
                "seed_rows": [r["row_index"] for r in seeds],
                "independent_seed_rows": independent_rows,
                "input_seed_rows": list(seed_rows),
                "anchors": params["anchors"],
                "max_relationship_hops": 3,
                "neighbors_per_seed": 8,
                "links_per_row": 64,
                "rows_truncated": truncated,
                "links_truncated": any(r.get("links_truncated", False) for r in rows),
            },
        }

    def _pinecone(self):
        self.require("pinecone")
        if self.index is None:
            from pinecone import Pinecone

            self.pc = Pinecone(api_key=self.cfg["PINECONE_API_KEY"], pool_threads=1)
            # Pinecone 7.3.0 has no public retry setting. Set it before either HTTP
            # client is created; otherwise one logged SDK request may hide retries.
            self.pc._openapi_config.retries = 0
            with self.operation("pinecone", "describe_index", kind="control"):
                # The high-level 7.3.0 describe method omits timeout parameters.
                info = self.pc.db.index._index_api.describe_index(
                    index_name=self.cfg.get("PINECONE_INDEX", "hybridqa-passages"),
                    _request_timeout=(10, 30),
                )
            if info.dimension != 384 or info.metric != "cosine":
                raise ValueError("Unexpected Pinecone dimension or metric")
            self.index = self.pc.Index(host=info.host)
        return self.index

    @staticmethod
    def _passage(record):
        meta = record.metadata
        if not meta.get("passage_id") or not isinstance(meta.get("text"), str):
            raise ValueError("Invalid Pinecone passage metadata")
        return {
            "title": meta["title"],
            "text": meta["text"],
            "values": list(record.values),
        }

    def search(self, table_id, vector, k):
        index = self._pinecone()
        with self.operation("pinecone", "query", top_k=k) as event:
            response = index.query(
                vector=list(map(float, vector)),
                top_k=k,
                filter={"table_ids": {"$in": [table_id]}},
                include_metadata=True,
                include_values=True,
                namespace="",
                _request_timeout=(10, 30),
            )
            event["read_units"] = (response.usage or {}).get("read_units", 0)
            event["response_bytes_estimate"] = response_bytes_estimate(response)
            event["returned_records"] = len(response.matches)
        return {
            record.metadata["passage_id"]: self._passage(record)
            for record in response.matches
        }

    def passages(self, ids):
        self.require("pinecone")
        if not ids:
            return {"records": {}, "missing_ids": []}
        index = self._pinecone()
        encoded = {quote(pid, safe="/_-.,()'!:"): pid for pid in sorted(set(ids))}
        keys, passages = list(encoded), {}
        for offset in range(0, len(keys), 500):
            batch = keys[offset : offset + 500]
            with self.operation("pinecone", "fetch", id_count=len(batch)) as event:
                response = index.fetch(
                    ids=batch, namespace="", _request_timeout=(10, 30)
                )
                event["read_units"] = (response.usage or {}).get("read_units", 0)
                event["response_bytes_estimate"] = response_bytes_estimate(response)
                event["returned_records"] = len(response.vectors)
            for key, record in response.vectors.items():
                if record.metadata["passage_id"] != encoded[key]:
                    raise ValueError("Pinecone identifier mismatch")
                passages[encoded[key]] = self._passage(record)
        return {"records": passages, "missing_ids": sorted(set(ids) - set(passages))}

    def close(self):
        if self.graph is not None:
            self.graph.close()
        if self.index is not None:
            self.index.close()
        if self.pc is not None:
            self.pc.db.index._index_api.api_client.close()
