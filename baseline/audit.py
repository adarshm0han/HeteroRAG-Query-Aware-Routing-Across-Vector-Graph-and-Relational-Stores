"""Independent source audit. Source data is never used to repair retrieval results."""

import json
import zipfile

from .common import CORPUS_REVISION


class SourceAudit:
    def __init__(self, path):
        self.archive = zipfile.ZipFile(path)
        self.prefix = f"WikiTables-WithLinks-{CORPUS_REVISION}/"

    def close(self):
        self.archive.close()

    def check(self, bundle):
        tid = bundle["table_id"]
        table = json.loads(
            self.archive.read(self.prefix + "tables_tok/" + tid + ".json")
        )
        passages = json.loads(
            self.archive.read(self.prefix + "request_tok/" + tid + ".json")
        )
        rows = bundle["sql"]["rows"]
        if bundle["sql"]["title"] != table.get("title", ""):
            raise ValueError("SQL/source table title mismatch")
        if len(rows) != len(table["data"]):
            raise ValueError("SQL/source row count mismatch")
        cell_count = 0
        for index, (row, expected) in enumerate(zip(rows, table["data"])):
            if row["row_index"] != index or len(row["cells"]) != len(expected):
                raise ValueError("SQL/source row position mismatch")
            expected_text = " | ".join(
                f"{table['header'][col][0]}: {cell[0]}"
                for col, cell in enumerate(expected)
            )
            if row["text"] != expected_text:
                raise ValueError("SQL/source rendered row text mismatch")
            for col, (cell, (value, links)) in enumerate(zip(row["cells"], expected)):
                if (
                    cell["column_index"],
                    cell["header"],
                    cell["value"],
                    cell["entity_ids"],
                ) != (col, table["header"][col][0], value, links):
                    raise ValueError("SQL/source cell or link mismatch")
                cell_count += 1
        expected_ids = {p for row in table["data"] for _, links in row for p in links}
        expected_ids |= {p for _, links in table["header"] for p in links}
        missing_nonempty = [
            pid
            for pid in bundle["passages"]["missing_ids"]
            if passages.get(pid, "").strip()
        ]
        if missing_nonempty:
            raise ValueError("Nonempty linked passage missing from Pinecone")
        for pid, passage in bundle["passages"]["records"].items():
            if (
                pid not in expected_ids
                or passage["text"].strip() != passages[pid].strip()
            ):
                raise ValueError("Pinecone/source text mismatch")
            if passage["title"] != pid.removeprefix("/wiki/").replace("_", " "):
                raise ValueError("Pinecone/source passage title mismatch")
        for graph_row in bundle.get("graph", {}).get("rows", []):
            index = graph_row["row_index"]
            if not 0 <= index < len(rows):
                raise ValueError("Graph row is absent from source")
            valid = {
                (link["column"], link["cell_text"], link["passage_id"])
                for link in rows[index]["links"]
            }
            if any(
                (link["column"], link["cell_text"], link["passage_id"]) not in valid
                for link in graph_row["links"]
            ):
                raise ValueError("Graph relationship differs from source cells")
        return {
            "passed": True,
            "rows": len(rows),
            "cells": cell_count,
            "passages": len(bundle["passages"]["records"]),
            "empty_passages_not_indexed": len(bundle["passages"]["missing_ids"]),
        }
