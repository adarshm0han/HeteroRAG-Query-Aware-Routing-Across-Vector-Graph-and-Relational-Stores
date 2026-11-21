"""Public table schema and graph availability, prepared without question labels.

This catalog contains no cells, passages, questions or answers. Its one local lookup
is measured separately from database retrieval and is available to every condition.
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from baseline.common import (
    CORPUS_REVISION,
    REPO,
    object_hash,
    read_json,
    sha256,
    write_json,
)

VERSION = "table_schema_catalog_v1"


def table_metadata(table, graph_available=None):
    headers = [str(header[0]) for header in table["header"]]
    return {
        "title": str(table.get("title", ""))[:300],
        "columns": [header[:120] for header in headers[:64]],
        "schema_truncated": len(headers) > 64 or any(len(h) > 120 for h in headers),
        "row_count": len(table["data"]),
        "graph_available": graph_available,
    }


def validate_metadata(value):
    if not isinstance(value, dict) or set(value) != {
        "title",
        "columns",
        "schema_truncated",
        "row_count",
        "graph_available",
    }:
        raise ValueError("Routing metadata must contain public schema fields only")
    if (
        not isinstance(value["title"], str)
        or len(value["title"]) > 300
        or not isinstance(value["columns"], list)
        or not 1 <= len(value["columns"]) <= 64
        or any(not isinstance(c, str) or len(c) > 120 for c in value["columns"])
        or type(value["schema_truncated"]) is not bool
        or type(value["row_count"]) is not int
        or value["row_count"] < 0
        or (
            value["graph_available"] is not None
            and type(value["graph_available"]) is not bool
        )
    ):
        raise ValueError("Invalid public table metadata")
    return value


def prepare(workspace, output, graph_catalog=None):
    if output.exists():
        raise ValueError("Use a new catalog file; existing catalogs are immutable")
    archive = workspace / "data/corpus.zip"
    lock = read_json(REPO / "postgres/sources.lock.json")
    digest = sha256(archive)
    if digest != lock["corpus"]["archive_sha256"]:
        raise ValueError("Routing catalog requires the pinned source corpus")
    covered = set(read_json(graph_catalog)) if graph_catalog else None
    tables = {}
    prefix = f"WikiTables-WithLinks-{CORPUS_REVISION}/tables_tok/"
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            if name.startswith(prefix) and name.endswith(".json"):
                table_id = name[len(prefix) : -5]
                tables[table_id] = validate_metadata(
                    table_metadata(
                        json.loads(source.read(name)),
                        table_id in covered if covered is not None else None,
                    )
                )
    if covered is not None and not covered <= set(tables):
        raise ValueError("Graph catalog contains tables outside the pinned corpus")
    catalog = {
        "version": VERSION,
        "corpus_sha256": digest,
        "graph_catalog_sha256": sha256(graph_catalog) if graph_catalog else None,
        "graph_table_ids_sha256": hashlib.sha256(
            "\n".join(sorted(covered)).encode()
        ).hexdigest()
        if covered is not None
        else None,
        "tables": tables,
    }
    catalog["catalog_id"] = object_hash(catalog)
    write_json(output, catalog)
    return catalog


class RoutingCatalog:
    def __init__(self, path):
        self.path = Path(path)
        self.digest = sha256(self.path)
        self.data = read_json(self.path)
        if self.data.get("version") != VERSION or object_hash(
            {k: v for k, v in self.data.items() if k != "catalog_id"}
        ) != self.data.get("catalog_id"):
            raise ValueError("Routing catalog checksum or version mismatch")
        lock = read_json(REPO / "postgres/sources.lock.json")
        if self.data["corpus_sha256"] != lock["corpus"]["archive_sha256"]:
            raise ValueError("Routing catalog uses a different corpus")

    def describe(self, table_id):
        if table_id not in self.data["tables"]:
            raise ValueError("Supplied table is missing from routing metadata")
        return validate_metadata(self.data["tables"][table_id]).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=REPO.parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-catalog", type=Path)
    args = parser.parse_args()
    result = prepare(args.workspace, args.output, args.graph_catalog)
    print(
        json.dumps(
            {
                "catalog_id": result["catalog_id"],
                "tables": len(result["tables"]),
                "question_labels_used": False,
                "database_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
