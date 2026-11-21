"""Optional verification against the pinned source corpus, never retrieval repair."""

import json
import zipfile

from baseline.common import CORPUS_REVISION

from .sql_audit import check_aggregate, check_paths


class EvidenceAudit:
    def __init__(self, archive):
        self.archive = zipfile.ZipFile(archive)
        self.prefix = f"WikiTables-WithLinks-{CORPUS_REVISION}/"

    def check(self, table_id, items):
        table = json.loads(
            self.archive.read(self.prefix + "tables_tok/" + table_id + ".json")
        )
        passages = json.loads(
            self.archive.read(self.prefix + "request_tok/" + table_id + ".json")
        )
        checked = {"rows": 0, "passages": 0}
        for item in items:
            if item["kind"] == "aggregate":
                check_aggregate(table_id, table, item)
                checked["aggregates"] = checked.get("aggregates", 0) + 1
            elif item["kind"] == "row":
                index = item["row_index"]
                if item["table_id"] != table_id or not 0 <= index < len(table["data"]):
                    raise ValueError(
                        "Evidence row is outside the supplied source table"
                    )
                cells = table["data"][index]
                expected = " | ".join(
                    f"{table['header'][column][0]}: {cell[0]}"
                    for column, cell in enumerate(cells)
                )
                if item["text"] != expected:
                    raise ValueError("Row text differs from the source")
                valid_links = {
                    (table["header"][column][0], cell[0], pid)
                    for column, cell in enumerate(cells)
                    for pid in cell[1]
                }
                if any(
                    (link["column"], link["cell_text"], link["passage_id"])
                    not in valid_links
                    for link in item["links"]
                ):
                    raise ValueError("Row link differs from the source")
                checked["rows"] += 1
                check_paths(table, item)
            elif item["kind"] == "passage":
                pid = item["passage_id"]
                if pid not in passages or item["text"].strip() != passages[pid].strip():
                    raise ValueError("Passage text differs from the source")
                checked["passages"] += 1
            else:
                raise ValueError("Unknown evidence kind")
        return {"passed": True, **checked}

    def close(self):
        self.archive.close()
