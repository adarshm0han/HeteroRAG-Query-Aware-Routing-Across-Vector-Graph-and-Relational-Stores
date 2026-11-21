"""Structured question context and numbered source excerpts."""

import json
import re


def obj(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def frame_context(context, evidence):
    try:
        header = json.loads(context.split("\nEvidence:", 1)[0])
    except (ValueError, TypeError):
        header = {"question": context, "table_title": ""}
    if not isinstance(header, dict) or not isinstance(header.get("question"), str):
        raise TypeError("Invalid question context")
    columns, rows, excerpts = {}, {}, {}
    passage_ids = {
        item.get("passage_id", item.get("source_id")): item["id"]
        for item in evidence
        if item.get("kind") == "passage"
    }
    for item in evidence:
        if item.get("kind") == "row":
            cells = {
                part.partition(": ")[0]: part.partition(": ")[2]
                for part in item["text"].split(" | ")
                if ": " in part
            }
            rows[item["id"]] = cells
            for name, value in cells.items():
                if value not in columns.setdefault(name, []):
                    columns[name].append(value)
        chunks = [item.get("text", "")]
        if item.get("kind") == "passage":
            chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", item["text"])
        for i, chunk in enumerate(chunks, 1):
            key = f"{item['id']}.S{i}" if item.get("kind") == "passage" else item["id"]
            if chunk.strip():
                excerpts[key] = {
                    "id": key,
                    "evidence_id": item["id"],
                    "kind": item.get("kind", "text"),
                    "text": chunk.strip(),
                    "title": item.get("title", ""),
                }
                if item.get("kind") == "row":
                    excerpts[key]["linked_evidence"] = sorted(
                        {
                            passage_ids[link["passage_id"]]
                            for link in item.get("links", [])
                            if link.get("passage_id") in passage_ids
                        }
                    )
    return header, columns, rows, excerpts
