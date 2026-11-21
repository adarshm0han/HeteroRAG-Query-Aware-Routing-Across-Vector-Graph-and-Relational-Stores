"""Independently recompute SQL aggregate evidence from the pinned raw table."""

import re
from decimal import Decimal

from .sql_plan import aggregate_text, validate_plan


def numeric(value):
    value = value.strip()
    if not re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", value):
        return None
    return Decimal(value.replace(",", ""))


def compare(left, op, right):
    if left is None:
        return False
    return {
        "eq": lambda: left == right,
        "ne": lambda: left != right,
        "gt": lambda: left > right,
        "gte": lambda: left >= right,
        "lt": lambda: left < right,
        "lte": lambda: left <= right,
        "contains": lambda: right in left,
    }[op]()


def check_aggregate(table_id, table, item):
    columns = [
        {"column_index": i, "header": h[0]} for i, h in enumerate(table["header"])
    ]
    plan = validate_plan(item["plan"], columns)
    if item["table_id"] != table_id or plan["operation"] in ("select", "fallback"):
        raise ValueError("Invalid computed evidence scope")
    rows = []
    for row in table["data"]:
        if all(
            compare(
                numeric(row[f["column"]][0])
                if f["type"] == "number"
                else row[f["column"]][0].lower(),
                f["op"],
                Decimal(f["value"]) if f["type"] == "number" else f["value"].lower(),
            )
            for f in plan["filters"]
        ):
            rows.append(row)
    result = item["result"]
    if set(result) != {"group_value", "value", "source_row_count"}:
        raise ValueError("Invalid aggregate evidence fields")
    if plan["group_by"] is not None:
        rows = [
            r for r in rows if r[plan["group_by"]][0].lower() == result["group_value"]
        ]
        if not rows:
            raise ValueError("Aggregate group absent from source")
    elif result["group_value"] is not None:
        raise ValueError("Unexpected aggregate group")
    op, col = plan["operation"], plan["column"]
    if op == "count":
        expected = Decimal(len(rows))
    elif op == "count_distinct":
        expected = Decimal(len({r[col][0].lower() for r in rows if r[col][0]}))
    else:
        values = [v for r in rows if (v := numeric(r[col][0])) is not None]
        expected = (
            {
                "sum": lambda: sum(values),
                "avg": lambda: sum(values) / len(values),
                "min": lambda: min(values),
                "max": lambda: max(values),
            }[op]()
            if values
            else None
        )
    actual = None if result["value"] is None else Decimal(result["value"])
    if (expected is None) != (actual is None) or (
        expected is not None and abs(expected - actual) > Decimal("1e-12")
    ):
        raise ValueError("SQL aggregate differs from source")
    if result["source_row_count"] != len(rows):
        raise ValueError("SQL aggregate source count differs")
    if plan["having"] and not compare(
        expected, plan["having"]["op"], Decimal(str(plan["having"]["value"]))
    ):
        raise ValueError("SQL aggregate violates its HAVING condition")
    if item["text"] != aggregate_text(plan, result, columns):
        raise ValueError("SQL aggregate text differs from computed value")


def check_paths(table, item):
    for path in item.get("graph_paths", []):
        seed, row = path["seed_row_index"], path["row_index"]
        if (
            type(seed) is not int
            or not 0 <= seed < len(table["data"])
            or row != item["row_index"]
        ):
            raise ValueError("Graph path points outside the source table")
        bridge = path["bridge_passage_id"]
        if bridge is None:
            if seed != row:
                raise ValueError("Unconnected graph path")
        else:
            for index in (seed, row):
                if not any(bridge in c[1] for c in table["data"][index]):
                    raise ValueError("Graph bridge is absent from source row links")
