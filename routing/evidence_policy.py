"""Conservative schema certificates for omitting passage retrieval.

A missing certificate is uncertainty, not proof that a passage is necessary.
Only complete, narrowly parsed table questions can omit passage access. This
policy runs before opening stores and never uses cells, references or an LLM.
"""

import re
from copy import deepcopy

from .catalog import validate_metadata
from .options import OPTIONS

VERSION = "schema_certificate_guard_v1"
PROBABILITY_METHOD = "required_store_pushforward_v1"


def normalized(text):
    return " ".join(text.casefold().split()).strip().removesuffix("?").strip()


def table_certificate(question, metadata):
    """Recognize entire supported questions, never just their requested column.

    Free-form text conditions are accepted only as quoted literals following an
    explicit schema column. Unknown clauses and unquoted descriptions fail closed.
    This intentionally prefers extra passage access over semantic guesswork.
    """
    if metadata is None:
        return {"certified": False, "reason": "schema_unavailable"}
    validate_metadata(metadata)
    if metadata["schema_truncated"]:
        return {"certified": False, "reason": "schema_truncated"}
    columns = {normalized(c): c for c in metadata["columns"] if normalized(c)}
    if not columns or len(columns) != len(
        [c for c in metadata["columns"] if c.strip()]
    ):
        return {"certified": False, "reason": "schema_ambiguous"}
    text = normalized(question)
    alternatives = (
        "(?:"
        + "|".join(re.escape(c) for c in sorted(columns, key=len, reverse=True))
        + ")"
    )

    def column(name):
        return "(?P<" + name + ">" + alternatives + ")"

    # Quotes delimit a literal cell value, not an unparsed natural-language
    # condition. Numeric values are the only unquoted literals admitted.
    literal = r"""(?P<literal>"[^"\n]{1,160}"|'[^'\n]{1,160}'|[+-]?\d+(?:,\d{3})*(?:\.\d+)?)"""
    scope = r"(?: in (?:this|the) table)?"
    patterns = [
        (
            "count_rows",
            r"how many (?:rows|records|entries) are (?:there )?in (?:this|the) table",
        ),
        (
            "count_rows",
            r"what is the (?:count|number) of (?:rows|records|entries)" + scope,
        ),
        (
            "aggregate",
            r"what is (?:the )?(?P<aggregate>sum|total|average|mean|minimum|maximum) (?:of (?:the )?)?"
            + column("value_column")
            + scope,
        ),
        (
            "extremum_row",
            r"(?:which|what) "
            + column("answer_column")
            + r" (?:has|had) the (?P<order>highest|lowest|largest|smallest) "
            + column("order_column")
            + scope,
        ),
        (
            "lookup",
            r"what (?:is|was|are|were) (?:the )?"
            + column("answer_column")
            + r" where (?:the )?"
            + column("filter_column")
            + r" (?:is|equals|=) "
            + literal,
        ),
        (
            "filtered_count",
            r"how many (?:rows|records|entries) (?:have|where) (?:the )?"
            + column("filter_column")
            + r" (?:is|equals|=) "
            + literal,
        ),
    ]
    for operation, pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match is None:
            continue
        parsed = match.groupdict()
        used = {
            name: columns[value]
            for name, value in parsed.items()
            if name.endswith("_column")
        }
        return {
            "certified": True,
            "reason": "whole_question_matches_table_grammar",
            "operation": operation,
            "columns": used,
            "scope": "provided_table",
            "literal": parsed.get("literal"),
        }
    return {"certified": False, "reason": "whole_question_not_certified"}


def required_option(option, required):
    stores = set(OPTIONS[option]) | set(required)
    return next(name for name, members in OPTIONS.items() if set(members) == stores)


def guard_decision(decision, question, metadata):
    """Apply the same minimum-store constraint to the choice and its distribution.

    The raw API result remains available for audit. Projecting probabilities
    preserves total mass and gives fusion weights to every executed store; the
    result is a routing preference, not a calibrated correctness probability.
    """
    certificate = table_certificate(question["question"], metadata)
    required = ["postgres"] if certificate["certified"] else ["pinecone"]
    result = deepcopy(decision)
    raw_option = result.get("raw_option", result["option"])
    raw_probabilities = result.get("raw_probabilities", result["probabilities"])
    effective = required_option(raw_option, required)
    mass = {}
    for item in raw_probabilities:
        target = required_option(item["value"], required)
        mass[target] = mass.get(target, 0.0) + item["probability"]
    result.update(
        raw_option=raw_option,
        raw_probabilities=deepcopy(raw_probabilities),
        raw_confidence=result.get("raw_confidence", result["confidence"]),
        confidence_scope="raw_decisions_choice",
        option=effective,
        probabilities=[
            {"value": name, "probability": mass[name]}
            for name in OPTIONS
            if name in mass
        ],
        probability_method=PROBABILITY_METHOD,
        evidence_policy={
            "version": VERSION,
            "certificate": certificate,
            "required_stores": required,
            "added_stores": [
                s for s in OPTIONS[effective] if s not in OPTIONS[raw_option]
            ],
            "changed": effective != raw_option,
            "stage": "before_retrieval",
            "uncertainty_policy": "include_passages_unless_whole_question_is_certified_table_only",
        },
    )
    return result
