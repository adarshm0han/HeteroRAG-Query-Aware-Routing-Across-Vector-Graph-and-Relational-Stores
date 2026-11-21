"""Source quotation and explicit-property validation for answer evidence."""

import re


def resolve_quote(quote, text):
    """Verify source tokens and expand explicit ellipses to an actual source span.

    Dataset tokenization inserts spaces around punctuation. Ignore those spaces,
    but never remove, paraphrase or reorder content words from a quotation.
    """
    matches = list(re.finditer(r"\w+|[^\w\s]", text))
    tokens = [m.group().casefold() for m in matches]
    cursor, first = 0, None
    for fragment in re.split(r"\.{3}|…", quote):
        target = [m.group().casefold() for m in re.finditer(r"\w+|[^\w\s]", fragment)]
        if not target:
            continue
        found = next(
            (
                i
                for i in range(cursor, len(tokens) - len(target) + 1)
                if tokens[i : i + len(target)] == target
            ),
            None,
        )
        if found is None:
            raise ValueError("support_quote_not_in_evidence")
        if first is None:
            first = found
        cursor = found + len(target)
    if first is None:
        raise ValueError("empty_support_quote")
    return text[matches[first].start() : matches[cursor - 1].end()]


def restore_requested_unit(answer, question):
    """Complete a numeric span only when both question and verified source agree."""
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", answer["answer"]):
        return answer
    unit = re.search(
        r"\bin (feet|inches|meters|metres|kilometers|kilometres|miles|centimeters|centimetres|kg|cm|mm)\b",
        question,
        re.IGNORECASE,
    )
    if unit:
        candidate = answer["answer"] + " " + unit[1].lower()
        for support in answer.get("support_quotes", []):
            try:
                resolve_quote(candidate, support["quote"])
                return {**answer, "answer": candidate, "unit_restored": True}
            except ValueError:
                pass
    return answer


def check_explicit_property(question, answer, evidence):
    """Reject an intermediate entity when a cited passage names the asked property.

    These conservative checks apply only to explicit linguistic relations; they
    never consult reference labels or supply an answer when the reader abstains.
    """
    if answer["abstain"]:
        return
    source = "\n".join(
        item.get("text", "") for item in evidence if item["id"] in answer["citations"]
    )

    def words(value):
        return [
            w
            for w in re.findall(r"\w+", value.casefold())
            if w not in ("a", "an", "the")
        ]

    result = words(answer["answer"])
    if re.search(
        r"\bname\b.*\b(?:founded|formed|established)\s+as\b", question, re.IGNORECASE
    ):
        clauses = re.findall(
            r"\b(?:founded|formed|established)\s+as\s+(.+?)(?:\s+in\b|[,.;\n]|$)",
            source,
            re.IGNORECASE,
        )
        if not clauses:
            raise ValueError("missing_explicit_founding_name_evidence")
        if not any(result == words(clause) for clause in clauses):
            raise ValueError("answer_must_match_explicit_founding_name_clause")
    if re.search(r"\bquantity\b.*\bunit\b", question, re.IGNORECASE):
        clauses = re.findall(r"\bunit\s+of\s+([^,.;\n]+)", source, re.IGNORECASE)
        if clauses and not any(
            result and words(clause)[: len(result)] == result for clause in clauses
        ):
            raise ValueError("answer_must_be_measured_quantity_not_unit_name")
