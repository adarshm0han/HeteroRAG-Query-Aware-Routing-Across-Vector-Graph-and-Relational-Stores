"""Conservative checks for explicit answer types and source number boundaries."""

import re
from decimal import Decimal

from .answer_validation import resolve_quote


def words(value):
    return " ".join(re.findall(r"\w+", value.casefold()))


def cells(sources):
    return [
        (part.partition(": ")[0], part.partition(": ")[2])
        for source in sources
        if source["kind"] == "row"
        for part in source["text"].split(" | ")
        if ": " in part
    ]


def check_answer_kind(answer, question, sources):
    interrogative = re.search(
        r"\b(what|which|who|where|when|how)\b", question, re.IGNORECASE
    )
    if not interrogative:
        return
    tail = question[interrogative.start() :]
    if re.match(r"how\s+many\b", tail, re.IGNORECASE):
        quantities = (
            r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
            r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
            r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
            r"million|billion|none|no|both|several|few|many|dozens?"
        )
        if not re.search(r"\d|\b(?:" + quantities + r")\b", answer, re.IGNORECASE):
            raise ValueError("question_requires_quantity_not_entity")
    if interrogative.group().lower() == "when" and not re.search(r"\d", answer):
        # Reject positive evidence of a location used as a time. Do not require
        # an exhaustive vocabulary of dates: named historical events are valid.
        geographic = {"location", "city", "country", "venue", "place"}
        location_cell = any(
            words(name) in geographic and words(answer) == words(value)
            for name, value in cells(sources)
        )
        dated_location = any(
            re.search(r"\bheld\s+(?:on|from)\s+\d", s["text"], re.IGNORECASE)
            and re.search(
                r"\b(?:in|at)\s+" + re.escape(answer) + r"\b", s["text"], re.IGNORECASE
            )
            for s in sources
        )
        if location_cell or dated_location:
            raise ValueError("question_requires_time_not_location")


def check_explicit_column(answer, question, excerpts, sources):
    lead = words(question)
    start = re.match(r"^(?:what|which)\s+(?:(?:is|was|are|were)\s+)?(?:the\s+)?", lead)
    if not start:
        return
    lead = lead[start.end() :]
    cited_rows = [s for s in sources if s["kind"] == "row"]
    selected = cells(cited_rows or excerpts.values())
    excluded = {"name", "title", "year", "date", "country", "location"}
    requested = {
        name
        for name, _ in selected
        if words(name.split("/")[0]) not in excluded
        and (
            lead == words(name.split("/")[0])
            or lead.startswith(words(name.split("/")[0]) + " ")
        )
    }
    if len(requested) != 1:
        return
    column = requested.pop()
    target = words(answer)

    def contains(value):
        return bool(target) and (" " + target + " ") in (" " + words(value) + " ")

    if not any(contains(value) for name, value in selected if name == column) and any(
        contains(value) for name, value in selected if name != column
    ):
        raise ValueError("answer_is_other_field_not_requested_column:" + column)


def check_work_title(answer, question, sources):
    if not re.fullmatch(r"\d{4}", answer):
        return
    lead = re.split(
        r"\b(?:by|of|that|who|with|for|from|in|at)\b",
        question,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    work = r"(?:movie|film|book|novel|album|song)"
    if not re.match(r"^(?:what|which)\b", lead, re.IGNORECASE) or not re.search(
        r"\b" + work + r"\b", lead, re.IGNORECASE
    ):
        return
    text = "\n".join(s["text"] for s in sources)
    named = (
        r"(?:\b(?:called|named|titled)\s+|\b(?:title|"
        + work
        + r")\s*[:,]\s*|^)"
        + re.escape(answer)
        + r"\b"
    )
    if re.search(named, text, re.IGNORECASE | re.MULTILINE) or any(
        words(s.get("title", "")) == words(answer) for s in sources
    ):
        return
    if re.search(
        r"\b(?:released|published)\s+(?:in|on)\s+" + re.escape(answer) + r"\b",
        text,
        re.IGNORECASE,
    ):
        raise ValueError("question_requires_work_title_not_release_year")


def numeric_atom_supported(answer, sources):
    """Return None for nonnumeric answers, else match whole signed numbers."""
    numeric = r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?"
    if not re.fullmatch(numeric, answer):
        return None
    value = Decimal(answer.replace(",", ""))
    for source in sources:
        text = source["text"].replace("\u2212", "-")
        for match in re.finditer(
            r"(?<![\w.])[+-]?\s*\d+(?:,\d{3})*(?:\.\d+)?(?![\w.])", text
        ):
            number = Decimal(re.sub(r"\s|,", "", match.group()))
            if re.search(
                r"\b(?:minus|negative)\s*$", text[: match.start()], re.IGNORECASE
            ):
                number = -abs(number)
            if value == number:
                return True
    return False


def named_subject_anchor(answer, sources, excerpts):
    """Supply a named subject only from the same passage's first sentence."""
    for source in sources:
        if source["kind"] != "passage" or words(answer) != words(
            source.get("title", "")
        ):
            continue
        if not re.match(r"^(?:it|they|he|she)\b", source["text"], re.IGNORECASE):
            continue
        first = excerpts.get(source["evidence_id"] + ".S1")
        if first and words(first["text"]).startswith(words(answer) + " "):
            resolve_quote(answer, first["text"])
            return first
    return None
