"""Configuration, provenance, deterministic selection, and official EM/F1 rules."""

import collections
import hashlib
import json
import os
import re
import string
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QUESTION_REVISION = "db22fda8c5951438fade3c69d75b350335ba93b3"
CORPUS_REVISION = "dc066e1a6d5281511d8b73a6107d5ad2824cc2b2"
MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
PLANS = ("sql_only", "vector_only", "sql_links", "all_stores")


def read_env(path):
    values = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def settings(workspace):
    """Never return this dictionary in logs or manifests."""
    values = {}
    for path in (
        workspace / ".env.rag_reader",
        REPO / "postgres/.env.rag_reader",
        REPO / "vector/.env",
        REPO / ".env.neo4j",
        workspace / ".env.open-ai-key",
    ):
        values.update(read_env(path))
    for key in (
        "DATABASE_URL",
        "PINECONE_API_KEY",
        "PINECONE_INDEX",
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
        "NEO4J_DATABASE",
        "OPENAI_API_KEY",
        "DECISIONS-API-KEY",
    ):
        if key in os.environ:
            values[key] = os.environ[key]
    values["OPENAI_API_KEY"] = values.get("OPENAI_API_KEY") or values.get(
        "DECISIONS-API-KEY", ""
    )
    return values


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def select_questions(questions, count, seed):
    if not 1 <= count <= len(questions):
        raise ValueError(
            "Sample size must be between one and the available question count"
        )
    ids = [q["question_id"] for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question identifiers")
    # Selection depends only on question ID and seed, never answer/trace/coverage.
    return sorted(
        questions,
        key=lambda q: hashlib.sha256(f"{seed}:{q['question_id']}".encode()).hexdigest(),
    )[:count]


def public_question(question):
    # An explicit allowlist prevents answer text and traced annotations entering retrieval/prompts.
    return {k: question[k] for k in ("question_id", "question", "table_id")}


def normalize_answer(value):
    # Same normalization as the pinned HybridQA evaluate_script.py.
    text = "".join(c for c in value.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_scores(gold, prediction):
    a, b = normalize_answer(gold), normalize_answer(prediction)
    left, right = a.split(), b.split()
    common = sum((collections.Counter(left) & collections.Counter(right)).values())
    if not left or not right:
        f1 = float(left == right)
    else:
        f1 = 2 * common / (len(left) + len(right))
    return {"em": float(a == b), "f1": f1}


def safe_error(exc):
    # SDK exception messages can contain credentials, URLs or request headers.
    result = {"type": type(exc).__name__}
    if isinstance(getattr(exc, "status", None), int):
        result["http_status"] = exc.status
    return result
