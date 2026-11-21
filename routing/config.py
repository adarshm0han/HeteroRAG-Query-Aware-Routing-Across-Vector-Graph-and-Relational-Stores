"""Load only the credentials needed for the requested operation; never log them."""

import os

from baseline.common import REPO, read_env

DATABASE_KEYS = (
    "DATABASE_URL",
    "PINECONE_API_KEY",
    "PINECONE_INDEX",
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
)


def database_settings(workspace):
    cfg = {}
    for path in (
        workspace / ".env.rag_reader",
        REPO / "postgres/.env.rag_reader",
        REPO / "vector/.env",
        REPO / ".env.neo4j",
    ):
        cfg.update(
            {
                key: value
                for key, value in read_env(path).items()
                if key in DATABASE_KEYS
            }
        )
    cfg.update({key: os.environ[key] for key in DATABASE_KEYS if key in os.environ})
    return cfg


def decisions_key(workspace):
    cfg = read_env(workspace / ".env.open-ai-key")
    return os.environ.get("DECISIONS-API-KEY") or cfg.get("DECISIONS-API-KEY", "")
