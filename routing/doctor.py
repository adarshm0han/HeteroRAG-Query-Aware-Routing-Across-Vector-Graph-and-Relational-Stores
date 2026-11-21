"""Check local setup without contacting Decisions or any cloud database."""

import argparse
import importlib.metadata
import json

from baseline.common import REPO, safe_error

from .config import database_settings, decisions_key
from .tokenizer import local_tokenizer


def inspect(workspace):
    cfg = database_settings(workspace)
    required = (
        "DATABASE_URL",
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
        "PINECONE_API_KEY",
    )
    report = {
        "credential_presence": {key: bool(cfg.get(key)) for key in required},
        "decisions_key_present": bool(decisions_key(workspace)),
        "paid_calls_enabled": False,
        "cloud_database_connections_tested": False,
        "versions": {},
    }
    for package in (
        "psycopg",
        "neo4j",
        "pinecone",
        "numpy",
        "sentence-transformers",
        "tiktoken",
    ):
        try:
            report["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["versions"][package] = None
    try:
        tokenizer = local_tokenizer(workspace)
        report["tokenizer"] = {"ready": bool(tokenizer.encode("setup check"))}
    except (ValueError, OSError) as exc:
        report["tokenizer"] = {"ready": False, "error": safe_error(exc)}
    cache = workspace / ".local/hf-cache/models--BAAI--bge-small-en-v1.5/snapshots"
    report["embedding_cache_present"] = cache.is_dir() and any(cache.iterdir())
    try:
        from baseline.local_reader import LocalReader

        reader = LocalReader("qwen3.5:4b")
        report["qwen"] = {
            "ready": True,
            "model": reader.metadata,
            "ollama": reader.runtime,
        }
    except (OSError, ValueError) as exc:
        report["qwen"] = {"ready": False, "error": safe_error(exc)}
    report["local_setup_ready"] = (
        all(report["credential_presence"].values())
        and report["decisions_key_present"]
        and all(report["versions"].values())
        and report["tokenizer"]["ready"]
        and report["embedding_cache_present"]
        and report["qwen"]["ready"]
    )
    return report


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=REPO.parent)
    args = parser.parse_args()
    print(json.dumps(inspect(args.workspace), indent=2))


if __name__ == "__main__":
    main()
