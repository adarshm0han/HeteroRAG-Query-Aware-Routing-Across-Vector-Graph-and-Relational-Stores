"""Environment-based configuration for the HeteroRAG Neo4j backend."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")


@dataclass(frozen=True)
class PostgresConfig:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    dbname: str = os.getenv("POSTGRES_DB", "hybridqa")
    user: str = os.getenv("POSTGRES_USER", "")
    password: str = os.getenv("POSTGRES_PASSWORD", "")
    source_view: str = os.getenv("POSTGRES_SOURCE_VIEW", "hybridqa.rag_rows")
