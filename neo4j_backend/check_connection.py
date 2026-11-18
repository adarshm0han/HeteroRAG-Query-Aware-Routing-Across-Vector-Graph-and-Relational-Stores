"""Verify connectivity and report basic graph counts."""

from neo4j import GraphDatabase
from neo4j_backend.config import Neo4jConfig


def main() -> None:
    cfg = Neo4jConfig()
    if not cfg.password:
        raise SystemExit("Set NEO4J_PASSWORD in your environment first.")
    driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
    try:
        driver.verify_connectivity()
        with driver.session(database=cfg.database) as session:
            result = session.run(
                "MATCH (n) RETURN labels(n) AS labels, count(*) AS count "
                "ORDER BY labels"
            ).data()
        print(f"Connected to {cfg.uri}, database={cfg.database}")
        print("Node counts:")
        for row in result:
            print(f"  {row['labels']}: {row['count']}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
