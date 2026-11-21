"""The seven executable database subsets, in stable evaluation order."""

OPTIONS = {
    "postgres": ("postgres",),
    "neo4j": ("neo4j",),
    "pinecone": ("pinecone",),
    "postgres_neo4j": ("postgres", "neo4j"),
    "postgres_pinecone": ("postgres", "pinecone"),
    "neo4j_pinecone": ("neo4j", "pinecone"),
    "all": ("postgres", "neo4j", "pinecone"),
}

CAPABILITIES = """The supplied table ID scopes all retrieval. Only the supplied public
schema/coverage metadata is available before routing, never table cells or passage bodies.
postgres: a local planner produces a validated, read-only SQL plan. Supported:
AND filters on text/numbers, numeric or lexical sorting, dense-rank selection with
ties, count/count-distinct, sum/avg/min/max, grouping and aggregate filtering.
These operations run over the complete table before the output limit. Unsupported
or invalid plans use an explicitly recorded lexical fallback; general date parsing,
arbitrary SQL and external facts are not supported. Original hyperlinks remain and
can fetch direct linked bodies only if Pinecone is selected.
neo4j: keeps independent semantic/lexical row seeds and may add SQL row seeds;
follows row->passage<-row->passage paths within
the supplied table, with bounded seeds/neighbors/links and recorded provenance.
It has partial table coverage, table row text and links, but no passage bodies.
pinecone: passage bodies, semantic search filtered to the supplied table, and fetch
by linked passage identifier. No complete table or global table aggregation.
Its table-membership metadata is capped; link fetches bypass that metadata cap.
Combinations execute exactly their selected stores, with no hidden dependencies.
SQL can apply numeric/date conditions only when the relevant fact is actually in
the table and uses a supported representation. Pinecone is required for missing attributes.
A direct row link may suffice; graph traversal adds explicit shared-link connections.
"""


def allowed_options(graph_available=None):
    return [
        name
        for name, stores in OPTIONS.items()
        if graph_available is not False or "neo4j" not in stores
    ]
