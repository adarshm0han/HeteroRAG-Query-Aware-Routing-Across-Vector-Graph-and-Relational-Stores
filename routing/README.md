# Routing and retrieval

The pipeline combines Decisions database selection, a schema-based evidence guard, read-only retrieval, evidence fusion, and a local Qwen3.5 4B reader. Experiment preparation and reporting are documented in [the evaluation guide](../docs/evaluation.md).

## Database combinations

| Option | Retrieval behavior |
|---|---|
| `postgres` | Locally plan typed filtering, ordering, or aggregation over the supplied table; execute parameterized SQL. |
| `neo4j` | Find independent question-based graph seeds and traverse shared-passage links to rows. |
| `pinecone` | Embed the question locally and query table-filtered passage vectors. |
| `postgres_neo4j` | Combine SQL evidence with independent graph seeds and grounded SQL seeds. |
| `postgres_pinecone` | Retrieve rows and semantic passages, rank fallback rows using passage links, and fetch linked passage bodies. |
| `neo4j_pinecone` | Connect semantic passage anchors to rows through the graph and fetch linked passage bodies. |
| `all` | Combine all three stores and fuse their evidence rankings. |

Adapters are created lazily and enforce the route's database allowlist. An absent graph table is recorded as missing coverage. Backend errors retain their call logs and fail the outcome; empty evidence produces an abstention.

## Routing policy

Decisions receives the question, table ID, and a local catalog of public table title, column names, row count, and graph availability. It receives no cells, passages, reference answers, or answer traces.

The evidence guard certifies only complete questions matching a conservative grammar for literal column lookups, counts, aggregates, and extrema. A certificate requires PostgreSQL. Unparsed clauses, ambiguous scope, or missing/truncated schema require Pinecone. The guard adds required stores before retrieval and makes no extra model or database requests. It does not add Neo4j automatically.

Results retain the raw choice, distribution, confidence, certificate, and effective route. Fusion weights use the probability mass after applying the same guard to every possible raw choice. These probabilities are not calibrated probabilities of answer correctness. The seven fixed controls bypass this policy.

## Evidence and answering

Retrieval uses canonical row and passage identifiers, bounded linked-passage expansion, and deduplication. Fusion supports RRF, probability-weighted RRF, and concatenation. Context packing uses a common 6,000-token budget with table evidence first.

Qwen produces a draft and an independent full-question review. Source excerpts, quotation spans, answer types, and numeric expressions are checked in code. These checks do not guarantee entailment or complete reasoning. Source fidelity and weak answer-trace overlap are reported separately from answer EM/F1.

## Setup

Install the root requirements and run commands from the repository root. The default workspace is its parent directory; `--workspace PATH` selects another workspace for datasets, caches, and credential files.

```text
<workspace>/data/questions.zip
<workspace>/data/corpus.zip
<workspace>/.local/hf-cache/
<workspace>/.local/tiktoken-cache/
```

The [evaluation guide](../docs/evaluation.md#datasets-and-reproducibility) explains the pinned archives. Provision the local model and caches once; execution itself does not download them:

```powershell
ollama pull qwen3.5:4b
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5', cache_folder='../.local/hf-cache', revision='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a')"
$env:TIKTOKEN_CACHE_DIR = '../.local/tiktoken-cache'
python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
```

Adjust cache paths when using another workspace. The experiment definition pins the Qwen model digest and Python dependency versions. A different digest under the same model tag requires a new experiment definition. The tokenizer cache is checksum-verified before use.

Environment variables override credential files:

| Settings | Optional file |
|---|---|
| `DATABASE_URL` | `<workspace>/.env.rag_reader` or `postgres/.env.rag_reader` |
| `PINECONE_API_KEY`, `PINECONE_INDEX` | `vector/.env` |
| `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` | `.env.neo4j` |
| `DECISIONS-API-KEY` | `<workspace>/.env.open-ai-key` |

Use read-only database accounts. Run `python -m routing.doctor` to check credential presence, package versions, caches, and the local model without querying cloud databases or Decisions.

## Individual runs

`questions.json` contains objects with `question_id`, `question`, and `table_id`. A fixed PostgreSQL/Pinecone run uses:

```powershell
python -m routing --questions questions.json --output artifacts/runs/example --option postgres_pinecone --routing-catalog artifacts/bundles/new-routing-fusion/routing-catalog.json --graph-catalog artifacts/bundles/new-routing-fusion/graph-table-ids.json
```

The `decisions` option additionally requires `--allow-decisions-api` and `--max-decision-requests N`. Use `evaluation.runner` for checkpointed studies; the individual-run CLI expects a new output directory.

## Modules

| Modules | Responsibility |
|---|---|
| `router`, `decisions`, `evidence_policy`, `options` | Proposals, transport contract, evidence guard, database combinations |
| `catalog`, `stores`, `retrieval` | Public metadata, database boundaries, candidate retrieval |
| `sql_plan`, `sql_audit`, `graph_query` | Typed SQL plans, aggregate verification, graph traversal |
| `row_selection`, `passage_selection`, `fusion` | Candidate ranking and fusion |
| `reader`, `answer_context`, `answer_validation`, `answer_contract`, `evidence_answer` | Extraction and source validation |
| `pipeline`, `audit`, `report` | Inference orchestration, source checks, single-run metrics |
