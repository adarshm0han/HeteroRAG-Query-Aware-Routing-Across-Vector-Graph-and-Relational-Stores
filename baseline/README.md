# Fixed-plan baselines and shared utilities

This package contains the initial four-plan retrieval implementation and utilities shared by the current [routing pipeline](../routing/README.md). The seven-combination comparison and controlled fusion experiments use [the evaluation package](../docs/evaluation.md).

| Plan | Reader evidence |
|---|---|
| `sql_only` | Lexically ranked PostgreSQL rows |
| `vector_only` | Semantically ranked table-linked passages |
| `sql_links` | SQL rows and passages linked from those rows |
| `all_stores` | Rows, passages, and graph-expanded evidence with fixed RRF |

Here, `vector_only` obtains complete table membership from PostgreSQL before fetching Pinecone records. It is a passages-only evidence control, not an independent Pinecone-only system. The current router implements a separate Pinecone-only option. All plans use the supplied source table.

## Usage

Install the root requirements and configure [database access](../routing/README.md#setup). Retrieve a sample, then replay its saved contexts through an installed local Ollama model:

```powershell
python -m baseline --count 50 --output artifacts/baseline/retrieval --retrieve-only
python -m baseline.local --source artifacts/baseline/retrieval --output artifacts/baseline/qwen --model qwen3.5:4b --count 50
python -m baseline.verify artifacts/baseline/qwen
```

Local replay connects only to Ollama and makes no database or paid inference calls. It checks model, source, and configuration hashes when resuming. The optional paid baseline reader requires `--allow-paid-api`; it is not used by the current evaluation.

## Shared code

`common.py` provides canonical public question fields, hashes, deterministic sampling, and HybridQA normalization. `retrieval.py` supplies context packing and ranking helpers. `local_reader.py` implements local inference. `answers.py` and `report.py` provide answer contracts and scoring utilities.

The other modules support fixed-plan retrieval, saved-context replay, source verification, and paired reader comparisons. Runtime artifacts are ignored by Git. Answer-trace overlap is a weak diagnostic, not complete gold supporting evidence.
