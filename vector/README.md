# Vector component

Embeds HybridQA's hyperlinked Wikipedia passages locally and serves them from a shared Pinecone index. For querying the index that is already seeded, use [the teammate guide](../docs/vector-access.md); no reseeding is needed.

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | `download`, `embed`, `upload` (resumable), `verify` commands |
| `retriever.py` | `VectorRetriever.search(question, k, table_ids=None)` for the RAG pipeline |
| `examples/query.py` | Runnable query example |
| `test_pipeline.py` | Offline parsing/id/metadata tests |

## Corpus and model

- Source: `WikiTables-WithLinks/request_tok` at revision `dc066e1a`, the passage text behind every `/wiki/...` link in the HybridQA tables.
- 512,734 non-empty table-to-passage links dedupe to **286,267 unique non-empty passages** (median 120 words, p99 411). One passage = one vector; no chunking, so record identity matches HybridQA's gold passage links.
- Embedding model: `BAAI/bge-small-en-v1.5` (384-d, cosine, run locally — free). Passages are embedded as `"{title}. {text}"`; queries get the bge query instruction (handled by `retriever.py`).
- Size: 286K × 384 × 4 B ≈ 440 MB of vectors plus ~190 MB of passage text.

The encoder accepts at most [512 tokens](https://huggingface.co/BAAI/bge-small-en-v1.5), including the title; longer passages are truncated for embedding, while metadata retains the full text.

## Pinecone index

`hybridqa-passages`, serverless AWS us-east-1 (Starter/free plan), dimension 384, cosine.

Record id: the `/wiki/...` link, percent-encoded because Pinecone ids must be ASCII (22K links are not, e.g. `/wiki/São_Paulo`). Metadata:

| Field | Meaning |
|---|---|
| `passage_id` | Original `/wiki/...` link; same identifier as `entity_ids` in `hybridqa.rag_rows` |
| `title`, `text` | Passage title and full text |
| `table_ids` | HybridQA tables linking to this passage, capped at 300 (Pinecone's 40 KB metadata limit) |
| `table_count` | True number of linking tables; the complete link map is in PostgreSQL |

Table filtering uses the capped `table_ids` metadata. For the 12 passages linked from more than 300 tables, filtering on an omitted table can miss a relevant passage (2,418 omitted links across this corpus). This is a retrieval limitation; use PostgreSQL's complete link map when exhaustive table membership is required.

## Free-plan budget

Starter allows 2 GB storage, 2M write units and 1M read units per month. A query costs 1 RU per GB of namespace (this index ≈ 0.65 GB), regardless of `top_k`, so ~1.5M queries/month. Upserts cost 1 WU per KB: a full seed uses roughly 0.7–1.3M WU, so **only seed once per month**. `upload` checkpoints to `data/upload_checkpoint.json` and resumes instead of re-sending records.

## Rebuild as the index maintainer

Requires Python 3.10–3.12. Run from this `vector/` directory.

```bash
python -m pip install -r requirements.txt
cp .env.example .env            # add the Pinecone API key; never commit it
python pipeline.py download
python pipeline.py embed        # uses CUDA/MPS when available; resumes from data/embedding_chunks/
python pipeline.py upload
python pipeline.py verify
```

`download` caches and checksum-verifies the pinned corpus ZIP without extracting filenames, so it also works on Windows. To reuse the same archive already downloaded for PostgreSQL, run `python pipeline.py download --archive /path/to/corpus.zip`. Existing `data/WikiTables-WithLinks/request_tok/` checkouts remain readable when no cached ZIP is present.

Upload checkpoints are bound to the Pinecone index host and the hashes of both local data files. A different index, modified data, or an older checkpoint containing only `next` is rejected before upserting. Keep the checkpoint for ordinary resumes. For an intentional fresh upload, move `data/upload_checkpoint.json` aside first; this re-sends every record and consumes write units. Recreating or clearing an index also requires starting a fresh upload.

Offline checks: `python -m pytest -q`. These cover parsing, portable downloads, checkpoint isolation and retry, and the retriever's query/evidence contract without calling Pinecone or downloading a model.

On Apple Silicon the MPS backend ran out of memory on the full corpus; `python pipeline.py embed --device cpu` finished all 286,267 passages in 28 minutes on an M-series CPU.

`data/embeddings.npy` + `data/passages.parquet` (row i ↔ vector i) can also back a local FAISS index for offline experiments.
