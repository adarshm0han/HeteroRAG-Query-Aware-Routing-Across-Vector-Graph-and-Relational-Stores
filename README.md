# HeteroRAG

Query-aware routing and evidence fusion across PostgreSQL, Neo4j, and Pinecone for question answering over tables and linked text.

This repository contains the implementation and **one consolidated experiment with 6,000 main evaluation outcomes across 600 distinct HybridQA questions**. The experiment examines routing, fusion, and access to PostgreSQL hyperlinks. Pilot runs, smoke-test outputs, validation results, timing repeats, presentations, and duplicate reports are excluded from the published results.

## System

The system receives a question and its source table ID. OpenAI Decisions proposes a subset of the available databases using the question and public table metadata. An evidence guard adds passage retrieval when it cannot certify that the complete question is answerable from the table alone. Retrieved evidence passes through ranking, fusion, and context packing before a local Qwen3.5 4B model extracts and checks an answer.

```text
Question + source table ID
    -> public schema and graph-coverage metadata
    -> database selection + evidence guard
    -> PostgreSQL / Neo4j / Pinecone retrieval
    -> evidence ranking, fusion and bounded context
    -> local answer extraction and source checks
```

| Component | Role |
|---|---|
| PostgreSQL | Table rows, cells, and original hyperlinks; structured SQL planning with lexical fallback |
| Neo4j | Table-row/passage hyperlink connectivity and graph traversal |
| Pinecone | Passage embeddings, semantic search, and passage retrieval |
| Decisions | A bounded choice among available database combinations |
| Evidence guard | Requires passage access when a complete table-only certificate is unavailable |
| Local reader | Qwen3.5 4B with a pinned model digest and source-excerpt validation |

The router sees the table title, column names, row count, and graph availability, but not the table cells, passages, or reference answer. Its **raw route** is the model's proposal. Its **guarded route** is the proposal after the evidence guard adds required passage access. The guard can add Pinecone but does not automatically add Neo4j.

**Reciprocal-rank fusion (RRF)** combines evidence lists using each item's rank in each list. Weighted RRF additionally weights sources using the routing distribution. Concatenation joins the evidence lists without that rank-based combination. All variants pass through the same bounded context packer, which places table evidence first. Routing probabilities describe source preferences; they are not calibrated answer-correctness probabilities.

Fixed controls bypass the router and guard. Reference answers enter scoring after inference. This is HybridQA's **provided-table setting**, not corpus-wide table discovery.

## Experiment design

The experiment has two sequential phases. Each condition within a phase uses the same 300 questions. The phases have disjoint question IDs, although source tables can overlap. The link-access phase was designed after inspection of the routing/fusion results; it is an exploratory ablation within the consolidated experiment.

| Phase | Questions | Conditions | Main outcomes | Questions with graph coverage |
|---|---:|---:|---:|---:|
| Routing and fusion | 300 | 12 | 3,600 | 85 |
| PostgreSQL link access | 300 | 8 | 2,400 | 80 |
| **Total** | **600 distinct** | **20 phase-specific conditions** | **6,000** | **165** |

An outcome is one question evaluated under one condition. The 6,000 outcomes are paired observations within their respective phases, not 6,000 independent questions. All **14 failed main outcomes** remain in their original denominators with zero answer scores.

The routing/fusion phase compares all seven fixed database subsets, guarded and raw routing, weighted RRF, standard RRF, and concatenation. PostgreSQL + Pinecone was selected as the fixed baseline using separate training-split validation before the main evaluation. Those preliminary outputs are not included here.

The link-access phase crosses four policies with two PostgreSQL link settings: **1,200 visible-link controls and 1,200 hidden-link outcomes**. Masking removes row links, passage IDs, and cell entity IDs at the application boundary before evidence ranking and packing. Table values remain available. Neo4j retains its original links and coverage. The mask does not modify persistent PostgreSQL data or reduce the original SQL response bytes.

Graph coverage means Neo4j contains graph data for the question's source table. It does not imply that the answer is present, that the question requires Neo4j, or that the system answers it correctly.

## Main results

EM is normalized exact match. Token F1 measures overlap between the prediction and reference answer. Scores use all 300 questions per condition, including failures.

### Routing and fusion

| Condition | EM (%) | F1 (%) | Data requests / question |
|---|---:|---:|---:|
| PostgreSQL | 5.33 | 6.22 | 2.00 |
| Neo4j | 1.00 | 1.00 | 1.28 |
| Pinecone | 22.33 | 27.20 | 1.00 |
| PostgreSQL + Neo4j | 4.67 | 5.55 | 3.28 |
| PostgreSQL + Pinecone | 55.33 | 61.17 | 3.93 |
| Neo4j + Pinecone | 32.67 | 37.56 | 2.51 |
| All stores, RRF | 55.67 | 61.11 | 5.22 |
| All stores, concatenation | 56.00 | 61.70 | Replay |
| Raw Decisions | 35.67 | 40.71 | 3.18 |
| Guarded Decisions, RRF | 55.00 | 60.84 | 3.92 |
| Guarded Decisions, weighted RRF | 55.00 | 60.84 | Replay |
| Guarded Decisions, concatenation | 55.00 | 60.84 | Replay |

All seven named database subsets are fixed controls. **Replay** means the condition reused saved candidates to isolate the fusion comparison. Its zero new retrieval calls do not mean the equivalent deployed system would retrieve evidence for free.

Guarded routing made **24.90% fewer application data requests** than all-store retrieval. Its paired F1 difference was **−0.28 points**, with a 95% interval of **[−2.45, 1.91]**. This did not meet the prespecified two-point noninferiority margin.

All 299 successful guarded routes selected PostgreSQL + Pinecone and matched that fixed pair's scores and requests. The guard added passage retrieval to 112 PostgreSQL-only proposals. Guarded versus raw routing improved F1 by **20.12 points [15.70, 24.60]**.

Routed RRF, weighted RRF, and concatenation produced identical final reader contexts on all 299 successful paired questions. Their tied scores therefore do not establish the effect of different reader-visible evidence rankings. The separate all-store concatenation comparison also did not establish a quality advantage.

### PostgreSQL link access

| Policy | Links visible: F1 (%) | Links hidden: F1 (%) |
|---|---:|---:|
| Fixed PostgreSQL + Pinecone | 60.35 | 48.94 |
| Fixed all stores | 61.55 | 52.20 |
| Guarded Decisions | 60.18 | 48.50 |
| Raw Decisions | 39.76 | 38.00 |

Hiding PostgreSQL links reduced fixed PostgreSQL + Pinecone quality by **11.41 F1 points [5.81, 17.09]**. Adding Neo4j to the hidden-link fixed pair improved F1 by **3.26 points [0.78, 6.13]** overall and **13.46 points [4.22, 23.46]** on the 80 graph-covered questions.

The increase in Neo4j's benefit between visible and hidden links was **2.06 points [−0.60, 5.08]** overall. This interaction interval includes zero. A positive graph contribution within the hidden-link setting does not establish that its contribution reliably increased across settings.

None of the 600 main routing proposals in this phase selected Neo4j, even though all seven choices were available on the 80 graph-covered questions in each arm. Guarded routing did not outperform the fixed pair.

### Interpretation and limits

- Access to linked evidence materially affected quality. Total quality remained higher when PostgreSQL links were available.
- Adaptive routing and weighted fusion showed no incremental advantage in this implementation. This does not isolate the Decisions model from its prompt, metadata, guard, retrieval, and reader.
- PostgreSQL used lexical fallback in 298 of 299 successful guarded retrievals in the routing/fusion phase and every successful PostgreSQL retrieval in the link-access phase. These results do not evaluate a consistently functioning text-to-SQL alternative.
- Graph coverage is partial. The main questions come from the labeled development split, and this is not a full HybridQA leaderboard evaluation.
- Intervals use 2,000 source-table bootstrap samples without multiple-comparison adjustment. Post-hoc diagnostics and sensitivity analyses are labeled in the data.
- Data request counts exclude control operations and Decisions calls. Replay conditions do not measure operational savings. Per-outcome timings are descriptive; no controlled speedup or total billing reduction is claimed.
- No gold-evidence precision/recall claim or matched alternative-router comparison is included. Any redesign informed by these results needs a fresh held-out evaluation.

## Results and reproducibility

The [single experiment results package](results/heterorag/README.md) contains:

- [All 6,000 outcomes](results/heterorag/outcomes.csv), including predictions, reference answers, scores, routes, requests, descriptive timings, and original result hashes.
- [Condition metrics](results/heterorag/condition-metrics.csv), [paired comparisons](results/heterorag/paired-comparisons.csv), and [graph strata](results/heterorag/graph-strata.csv).
- [Analysis and diagnostics](results/heterorag/analysis.json), including uncertainty, retained failures, and the labeled complete-case sensitivity check.
- [Compact evidence traces](results/heterorag/evidence-traces.jsonl), [experiment definition](results/heterorag/experiment.json), [provenance](results/heterorag/provenance.json), and [checksums](results/heterorag/manifest.json).

Every exported outcome was checked against its original saved checkpoint. No inference was rerun to produce this package. Original bundle IDs and source hashes remain available for provenance. The public package excludes full retrieved corpus contexts, service logs, credentials, model caches, and preliminary run artifacts.

Verify the published results using Python 3.11+ and the standard library, without database or API access:

```bash
python -m tools.experiment verify
python -m tools.experiment summarize
```

Verification checks file hashes, complete paired cohorts, outcome identities, all answer scores, failure denominators, summary means, paired effect estimates, and link-access interactions. The [evaluation guide](docs/evaluation.md) explains the result schema and execution provenance.

## Repository layout

| Path | Contents |
|---|---|
| `routing/` | Router, guard, database adapters, SQL planning, evidence fusion, local reader |
| `evaluation/` | Frozen protocols, checkpointed execution, scoring, paired analysis |
| `experiments/heterorag/` | Execution definitions and link-access adapter for the experiment phases |
| `results/heterorag/` | The consolidated main evaluation package |
| `baseline/` | Fixed-plan implementation and shared scoring/context utilities |
| `postgres/`, `vector/`, `neo4j_backend/` | Database ingestion, verification, and query utilities |
| `tools/` | Offline result verification and summary commands |
| `docs/` | Setup, access contracts, and evaluation documentation |

## Installation and use

For live retrieval, install the pinned dependencies in Python 3.11 and configure a local Ollama service:

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements-dev.txt
```

Follow the [routing setup guide](routing/README.md#setup) for environment variables, pinned local assets, and individual runs. Store setup is documented for [PostgreSQL](postgres/README.md), [Pinecone](vector/README.md), and [Neo4j](neo4j_backend/README.md). Credentials belong in local environment files, which Git ignores.

To run offline development checks:

```bash
python -m pytest -q
python -m ruff check baseline routing evaluation tools
python -m ruff format --check baseline routing evaluation tools
```

Synthetic tests block external services by default. A new live execution requires its own frozen bundle and output directory; do not overwrite the published experiment.

## Data and attribution

Dataset revisions and archive hashes are pinned in [sources.lock.json](postgres/sources.lock.json). The outcome export includes benchmark question/reference text and model predictions for auditability; complete tables and passage corpora are obtained from upstream.

- [HybridQA questions and evaluator](https://github.com/wenhuchen/HybridQA)
- [WikiTables-WithLinks evidence corpus](https://github.com/wenhuchen/WikiTables-WithLinks)
- [Chen et al. (2020), HybridQA](https://aclanthology.org/2020.findings-emnlp.91/)
- [Dataset attribution and license](https://hybridqa.github.io/)
