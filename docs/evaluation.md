# Experiment and reproducibility

HeteroRAG's published experiment contains 6,000 main outcomes in two sequential phases: routing/fusion on 300 questions and PostgreSQL link access on 300 different questions. Use the [root README](../README.md) for findings and the [results package](../results/heterorag/README.md) for data. Preliminary development, smoke, validation, and timing-repeat outputs are not part of the package.

## Published results

```bash
python -m tools.experiment verify
python -m tools.experiment summarize
```

These commands use the standard library and local files only. Verification recomputes EM/F1 using the project's pinned HybridQA normalization, checks condition means and paired effect estimates, verifies complete cohort membership and checksums, and confirms that all 14 failures retain zero scores. Summary output keeps phases separate.

The original 95% bootstrap intervals are preserved in `analysis.json` and `paired-comparisons.csv`. They use 2,000 draws clustered by source table and seed 20261116, without multiplicity adjustment. The verifier recomputes point estimates; it does not rerun bootstrap resampling. The original bootstrap implementation is `baseline.report.cluster_interval`.

Original run names and bundle IDs in provenance identify saved execution artifacts. They are not additional experiments or extra published outcome sets. `source_result_sha256` links each public outcome to the original saved checkpoint file. The manifest checks the public files themselves. Checksum verification detects changes relative to the manifest, not malicious replacement of both data and manifest.

## Cohorts, controls, and scope

The [experiment definition](../results/heterorag/experiment.json) lists the exact main question IDs, condition settings, graph coverage, and original bundle IDs. Questions come from the upstream labeled development split. The second cohort excludes all first-phase questions and two previously inspected examples. Source tables can overlap.

In the routing/fusion phase, the seven fixed database subsets bypass the router and guard. Raw routing reuses the guarded condition's paid proposal but retrieves independently. Weighted and concatenation routing conditions replay the same saved retrieval candidates as guarded RRF; `all_concat` replays the all-store candidates. Their replay request counts are not efficiency measurements. PostgreSQL + Pinecone was chosen using separate training-split validation before the main evaluation. The selection decision and its source hash are preserved, but those preliminary scores and outcomes are excluded.

In the link-access phase, four policies run with PostgreSQL links visible and hidden. All eight conditions use the same 300 questions. Raw routing reuses its corresponding arm's paid proposal with independent retrieval. The adapter masks PostgreSQL link identifiers before ranking and evidence packing. The duplicate-row merge preserves links actually returned by Neo4j. Both arms use that merge rule. Capability descriptions also change in the routed hidden-link arm, so only the fixed-policy contrasts isolate link access alone.

Failures remain in each 300-question denominator. There are seven in each phase. Main failed outcomes were preserved rather than retried to improve accuracy. Per-outcome times are descriptive because replay, recovery, cache, and power conditions limit attribution. Gold-evidence precision/recall is not claimed.

## Datasets and reproducibility

Use the revisions and SHA-256 hashes in [sources.lock.json](../postgres/sources.lock.json). The default workspace is the repository's parent directory and holds `data/questions.zip` and `data/corpus.zip`.

```powershell
python postgres/pipeline.py download
New-Item -ItemType Directory -Force ../data
Copy-Item postgres/data/questions.zip, postgres/data/corpus.zip ../data/
```

These commands download local archives; they do not seed databases. Configure the [reader, tokenizer, embeddings, and credentials](../routing/README.md#setup) before a new live execution. Database contents must follow the same pinned evidence contracts. Graph availability is a frozen snapshot.

## Execution definitions

The [experiment execution files](../experiments/heterorag/README.md) retain the original preparation settings and intervention code. The routing/fusion preparation definition includes the historical phase gates and preliminary input IDs for reproducibility; it contains no preliminary result files. Those gates are execution controls and do not change the published 6,000-outcome denominator.

```powershell
python -m evaluation.protocol prepare --definition experiments/heterorag/routing-fusion --bundle artifacts/bundles/new-routing-fusion
python -m evaluation.protocol verify --bundle artifacts/bundles/new-routing-fusion
```

Preparation reconstructs inputs from pinned archives and freezes the current source. It makes no model or database requests. A nonempty destination is rejected. A fresh preparation has a new source identity and must not be represented as the original saved execution.

Live execution requires explicit runner flags for external reads and paid Decisions calls, a request cap matching the frozen protocol, and a new output directory. Use `python -m evaluation.runner --help`. The runner preserves checkpoints and an at-most-once paid-request ledger. Do not edit active run files or bypass the ledger by changing output directories. See [the link-access preparation notes](../experiments/heterorag/README.md#link-access-phase) for its additional runtime requirements.

## Interpretation

Compare conditions within a phase and preserve table-cluster pairing. Do not average the two phases into a single leaderboard score. The second phase was designed after inspection of the first; consolidation does not make it jointly prespecified.

Diagnostics concerning route probabilities, SQL fallback, identical contexts, and failure sensitivity are post hoc. The complete-case sensitivity analysis does not replace the primary complete-denominator results. Changes informed by these findings require new development and a fresh held-out evaluation.
