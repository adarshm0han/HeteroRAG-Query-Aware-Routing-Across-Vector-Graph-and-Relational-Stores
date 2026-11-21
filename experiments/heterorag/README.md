# HeteroRAG experiment execution files

The published experiment is a single [6,000-outcome results package](../../results/heterorag/README.md) with two sequential phases. These directories contain the preparation definitions and intervention code, not additional result sets.

## Routing and fusion phase

`routing-fusion/config.json` preserves the original preparation settings, main question IDs, historical input IDs, and expected archive/input hashes. `routing-fusion/graph-table-ids.json` is the frozen graph coverage snapshot. Original preliminary phase gates remain in this definition so the execution design is documented, while their result files are excluded from publication.

```powershell
python -m evaluation.protocol prepare --definition experiments/heterorag/routing-fusion --bundle artifacts/bundles/new-routing-fusion
python -m evaluation.protocol verify --bundle artifacts/bundles/new-routing-fusion
```

The 12 main conditions cover all seven fixed database subsets and controlled routing/fusion variants. The published phase contains exactly 3,600 main outcomes. See the [evaluation guide](../../docs/evaluation.md) for pairing, replay, and baseline selection.

## Link-access phase

`link-access/link_access.py` implements application-level PostgreSQL link masking and the corresponding router capability descriptions. `link-access/prepare.py` constructs the frozen execution runtime, including the duplicate-row merge that retains links supplied by Neo4j. `link-access/verify_local.py` checks the intervention with synthetic fixtures.

The preparation script is an archival reproduction recipe. It requires the original frozen routing/fusion bundle under `artifacts/bundles/hybridqa-dev300-r2`, including its source snapshot, along with the pinned local archives. That original bundle is not included in the public repository. The script verifies source hashes before copying and patching it and refuses to overwrite an existing runtime or bundle. A new clone can verify all published results without this bundle; reproducing the original inference execution requires the original assets and live services.

```powershell
python experiments/heterorag/link-access/prepare.py
python experiments/heterorag/link-access/verify_local.py
```

The published phase contains exactly 2,400 main outcomes: four policies under visible and hidden link settings on the same 300 questions. Fixed policies isolate access to links; routed-arm comparisons also change the capability descriptions. Both arms retain the existing graph coverage and use the same rule for merging observed links.

## Provenance

Original runtime IDs remain in [provenance.json](../../results/heterorag/provenance.json) to identify the source artifacts. Organizing the phases together does not rewrite those identities, change predictions, or make the sequential design jointly prespecified. Source tables can overlap across the distinct question cohorts. New changes informed by this experiment need a new frozen bundle and fresh held-out questions.
