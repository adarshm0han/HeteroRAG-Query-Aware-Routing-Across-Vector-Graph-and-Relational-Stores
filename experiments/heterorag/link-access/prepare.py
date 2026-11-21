"""Build a separate, frozen link-access ablation from the completed source snapshot.

No inference or database operations. Existing source, bundles and results are read-only.
"""

import json
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WORKSPACE = REPO.parent
PRIOR = REPO / "artifacts/bundles/hybridqa-dev300-r2"
RUNTIME = REPO / "artifacts/link-access-runtime"
BUNDLE = REPO / "artifacts/bundles/hybridqa-link-access-v1"
SEED = "heterorag-link-access-v1-2026-11-20"


def replace(path, old, new):
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one patch location in {path}: {old[:60]}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def main():
    if RUNTIME.exists() or BUNDLE.exists():
        raise ValueError("Use a new version; never overwrite a prepared study")
    prior = json.loads((PRIOR / "protocol.json").read_text(encoding="utf-8"))
    import hashlib

    for name, digest in prior["code"].items():
        source = PRIOR / "source-code" / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError("Original frozen source checksum mismatch: " + name)
    shutil.copytree(PRIOR / "source-code", RUNTIME)
    shutil.copyfile(
        Path(__file__).with_name("link_access.py"), RUNTIME / "routing/link_access.py"
    )
    protocol_file = RUNTIME / "evaluation/protocol.py"
    text = protocol_file.read_text(encoding="utf-8")
    start = text.index("CONDITIONS = {")
    end = text.index("DEV300_COUNTS =", start)
    text = (
        text[:start]
        + """CONDITIONS = {
    f"{method}_links_{arm}": {"option": option, "fusion": "rrf", "postgres_links": arm == "on"}
    for arm in ("on", "off")
    for method, option in (
        ("fixed_pgpc", "postgres_pinecone"),
        ("fixed_all", "all"),
        ("routed_guarded", "decisions"),
        ("routed_raw", "decisions"),
    )
}
DEFAULT_PHASES = ["smoke", "evaluation"]
TIMING_CONDITIONS = []
DEPENDENCIES = {
    f"routed_raw_links_{arm}": f"routed_guarded_links_{arm}"
    for arm in ("on", "off")
}
"""
        + text[end:]
    )
    text = text.replace('name != "routed_raw"', 'not name.startswith("routed_raw_")')
    protocol_file.write_text(text, encoding="utf-8")
    runner = RUNTIME / "evaluation/runner.py"
    replace(
        runner,
        'selected_phase == "evaluation"\n                    and not',
        'selected_phase == "evaluation"\n                    and "validation" in protocol["phases"]\n                    and not',
    )
    replace(
        runner,
        "from routing.router import DecisionsRouter, FixedRouter",
        "from routing.router import FixedRouter\n        from routing.link_access import LinkAccessRouter as DecisionsRouter",
    )
    replace(
        runner,
        "from routing.stores import LiveStores",
        "from routing.link_access import LinkAccessStores as LiveStores",
    )
    replace(
        runner,
        'ledger.client(job["id"], api_key, allow_paid=True)\n                        )',
        'ledger.client(job["id"], api_key, allow_paid=True),\n                            postgres_links=condition["postgres_links"],\n                        )',
    )
    replace(
        runner,
        "journal=journal,\n                        )",
        'journal=journal,\n                            postgres_links=condition["postgres_links"],\n                        )',
    )
    report = RUNTIME / "evaluation/report.py"
    replace(
        report,
        'if phase == "evaluation":',
        'if phase == "evaluation" and "validation" in protocol["phases"]:',
    )
    text = report.read_text(encoding="utf-8")
    start = text.index('        controls = {\n            "weighted":')
    end = text.index("    payload = {", start)
    text = (
        text[:start]
        + """        for left, right in pairs:
            comparisons[left + " -> " + right]["by_graph_coverage"] = {
                str(flag).lower(): paired(
                    [r for r in rows[left] if r["graph_covered_snapshot"] == flag],
                    [r for r in rows[right] if r["graph_covered_snapshot"] == flag],
                    protocol["noninferiority_margin_f1_points"],
                ) for flag in (True, False)
            }
"""
        + text[end:]
    )
    text = text.replace(
        '        "fusion_controls": controls,',
        '        "fusion_controls": controls,\n        "experiment": protocol["experiment"],',
    )
    report.write_text(text, encoding="utf-8")
    retrieval = RUNTIME / "routing/retrieval.py"
    replace(
        retrieval,
        '                records[key]["sources"].append(store)',
        """                records[key]["sources"].append(store)
                # Retain links actually supplied by a selected graph store when
                # PostgreSQL's view is masked. No link is fabricated or fetched here.
                records[key]["links"] = list({
                    (link["column"], link["cell_text"], link["passage_id"]): link
                    for link in records[key]["links"] + item["links"]
                }.values())""",
    )

    sys.path.insert(0, str(RUNTIME))
    from baseline.common import object_hash, public_question, select_questions, sha256
    from evaluation.checkpoints import atomic_json
    from evaluation.cohorts import EXPOSED_DEV, code_hashes, weak_ids
    from evaluation.protocol import CONDITIONS, schedule, verify

    for key in ("questions", "corpus"):
        if (
            sha256(WORKSPACE / "data" / (key + ".zip"))
            != prior["sources"][key]["archive_sha256"]
        ):
            raise ValueError("Dataset archive checksum mismatch")
    # Selection uses IDs only; answer labels do not enter the sampling function.
    excluded = set(EXPOSED_DEV) | {
        q["question_id"]
        for q in json.loads(
            (PRIOR / "evaluation/questions.json").read_text(encoding="utf-8")
        )
    }
    with zipfile.ZipFile(WORKSPACE / "data/questions.zip") as archive:
        dev_name = next(
            n
            for n in archive.namelist()
            if n.endswith("/released_data/dev.traced.json")
        )
        dev = json.loads(archive.read(dev_name))
    public = [public_question(q) for q in dev if q["question_id"] not in excluded]
    chosen = select_questions(public, 300, SEED)
    chosen_ids = {q["question_id"] for q in chosen}
    references = {
        q["question_id"]: {"answer": q["answer-text"], "weak_evidence_ids": weak_ids(q)}
        for q in dev
        if q["question_id"] in chosen_ids
    }
    BUNDLE.mkdir(parents=True)
    atomic_json(BUNDLE / "evaluation/questions.json", chosen)
    atomic_json(BUNDLE / "evaluation/references.json", references)
    for name in ("questions.json", "references.json"):
        target = BUNDLE / "smoke" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(PRIOR / "smoke" / name, target)
    for name in ("graph-table-ids.json", "routing-catalog.json"):
        shutil.copyfile(PRIOR / name, BUNDLE / name)
    atomic_json(
        BUNDLE / "selection.json",
        {
            "seed": SEED,
            "excluded_ids": sorted(excluded),
            "selection": "300 public question IDs, independent of answers and graph coverage",
            "prior_questions_excluded": 300,
            "prior_inspected_examples_excluded": len(EXPOSED_DEV),
            "scope": "Fresh questions from upstream dev; source tables may overlap prior study",
        },
    )
    covered = set(
        json.loads((BUNDLE / "graph-table-ids.json").read_text(encoding="utf-8"))
    )
    cohorts = {
        phase: json.loads(
            (BUNDLE / phase / "questions.json").read_text(encoding="utf-8")
        )
        for phase in ("smoke", "evaluation")
    }
    jobs = [
        job
        for phase, questions in cohorts.items()
        for job in schedule(questions, phase)
    ]
    atomic_json(BUNDLE / "schedule.json", jobs)
    pairs = []
    for arm in ("on", "off"):
        pairs += [
            [f"fixed_pgpc_links_{arm}", f"fixed_all_links_{arm}"],
            [f"fixed_pgpc_links_{arm}", f"routed_guarded_links_{arm}"],
            [f"fixed_pgpc_links_{arm}", f"routed_raw_links_{arm}"],
            [f"routed_raw_links_{arm}", f"routed_guarded_links_{arm}"],
        ]
    pairs += [
        [f"{name}_links_on", f"{name}_links_off"]
        for name in ("fixed_pgpc", "fixed_all", "routed_guarded", "routed_raw")
    ]
    protocol = {
        "version": 1,
        "study": "hybridqa",
        "profile": "link-access-v1",
        "experiment": "PostgreSQL link access x graph availability; fixed and routed controls",
        "base_bundle_id": prior["bundle_id"],
        **{
            key: prior[key]
            for key in (
                "sources",
                "dependency_versions",
                "reader_digest",
                "config",
                "routing_catalog",
            )
        },
        "conditions": CONDITIONS,
        "phases": list(cohorts),
        "primary_pairs": pairs,
        "outcomes": len(jobs),
        "paid_request_cap": sum(j["paid"] for j in jobs),
        "paid_retries": 0,
        "wall_time_budget_seconds": 12 * 3600,
        "noninferiority_margin_f1_points": 2.0,
        "cohorts": {
            phase: {
                "questions": len(qs),
                "tables": len({q["table_id"] for q in qs}),
                "graph_covered": sum(q["table_id"] in covered for q in qs),
            }
            for phase, qs in cohorts.items()
        },
        "design": {
            "mask": "Application adapter clears PostgreSQL links, passage IDs and cell entity IDs before row ranking, expansion or reader packing; persistent data and DB response volume unchanged.",
            "router": "Original payload in links-on arm; capability statements corrected for hidden links in links-off arm; no instruction to favor Neo4j.",
            "guard": "Original deterministic guard unchanged in both arms; adds Pinecone, never automatically adds Neo4j.",
            "graph": "Existing coverage snapshot preserved; no graph seeding or database writes. Report covered/uncovered strata.",
            "controls": "Fixed PG+PC and all stores are prespecified controls in each arm; no best-baseline selection on evaluation.",
            "raw": "Reuse each arm's same paid proposal and retrieve independently without guard; no paid retry.",
            "timing": "Descriptive service timing only; no repeated controlled timing phase or causal speed claim.",
            "primary_scope": "Exploratory link-access phase on fresh questions; not a Neo4j engine superiority test. Router-arm changes include accurate capability descriptions.",
            "failure": "Retain failures with zero scores; stop on failure for inspection; do not repeat attempted external stages.",
            "merge": "Union actually observed links on duplicate row evidence, preserving Neo4j-supplied links when PostgreSQL links are masked; same rule in both arms.",
        },
        "code": code_hashes(),
    }
    protocol["inputs"] = {
        p.relative_to(BUNDLE).as_posix(): sha256(p)
        for p in sorted(BUNDLE.rglob("*.json"))
    }
    protocol["bundle_id"] = object_hash(protocol)
    atomic_json(BUNDLE / "protocol.json", protocol)
    for relative in protocol["code"]:
        target = BUNDLE / "source-code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(RUNTIME / relative, target)
    verify(BUNDLE, check_code=True)
    print(
        json.dumps(
            {
                "bundle": str(BUNDLE),
                "runtime": str(RUNTIME),
                "bundle_id": protocol["bundle_id"],
                "cohorts": protocol["cohorts"],
                "outcomes": len(jobs),
                "paid_request_cap": protocol["paid_request_cap"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
