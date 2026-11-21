"""Blinded independent annotation packets; never invents human evidence labels."""

import argparse
import json
import zipfile
from pathlib import Path

from baseline.common import (
    CORPUS_REVISION,
    object_hash,
    read_json,
    select_questions,
    sha256,
)
from evaluation.checkpoints import atomic_json
from evaluation.protocol import verify

GUIDE = """# Independent evidence annotation

This packet contains 100 prespecified development questions and their complete
source tables and linked passages. It contains no model prediction, route,
reference answer, automatic answer trace, or retrieved candidate ranking.

Assign two different human reviewers. They work independently, without viewing
each other's labels or system outputs. Each reviewer copies their blank template
and records their identity, an independently derived answer, and canonical IDs
for a minimal sufficient evidence set. Include the row identifying the subject,
intermediate relationships, and the passage containing the requested property.
For table aggregates, cite all contributing rows and describe the operation.
Use additional alternative sets when multiple independent support chains work;
do not combine alternatives into a single mandatory chain. Explain ambiguity or
missing evidence. Leave complete=false when the answer cannot be established.

Rows use row:<table_id>:<zero-based index>. Passage IDs use /wiki/.... Read the
full supplied corpus, not a system's top-k evidence. Do not use current Wikipedia
or outside facts. Each question's corpus_file is relative to this packet.

After both reviews, a third person adjudicates disagreements using both completed
reviews and the source. Retain reviewer originals; report denominators, pairwise
set agreement and adjudication decisions. Validation checks IDs and independence
declarations only: it cannot prove entailment or genuine human independence.

The frozen evaluation protocol keeps gold-evidence precision/recall disabled. This is
a development annotation exercise, not held-out gold. A new explicitly reviewed
analysis protocol is required before using completed labels for gold claims.
Source fidelity and automatic trace overlap remain separately named proxies.
"""


def prepare(bundle, archive, output):
    protocol, _ = verify(bundle)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new annotation directory")
    if sha256(archive) != protocol["sources"]["corpus"]["archive_sha256"]:
        raise ValueError("Annotation corpus is not pinned")
    questions = select_questions(
        read_json(bundle / "development/questions.json"),
        100,
        "heterorag-evaluation-v1:audit",
    )
    output.mkdir(parents=True, exist_ok=True)
    template = {}
    prefix = f"WikiTables-WithLinks-{CORPUS_REVISION}/"
    with zipfile.ZipFile(archive) as source:
        for q in questions:
            tid = q["table_id"]
            table = json.loads(source.read(prefix + "tables_tok/" + tid + ".json"))
            passages = json.loads(source.read(prefix + "request_tok/" + tid + ".json"))
            corpus_file = "corpus/" + object_hash(tid) + ".json"
            atomic_json(
                output / corpus_file,
                {
                    "table_id": tid,
                    "table": table,
                    "row_ids": [f"row:{tid}:{i}" for i in range(len(table["data"]))],
                    "passages": passages,
                },
            )
            template[q["question_id"]] = {
                "question": q,
                "corpus_file": corpus_file,
                "reviewer": "",
                "independent": False,
                "complete": False,
                "answer": "",
                "evidence_sets": [],
                "rationale": "",
                "uncertainties": "",
            }
    for reviewer in ("reviewer-a", "reviewer-b"):
        atomic_json(output / (reviewer + "-template.json"), template)
    (output / "GUIDE.md").write_text(GUIDE, encoding="utf-8")
    manifest = {
        "bundle_id": protocol["bundle_id"],
        "cohort": "development",
        "questions": list(template),
        "corpus_sha256": sha256(archive),
        "inputs": {
            p.relative_to(output).as_posix(): sha256(p)
            for p in output.rglob("*")
            if p.is_file()
        },
        "status": "unannotated",
        "gold_claims_enabled": False,
    }
    manifest["packet_id"] = object_hash(manifest)
    atomic_json(output / "packet.json", manifest)
    return manifest


def validate(packet, review_a, review_b):
    manifest = read_json(packet / "packet.json")
    if (
        object_hash({k: v for k, v in manifest.items() if k != "packet_id"})
        != manifest["packet_id"]
    ):
        raise ValueError("Annotation packet checksum mismatch")
    for relative, digest in manifest["inputs"].items():
        path = (packet / relative).resolve()
        if not path.is_relative_to(packet.resolve()) or sha256(path) != digest:
            raise ValueError("Frozen annotation packet changed")
    template = read_json(packet / "reviewer-a-template.json")
    reviews = [read_json(review_a), read_json(review_b)]
    reviewers = []
    for review in reviews:
        if set(review) != set(template):
            raise ValueError("Annotation question set differs")
        names = set()
        for qid, record in review.items():
            original = template[qid]
            if (
                record["question"] != original["question"]
                or record["corpus_file"] != original["corpus_file"]
            ):
                raise ValueError("Annotation question or source changed")
            name = record["reviewer"].strip().casefold()
            if not name or record["independent"] is not True:
                raise ValueError("Named independent human reviewers are required")
            names.add(name)
            corpus = read_json(packet / original["corpus_file"])
            allowed = set(corpus["row_ids"]) | set(corpus["passages"])
            if record["complete"] is True:
                if (
                    not record["answer"].strip()
                    or not record["rationale"].strip()
                    or not record["evidence_sets"]
                ):
                    raise ValueError(
                        "Completed annotation is missing an answer, rationale, or support"
                    )
                for ids in record["evidence_sets"]:
                    if (
                        not isinstance(ids, list)
                        or not ids
                        or len(ids) != len(set(ids))
                        or not set(ids) <= allowed
                    ):
                        raise ValueError("Invalid canonical evidence set")
        reviewers.append(names)
    if reviewers[0] & reviewers[1]:
        raise ValueError("The two independent reviews share a reviewer identity")
    paired = []
    for qid in template:
        a, b = (review[qid] for review in reviews)
        if a["complete"] is True and b["complete"] is True:
            aset = set().union(*map(set, a["evidence_sets"]))
            bset = set().union(*map(set, b["evidence_sets"]))
            paired.append(
                {
                    "question_id": qid,
                    "union_evidence_jaccard": len(aset & bset) / len(aset | bset),
                    "requires_adjudication": a["evidence_sets"] != b["evidence_sets"]
                    or a["answer"] != b["answer"],
                }
            )
    return {
        "packet_id": manifest["packet_id"],
        "reviews_sha256": [sha256(review_a), sha256(review_b)],
        "independently_completed_pairs": len(paired),
        "agreement": paired,
        "gold_claims_enabled": False,
        "adjudication_required": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate"))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--review-a", type=Path)
    parser.add_argument("--review-b", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.bundle, args.archive, args.packet)
        print({"questions": len(result["questions"]), "status": result["status"]})
    else:
        if not args.output or args.output.exists():
            raise ValueError("Supply a new --output for the validation receipt")
        atomic_json(args.output, validate(args.packet, args.review_a, args.review_b))


if __name__ == "__main__":
    main()
