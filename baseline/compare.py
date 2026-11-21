"""Compare two completed local-reader runs over exactly matching saved evidence."""

import argparse
import collections
from pathlib import Path

from .common import answer_scores, read_json, write_json
from .report import cluster_interval


def compare_runs(left, right, output):
    runs = [left, right]
    manifests = [read_json(run / "manifest.json") for run in runs]
    summaries = [read_json(run / "summary.json") for run in runs]
    questions = [read_json(run / "questions.json") for run in runs]
    references = [read_json(run / "references.json") for run in runs]
    if questions[0] != questions[1] or references[0] != references[1]:
        raise ValueError(
            "Readers must be compared on identical questions and references"
        )
    for key in (
        "plans",
        "retrieval_sha256",
        "source_run_signature",
        "context_tokens",
        "temperature",
        "max_output_tokens",
        "thinking",
        "local_context_window",
    ):
        if manifests[0]["config"][key] != manifests[1]["config"][key]:
            raise ValueError("Reader runs differ in controlled setting: " + key)
    for summary in summaries:
        if not summary["complete"] or summary["api_requests"]:
            raise ValueError(
                "Comparison requires complete local runs with no paid API calls"
            )
    plans = manifests[0]["config"]["plans"]
    differences = collections.defaultdict(list)
    examples = {"right_only_correct": [], "left_only_correct": []}
    for question in questions[0]:
        qid = question["question_id"]
        retrievals = [read_json(run / "retrieval" / (qid + ".json")) for run in runs]
        if retrievals[0]["contexts"] != retrievals[1]["contexts"]:
            raise ValueError("Models received different evidence for " + qid)
        for plan in plans:
            answers = [
                read_json(run / "answers" / f"{qid}.{plan}.json") for run in runs
            ]
            scores = [
                answer_scores(references[0][qid]["answer"], answer["answer"])
                if answer["ok"]
                else {"em": 0.0, "f1": 0.0}
                for answer in answers
            ]
            differences[plan].append(scores[1]["f1"] - scores[0]["f1"])
            if plan == "all_stores" and scores[0]["em"] != scores[1]["em"]:
                category = (
                    "right_only_correct" if scores[1]["em"] else "left_only_correct"
                )
                examples[category].append(
                    {
                        **question,
                        "reference": references[0][qid]["answer"],
                        "left_answer": answers[0].get("answer"),
                        "right_answer": answers[1].get("answer"),
                    }
                )
    models = [manifest["config"]["model"] for manifest in manifests]
    table_ids = [question["table_id"] for question in questions[0]]
    result = {
        "questions": len(questions[0]),
        "unique_tables": len(set(table_ids)),
        "models": models,
        "model_details": [m["local_model"] for m in manifests],
        "run_signatures": [m["signature"] for m in manifests],
        "same_questions_and_contexts": True,
        "graph_covered_questions": summaries[0]["graph_covered_questions"],
        "paid_api_calls": 0,
        "new_database_calls": 0,
        "results": {
            model: summary["results"] for model, summary in zip(models, summaries)
        },
        "paired_f1_differences": {
            plan: {
                "right_minus_left_points": 100 * sum(values) / len(values),
                "table_cluster_95ci_points": [
                    100 * v for v in cluster_interval(values, table_ids)
                ],
            }
            for plan, values in differences.items()
        },
        "all_stores_examples": examples,
        "limitations": [
            "50-question training pilot; one greedy output per question, plan, and model.",
            "Both readers use identical saved evidence; no new live database retrieval is timed.",
            "Models use their native tokenizers and quantization, a 16K window, and thinking disabled.",
            "Latency includes local inference and may benefit from prompt caching; it is not end-to-end system latency.",
            "Invalid answer contracts receive zero EM/F1; citation IDs are checked, not entailment.",
            "Exploratory table-cluster confidence intervals; no evidence of equivalence from overlapping intervals.",
        ],
    }
    write_json(output / "comparison.json", result)
    lines = [
        "# Local reader comparison",
        "",
        f"{len(questions[0])} training questions, {len(set(table_ids))} tables, four retrieval plans, and two local readers.",
        f"Graph evidence was available for {result['graph_covered_questions']} questions. Both readers received identical saved contexts.",
        "No paid inference or new database calls were made.",
        "",
        "| Retrieval plan | Reader | EM % | F1 % | Invalid outputs | Median seconds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for plan in plans:
        for model, summary in zip(models, summaries):
            row = summary["results"][plan]
            lines.append(
                f"| {plan} | {model} | {row['em_percent']:.1f} | {row['f1_percent']:.1f} | "
                f"{row['failed']} | {row['reader_latency_p50_seconds']:.2f} |"
            )
    lines += [
        "",
        f"## Paired F1 difference: {models[1]} minus {models[0]}",
        "",
        "| Plan | Difference (points) | Exploratory 95% interval |",
        "|---|---:|---|",
    ]
    for plan, delta in result["paired_f1_differences"].items():
        low, high = delta["table_cluster_95ci_points"]
        lines.append(
            f"| {plan} | {delta['right_minus_left_points']:+.1f} | {low:+.1f} to {high:+.1f} |"
        )
    lines += ["", "## Interpretation limits", ""] + [
        "- " + value for value in result["limitations"]
    ]
    lines += [
        "",
        "The JSON report includes graph-coverage slices, model digests, and examples where only one reader was correct.",
        "",
    ]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(
        args.left.resolve(), args.right.resolve(), args.output.resolve()
    )
    print(
        f"Compared {result['questions']} identical questions: {', '.join(result['models'])}"
    )
    print(args.output / "comparison.md")


if __name__ == "__main__":
    main()
