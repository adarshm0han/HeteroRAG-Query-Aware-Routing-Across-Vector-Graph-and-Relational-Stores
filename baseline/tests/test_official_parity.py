"""Cross-check metric behavior against the locally pinned upstream evaluator."""

import ast
import collections
import random
import re
import string
import zipfile
from pathlib import Path

import pytest

from baseline.common import answer_scores


def test_scores_match_pinned_official_evaluator():
    archive_path = Path(__file__).resolve().parents[3] / "data/questions.zip"
    if not archive_path.exists():
        pytest.skip("Pinned dataset archive is optional for offline unit tests")
    with zipfile.ZipFile(archive_path) as archive:
        name = next(n for n in archive.namelist() if n.endswith("/evaluate_script.py"))
        tree = ast.parse(archive.read(name).decode())
    names = {"normalize_answer", "get_tokens", "compute_exact", "compute_f1"}
    definitions = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ],
        type_ignores=[],
    )
    namespace = {"collections": collections, "re": re, "string": string}
    exec(compile(definitions, "pinned_hybridqa_evaluator", "exec"), namespace)  # noqa: S102 - Isolated official scoring functions.
    rng = random.Random(57)
    vocabulary = [
        "the",
        "A",
        "name",
        "New",
        "York",
        "19.2",
        "2013-14",
        "2013â€“14",
        "VÃ¤sternorrland",
        "red",
        "red",
        "",
    ]
    for _ in range(1000):
        gold = " ".join(rng.choices(vocabulary, k=rng.randrange(6)))
        pred = " ".join(rng.choices(vocabulary, k=rng.randrange(6)))
        score = answer_scores(gold, pred)
        assert score["em"] == namespace["compute_exact"](gold, pred)
        assert score["f1"] == pytest.approx(namespace["compute_f1"](gold, pred))
