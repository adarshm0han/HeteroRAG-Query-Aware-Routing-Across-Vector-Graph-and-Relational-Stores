import json

import pytest


def test_default_runner_blocks_paid_generation_before_loading_credentials(
    monkeypatch, tmp_path
):
    from baseline import runner

    monkeypatch.setattr("sys.argv", ["baseline", "--output", str(tmp_path / "run")])

    def forbidden(*args):
        raise AssertionError(
            "Credentials must not be loaded before paid-generation opt-in"
        )

    monkeypatch.setattr(runner, "settings", forbidden)
    with pytest.raises(ValueError, match="Paid API generation is disabled"):
        runner.main()


def test_local_reader_rejects_cloud_models(monkeypatch):
    from baseline.local_reader import LocalReader

    monkeypatch.setattr(
        LocalReader,
        "request",
        lambda self, path: {
            "models": [{"name": "test:cloud", "remote_host": "external"}]
        },
    )
    with pytest.raises(ValueError, match="already installed local model"):
        LocalReader("test:cloud")


def test_local_reader_only_uses_loopback_and_disables_proxies(monkeypatch):
    from baseline import local_reader as local

    observed = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(
                {
                    "models": [
                        {
                            "name": "local:test",
                            "digest": "123",
                            "size": 1,
                            "details": {},
                        }
                    ]
                }
            ).encode()

    class Opener:
        def open(self, request, timeout):
            observed.append(request.full_url)
            return Response()

    def build(handler, redirects):
        assert handler.proxies == {}
        return Opener()

    monkeypatch.setattr(local.urllib.request, "build_opener", build)
    reader = local.LocalReader("local:test")
    assert observed == [
        "http://127.0.0.1:11434/api/tags",
        "http://127.0.0.1:11434/api/version",
    ]
    with pytest.raises(ValueError, match="Unsupported local inference operation"):
        reader.request("pull")


def test_local_reader_disables_thinking_and_preserves_failed_call_usage(monkeypatch):
    from baseline.answers import AnswerValidationError
    from baseline.local_reader import LocalReader

    seen = []

    def request(self, path, payload=None):
        if path == "tags":
            return {
                "models": [
                    {"name": "local:test", "digest": "abc", "size": 1, "details": {}}
                ]
            }
        if path == "version":
            return {"version": "test"}
        seen.append(payload)
        return {
            "done": True,
            "done_reason": "stop",
            "message": {"content": json.dumps({"answer": "Alpha", "citations": []})},
            "prompt_eval_count": 12,
            "eval_count": 4,
        }

    monkeypatch.setattr(LocalReader, "request", request)
    reader = LocalReader("local:test")
    with pytest.raises(AnswerValidationError) as failure:
        reader.generate("Only supplied evidence", [{"id": "E1"}])
    assert seen[0]["think"] is False
    assert seen[0]["options"]["temperature"] == 0
    assert failure.value.metadata["usage"] == {"input_tokens": 12, "output_tokens": 4}
    assert failure.value.metadata["api_cost_usd"] == 0


def test_comparison_rejects_different_question_sets(tmp_path):
    from baseline.common import write_json
    from baseline.compare import compare_runs

    left, right = tmp_path / "left", tmp_path / "right"
    for index, run in enumerate((left, right)):
        write_json(run / "manifest.json", {})
        write_json(run / "summary.json", {})
        write_json(run / "questions.json", [{"question_id": str(index)}])
        write_json(run / "references.json", {})
    with pytest.raises(ValueError, match="identical questions"):
        compare_runs(left, right, tmp_path / "comparison")
