import json
from types import SimpleNamespace

import pytest

from routing import cli, decisions
from routing.router import DecisionsRouter, RecordedRouter, confidence_weights

from .fixtures import QUESTION, FakeDecisions


def test_paid_calls_disabled_before_transport_creation(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Transport must not be created")

    monkeypatch.setattr(decisions.urllib.request, "build_opener", forbidden)
    with pytest.raises(ValueError, match="explicit opt-in"):
        decisions.DecisionsClient("unused")


def test_cli_default_blocks_paid_calls_before_reading_files():
    with pytest.raises(SystemExit) as error:
        cli.main(["--questions", "missing.json", "--output", "never-created"])
    assert error.value.code == 2


def test_only_decisions_endpoint_and_no_retries_after_uncertain_failure(monkeypatch):
    requests = []

    class Transport:
        def open(self, request, timeout):
            requests.append(request)
            raise TimeoutError("Simulated failure")

    monkeypatch.setattr(
        decisions.urllib.request, "build_opener", lambda *args: Transport()
    )
    client = decisions.DecisionsClient("fake-key", allow_paid=True, max_requests=1)
    with pytest.raises(TimeoutError):
        DecisionsRouter(client).choose(QUESTION)
    with pytest.raises(ValueError, match="request limit"):
        DecisionsRouter(client).choose(QUESTION)
    assert len(requests) == 1
    assert requests[0].full_url == decisions.ENDPOINT
    assert json.loads(requests[0].data)["model"] == "gpt-6-luna"
    assert client.events[0]["billing"] == "unknown"
    assert "fake-key" not in json.dumps(client.events)


@pytest.mark.parametrize(
    "mutation",
    [
        "refusal",
        "nan",
        "duplicate",
        "unknown",
        "missing_answer",
        "bool_probability",
        "bool_confidence",
    ],
)
def test_invalid_api_responses_are_rejected(mutation):
    response = FakeDecisions("postgres").create(
        {
            "questions": [
                {
                    "choices": [
                        {"value": "postgres"},
                        {"value": "pinecone"},
                        {"value": "postgres_pinecone"},
                    ]
                }
            ]
        }
    )
    answer = response["answers"][0]
    if mutation == "refusal":
        answer["type"] = "refusal"
    elif mutation == "nan":
        answer["confidence"] = float("nan")
    elif mutation == "duplicate":
        answer["probabilities"][1]["value"] = "postgres"
    elif mutation == "unknown":
        answer["choice"] = "all"
    elif mutation == "bool_probability":
        answer["probabilities"][0]["probability"] = True
    elif mutation == "bool_confidence":
        answer["confidence"] = True
    else:
        response["answers"] = []
    with pytest.raises(ValueError):
        DecisionsRouter(SimpleNamespace(create=lambda payload: response)).choose(
            QUESTION, False
        )


def test_query_dependent_weights_use_same_decisions_request():
    client = FakeDecisions("postgres_pinecone")
    result = DecisionsRouter(client).choose(QUESTION)
    assert len(client.payloads) == 1
    assert len(client.payloads[0]["questions"]) == 1
    assert result["option"] == "postgres_pinecone"
    assert result["store_weights"] == {"postgres": 0.5, "pinecone": 0.5}
    assert "store_scores" not in result
    once = RecordedRouter([{"question": QUESTION, "decision": result}]).choose(QUESTION)
    twice = RecordedRouter([{"question": QUESTION, "decision": once}]).choose(QUESTION)
    assert twice["original_routing_seconds"] == result["seconds"]
    assert result["weight_method"] == "route_marginals_v1"
    assert result["store_marginals"] == {"postgres": 1.0, "neo4j": 0.0, "pinecone": 1.0}


def test_marginal_weights_match_worked_example_and_keep_only_selected_stores():
    result = confidence_weights(
        "postgres_pinecone",
        [
            {"value": "postgres_pinecone", "probability": 0.6},
            {"value": "postgres", "probability": 0.3},
            {"value": "pinecone", "probability": 0.1},
        ],
    )
    assert result["store_weights"] == pytest.approx(
        {"postgres": 0.5625, "pinecone": 0.4375}
    )
    assert result["store_marginals"]["neo4j"] == 0


def test_selected_store_with_rounded_zero_probability_has_finite_positive_weight():
    result = confidence_weights(
        "all",
        [
            {"value": "all", "probability": 0.0},
            {"value": "postgres", "probability": 1.0},
        ],
    )
    assert all(0 < value <= 1 for value in result["store_weights"].values())
    assert sum(result["store_weights"].values()) == pytest.approx(1)


@pytest.mark.parametrize(
    "numbers", [[0, 0], [0.8, 0.8], [float("nan"), 0.5], [-0.1, 1.1], [True, 0]]
)
def test_bad_weight_distributions_are_rejected(numbers):
    with pytest.raises(ValueError):
        confidence_weights(
            "postgres",
            [
                {"value": "postgres", "probability": numbers[0]},
                {"value": "pinecone", "probability": numbers[1]},
            ],
        )


def test_replay_holds_routes_fixed_and_rejects_question_changes():
    decision = DecisionsRouter(FakeDecisions("postgres")).choose(QUESTION)
    decision["store_weights"] = {"postgres": 0.2}
    decision["store_scores"] = {"postgres": 0.4}
    router = RecordedRouter([{"question": QUESTION, "decision": decision}])
    result = router.choose(QUESTION)
    assert result["option"] == decision["option"] == "postgres_pinecone"
    assert result["method"] == "recorded_decisions"
    assert result["api_requests"] == 0
    assert result["seconds"] == 0
    assert result["store_weights"] == {"postgres": 0.5, "pinecone": 0.5}
    assert "store_scores" not in result
    with pytest.raises(ValueError, match="different question"):
        router.choose({**QUESTION, "question": "Changed question"})


def test_journal_records_attempt_before_transport_and_preserves_usage(monkeypatch):
    import copy

    journal = []

    class Response:
        def __init__(self):
            self.headers = {"x-request-id": "synthetic-request"}

        def __enter__(self):
            assert journal[0][0]["status"] == "started"
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"usage": {"input_tokens": 20}}).encode()

    monkeypatch.setattr(
        decisions.urllib.request,
        "build_opener",
        lambda *args: SimpleNamespace(open=lambda *args, **kwargs: Response()),
    )
    client = decisions.DecisionsClient(
        "fake",
        allow_paid=True,
        journal=lambda events: journal.append(copy.deepcopy(events)),
    )
    client.create({})
    assert journal[-1][0]["status"] == "completed"
    assert journal[-1][0]["usage"]["input_tokens"] == 20
