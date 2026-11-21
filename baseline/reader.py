"""Bounded OpenAI Responses reader with an append-only usage ledger."""

import json
import threading
import time
import urllib.error
import urllib.request

from .answers import INSTRUCTIONS, SCHEMA, validate_answer


class APIError(RuntimeError):
    def __init__(self, status):
        super().__init__(f"OpenAI request failed (HTTP {status})")
        self.status = status


class BudgetExceeded(RuntimeError):
    pass


class Reader:
    def __init__(
        self, key, model, encoder, ledger, budget=5.0, max_output=1024, workers=4
    ):
        if model != "gpt-6-luna":
            raise ValueError(
                "Supply verified pricing before enabling another reader model"
            )
        self.key, self.model, self.encoder = key, model, encoder
        self.ledger, self.budget, self.max_output = ledger, budget, max_output
        self.lock = threading.Lock()
        self.reserved = 0.0
        self.spent = 0.0
        if ledger.exists():
            self.spent = sum(
                json.loads(line)["accounted_usd"]
                for line in ledger.read_text().splitlines()
                if line
            )

    @staticmethod
    def price(usage):
        # Standard global pricing checked 2026-11-16; cache writes are included in input_tokens.
        details = usage.get("input_tokens_details") or {}
        cached = details.get("cached_tokens", 0)
        writes = details.get("cache_write_tokens", 0)
        uncached = max(0, usage.get("input_tokens", 0) - cached - writes)
        return (
            uncached * 0.10
            + cached * 0.01
            + writes * 0.125
            + usage.get("output_tokens", 0) * 0.50
        ) / 1e6

    def generate(self, question_id, plan, context, evidence):
        payload = {
            "model": self.model,
            "instructions": INSTRUCTIONS,
            "input": context,
            "store": False,
            "reasoning": {"effort": "low"},
            "max_output_tokens": self.max_output,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "grounded_answer",
                    "strict": True,
                    "schema": SCHEMA,
                }
            },
        }
        expected_input = (
            len(self.encoder.encode(context + INSTRUCTIONS + json.dumps(SCHEMA))) + 256
        )
        upper = (expected_input * 0.125 + self.max_output * 0.5) / 1e6
        started = time.perf_counter()
        for attempt in range(1, 4):
            with self.lock:
                if self.spent + self.reserved + upper > self.budget:
                    raise BudgetExceeded(
                        "Configured API budget exhausted; completed work is resumable"
                    )
                self.reserved += upper
            response, failure, status = None, None, None
            try:
                request = urllib.request.Request(
                    "https://api.openai.com/v1/responses",
                    data=json.dumps(payload).encode(),
                    headers={
                        "Authorization": "Bearer " + self.key,
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(request, timeout=90) as stream:
                    response = json.load(stream)
                    status = 200
            except urllib.error.HTTPError as exc:
                status, failure = exc.code, APIError(exc.code)
                exc.close()
            except (urllib.error.URLError, TimeoutError, OSError):
                failure = RuntimeError(
                    "OpenAI transport error; details omitted to protect credentials"
                )
            usage = response.get("usage", {}) if response else {}
            # Unknown transport outcomes conservatively consume the reservation.
            cost = self.price(usage) if response else (upper if status is None else 0.0)
            record = {
                "question_id": question_id,
                "plan": plan,
                "attempt": attempt,
                "http_status": status,
                "response_id": response.get("id") if response else None,
                "model": response.get("model", self.model) if response else self.model,
                "usage": usage,
                "accounted_usd": cost,
                "cost_is_estimate": True,
                "unknown_transport_charge": status is None,
            }
            with self.lock:
                self.reserved -= upper
                self.spent += cost
                with self.ledger.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
            if response:
                break
            if (
                status in (401, 403, 404)
                or (status is not None and status < 500 and status != 429)
                or attempt == 3
            ):
                raise failure
            time.sleep(min(2**attempt, 8))
        if response.get("status") != "completed":
            raise RuntimeError(
                "Reader response was incomplete or failed; usage has been recorded"
            )
        text = "".join(
            c.get("text", "")
            for item in response.get("output", [])
            if item.get("type") == "message"
            for c in item.get("content", [])
            if c.get("type") == "output_text"
        )
        parsed = validate_answer(json.loads(text), evidence)
        return {
            **parsed,
            "seconds": time.perf_counter() - started,
            "usage": usage,
            "model": response.get("model"),
            "response_id": response.get("id"),
            "attempts": attempt,
        }
