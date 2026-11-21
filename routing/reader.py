"""Local extraction followed by an independent full-question evidence check."""

import json
from time import perf_counter

from baseline.answers import validate_answer

from .answer_context import frame_context
from .answer_validation import (  # noqa: F401 -- public helpers
    check_explicit_property,
    resolve_quote,
    restore_requested_unit,
)
from .evidence_answer import (
    DRAFT_INSTRUCTIONS,
    REVIEW_INSTRUCTIONS,
    response_schema,
    validate_response,
)

VERSION = "full_question_evidence_check_v5_2"


class CheckedReader:
    def __init__(self, reader):
        self.reader = reader

    def _request(self, instructions, content, schema):
        response = self.reader.request(
            "chat",
            {
                "model": self.reader.model,
                "stream": False,
                "think": False,
                "messages": [
                    {"role": "system", "content": instructions},
                    {
                        "role": "user",
                        "content": json.dumps(content, ensure_ascii=False),
                    },
                ],
                "format": schema,
                "options": {
                    "temperature": 0,
                    "seed": 42,
                    "num_ctx": 16384,
                    "num_predict": 768,
                },
            },
        )
        attempt = {
            "usage": {
                "input_tokens": response.get("prompt_eval_count", 0),
                "output_tokens": response.get("eval_count", 0),
            },
            "local_timings_ns": {
                k: response.get(k)
                for k in (
                    "total_duration",
                    "load_duration",
                    "prompt_eval_duration",
                    "eval_duration",
                )
            },
            "prompt_cached_tokens": response.get("prompt_eval_cached_count", 0),
        }
        try:
            if (
                not response.get("done")
                or response.get("done_reason") == "length"
                or response.get("prompt_eval_count", 0) >= 16000
            ):
                raise ValueError("incomplete_answer_step")
            attempt["response"] = json.loads(response["message"]["content"])
        except (ValueError, KeyError, TypeError) as exc:
            attempt.update(status="rejected", reason=str(exc))
        return attempt

    def generate(self, context, evidence):
        started = perf_counter()
        header, _, _, excerpts = frame_context(context, evidence)
        content = {**header, "sources": list(excerpts.values())}
        schema = response_schema(excerpts)
        attempts = []
        answer = {"answer": "", "citations": [], "abstain": True, "support_quotes": []}
        draft = {}
        # One draft and an independent review. Invalid review output gets one
        # bounded correction; transport failures propagate instead of retrying.
        for step in range(3):
            attempt = self._request(
                DRAFT_INSTRUCTIONS if step == 0 else REVIEW_INSTRUCTIONS,
                content,
                schema,
            )
            attempt["step"] = "draft" if step == 0 else "review"
            if "response" in attempt:
                try:
                    candidate = validate_response(
                        attempt["response"], excerpts, header["question"]
                    )
                    validate_answer(
                        {k: candidate[k] for k in ("answer", "citations")}, evidence
                    )
                    candidate = restore_requested_unit(candidate, header["question"])
                    check_explicit_property(header["question"], candidate, evidence)
                    attempt["status"] = "validated"
                    if step > 0:
                        answer = candidate
                except (ValueError, TypeError, KeyError, SyntaxError) as exc:
                    attempt.update(status="rejected", reason=str(exc))
            attempts.append(attempt)
            if step == 0:
                draft = attempt.get("response", {})
                if not isinstance(draft, dict):
                    draft = {"invalid_response": draft}
                content["proposal"] = draft
            elif attempt["status"] == "validated":
                break
            else:
                content["proposal"] = attempt.get("response", {})
            if attempt.get("reason"):
                content["validation_error"] = attempt["reason"]
        last = attempts[-1]
        return {
            **answer,
            "model": self.reader.model,
            "model_digest": self.reader.metadata["digest"],
            "local_inference": True,
            "api_cost_usd": 0,
            "seconds": perf_counter() - started,
            "draft": draft,
            "answer_plan": {
                "plan": None,
                "status": "no_column_precommitment",
                "attempts": [],
            },
            "answer_check": {
                "version": VERSION,
                "status": last["status"],
                "reason": last.get("reason"),
                "response": last.get("response"),
                "attempts": attempts,
                "verdict": "insufficient"
                if answer["abstain"]
                else "accepted"
                if answer["answer"] == draft.get("answer")
                else "revised",
            },
            "local_generation_calls": len(attempts),
            "usage": {
                k: sum(a["usage"][k] for a in attempts)
                for k in ("input_tokens", "output_tokens")
            },
            "local_timings_ns": {
                k: sum(a["local_timings_ns"][k] for a in attempts)
                if all(a["local_timings_ns"][k] is not None for a in attempts)
                else None
                for k in attempts[0]["local_timings_ns"]
            },
            "prompt_cached_tokens": sum(a["prompt_cached_tokens"] for a in attempts),
        }
