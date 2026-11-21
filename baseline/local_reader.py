"""Inference through the local Ollama service; no cloud endpoints or credentials."""

import json
import urllib.request
from time import perf_counter

from .answers import INSTRUCTIONS, SCHEMA, AnswerValidationError, validate_answer


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        raise ValueError("The local model service must not redirect requests")


class LocalReader:
    """Only loopback traffic; no cloud model tags, downloads, keys, or proxy routing."""

    def __init__(self, model="qwen3.5:4b"):
        self.model = model
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirects()
        )
        installed = self.request("tags")["models"]
        matches = [m for m in installed if m["name"] == model]
        if (
            len(matches) != 1
            or "cloud" in model.lower()
            or matches[0].get("remote_host")
        ):
            raise ValueError(
                "Select an already installed local model; cloud models and downloads are disabled"
            )
        self.metadata = {
            key: matches[0][key] for key in ("name", "digest", "size", "details")
        }
        self.runtime = self.request("version")

    def request(self, path, payload=None):
        if path not in ("tags", "chat", "version"):
            raise ValueError("Unsupported local inference operation")
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/" + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with self.opener.open(request, timeout=300) as response:
            return json.load(response)

    def generate(self, context, evidence):
        started = perf_counter()
        response = self.request(
            "chat",
            {
                "model": self.model,
                "stream": False,
                "think": False,
                "messages": [
                    {"role": "system", "content": INSTRUCTIONS},
                    {"role": "user", "content": context},
                ],
                "format": SCHEMA,
                "options": {
                    "temperature": 0,
                    "seed": 42,
                    "num_ctx": 16384,
                    "num_predict": 256,
                },
            },
        )
        metadata = {
            "seconds": perf_counter() - started,
            "model": self.model,
            "model_digest": self.metadata["digest"],
            "local_inference": True,
            "api_cost_usd": 0,
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
            if not response.get("done") or response.get("done_reason") == "length":
                raise AnswerValidationError("output_limit_or_incomplete", None)
            if response.get("prompt_eval_count", 0) >= 16000:
                raise AnswerValidationError("context_limit_risk", None)
            parsed = validate_answer(
                json.loads(response["message"]["content"]), evidence
            )
        except (AnswerValidationError, json.JSONDecodeError) as exc:
            exc.metadata = metadata
            raise
        return {**parsed, **metadata}
