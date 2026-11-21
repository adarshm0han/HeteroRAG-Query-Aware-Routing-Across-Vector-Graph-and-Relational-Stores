"""Bounded Decisions-only HTTP client. No Responses or chat-completions endpoints."""

import hashlib
import json
import urllib.request
from time import perf_counter

from baseline.common import safe_error

from .failures import failure_details

ENDPOINT = "https://api.openai.com/v1/decisions"


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        raise ValueError("Decisions requests must not redirect")


class DecisionsClient:
    def __init__(
        self,
        api_key,
        *,
        allow_paid=False,
        max_requests=1,
        journal=None,
        capture_response=False,
    ):
        if not allow_paid:
            raise ValueError(
                "Live Decisions calls are billed; explicit opt-in is required"
            )
        if not api_key or max_requests < 1:
            raise ValueError("Decisions requires a key and a positive request limit")
        self.api_key = api_key
        self.max_requests = max_requests
        self.events = []
        self.journal = journal
        self.capture_response = capture_response
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), NoRedirects()
        )

    def create(self, payload):
        if len(self.events) >= self.max_requests:
            raise ValueError("Decisions request limit reached")
        body = json.dumps(payload).encode()
        if len(body) > 16000:
            raise ValueError("Decisions routing request exceeds 16 KB")
        event = {
            "endpoint": "decisions",
            "status": "started",
            "request_sha256": hashlib.sha256(body).hexdigest(),
        }
        self.events.append(event)
        if self.journal is not None:
            self.journal(self.events)
        started = perf_counter()
        try:
            request = urllib.request.Request(
                ENDPOINT,
                data=body,
                headers={
                    "Authorization": "Bearer " + self.api_key,
                    "Content-Type": "application/json",
                },
            )
            # No automatic retries: an interrupted call may already be billed.
            with self.opener.open(request, timeout=60) as response:
                parsed = json.load(response)
                event["request_id"] = response.headers.get("x-request-id")
            event.update(status="completed", usage=parsed.get("usage", {}))
            if self.capture_response:
                event["response"] = parsed
            return parsed
        except Exception as exc:
            event.update(
                status="failed",
                error=safe_error(exc),
                billing="unknown",
                failure_details=failure_details(exc),
            )
            raise
        finally:
            event["seconds"] = perf_counter() - started
            if self.journal is not None:
                self.journal(self.events)
