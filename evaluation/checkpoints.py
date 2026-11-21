"""Crash-safe checkpoints and a study-wide, at-most-once paid-request ledger.

Checksums detect damaged/tampered checkpoints, not malicious rewriting of both
payload and checksum. os.replace + fsync protects process-crash recovery; disk or
power loss still requires backups. A lost paid response is never retried.
"""

import json
import os
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from baseline.common import object_hash, read_json
from routing.decisions import DecisionsClient


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save(path, value):
    atomic_json(path, {"sha256": object_hash(value), "value": value})


def load(path):
    envelope = read_json(path)
    if set(envelope) != {"sha256", "value"} or envelope["sha256"] != object_hash(
        envelope["value"]
    ):
        raise ValueError("Checkpoint checksum mismatch: " + str(path))
    return envelope["value"]


@contextmanager
def study_lock(root):
    """OS-owned lock; releases on process exit, including hard termination."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "study.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another process owns this evaluation") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class InterruptedStage(RuntimeError):
    """An external side effect may already have happened; never retry it."""


class Checkpoints:
    def __init__(self, root, identity):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        identity_path = root / "identity.json"
        if identity_path.exists():
            if load(identity_path) != identity:
                raise ValueError("Checkpoint belongs to another job")
        else:
            save(identity_path, identity)
        self.resumed = any(root.glob("stage-*.json"))

    def step(self, name, function, *, recover=False):
        path = self.root / ("stage-" + name + ".json")
        prior = load(path) if path.exists() else None
        if prior and prior["status"] == "completed":
            return prior["value"]
        if prior and not recover:
            raise InterruptedStage(
                "Interrupted " + name + "; external work not repeated"
            )
        history = (
            [
                *prior.get("history", []),
                {k: v for k, v in prior.items() if k != "history"},
            ]
            if prior
            else []
        )
        save(path, {"status": "started", "history": history})
        start = perf_counter()
        value = function()
        save(
            path,
            {
                "status": "completed",
                "value": value,
                "seconds": perf_counter() - start,
                "history": history,
            },
        )
        return value


class PaidLedger:
    """Use only while holding the study lock. Budget includes uncertain attempts."""

    def __init__(self, root, limit, allowed_keys):
        self.root = root
        self.limit = limit
        self.allowed_keys = set(allowed_keys)
        root.mkdir(parents=True, exist_ok=True)
        self.records = {p.stem: load(p) for p in root.glob("*.json")}
        if not set(self.records) <= self.allowed_keys or len(self.records) > limit:
            raise ValueError("Paid ledger exceeds the frozen identities or budget")
        for key, record in self.records.items():
            if record["identity"] != key:
                raise ValueError("Paid request identity mismatch")

    def client(self, identity, key, *, allow_paid, factory=DecisionsClient):
        return DurableClient(self, identity, key, allow_paid, factory)


class DurableClient:
    def __init__(self, ledger, identity, api_key, allow_paid, factory):
        self.ledger, self.identity = ledger, identity
        self.api_key, self.allow_paid, self.factory = api_key, allow_paid, factory
        self.events = []
        self.reused = False
        if identity in ledger.records:
            self.events = [ledger.records[identity]["event"]]

    def create(self, payload):
        if self.identity not in self.ledger.allowed_keys:
            raise ValueError("Request identity is outside the frozen schedule")
        digest = object_hash(payload)
        previous = self.ledger.records.get(self.identity)
        if previous:
            if previous["payload_sha256"] != digest:
                raise ValueError("Paid request payload changed on resume")
            self.events = [previous["event"]]
            self.reused = True
            event = previous["event"]
            if event.get("status") == "completed" and "response" in event:
                return event["response"]
            raise InterruptedStage(
                "Paid response unavailable; this request will never be repeated"
            )
        if not self.allow_paid:
            raise PermissionError("New paid requests require explicit opt-in")
        if len(self.ledger.records) >= self.ledger.limit:
            raise ValueError("Study-wide paid request budget exhausted")

        def journal(events):
            record = {
                "identity": self.identity,
                "payload_sha256": digest,
                "event": events[-1],
            }
            save(self.ledger.root / (self.identity + ".json"), record)
            self.ledger.records[self.identity] = record
            self.events = events

        client = self.factory(
            self.api_key,
            allow_paid=True,
            max_requests=1,
            journal=journal,
            capture_response=True,
        )
        return client.create(payload)
