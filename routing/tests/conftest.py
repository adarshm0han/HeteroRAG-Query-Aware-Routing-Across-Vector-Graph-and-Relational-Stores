"""Tests may use local Ollama, but can never contact paid APIs or cloud databases."""

import socket

import pytest


@pytest.fixture(autouse=True)
def forbid_external_connections(monkeypatch):
    original = socket.socket.connect

    def local_only(sock, address):
        if not isinstance(address, tuple) or address[0] not in ("127.0.0.1", "::1"):
            raise AssertionError("Test attempted an external network connection")
        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_only)
