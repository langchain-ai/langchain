"""Test client utility functions."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from langchain_anthropic import ChatAnthropic
from langchain_anthropic._client_utils import (
    _AsyncHttpxClientWrapper,
    _get_default_async_httpx_client,
    _get_default_httpx_client,
    _SyncHttpxClientWrapper,
)

MODEL = "claude-sonnet-5-5"


def test_sync_client_without_proxy() -> None:
    """Test sync client creation without proxy."""
    client = _get_default_httpx_client(base_url="https://api.anthropic.com")

    # Should not have proxy configured
    assert not hasattr(client, "proxies") or client.proxies is None


def test_sync_client_with_proxy() -> None:
    """Test sync client creation with proxy."""
    proxy_url = "http://proxy.example.com:8080"
    client = _get_default_httpx_client(
        base_url="https://api.anthropic.com", anthropic_proxy=proxy_url
    )

    # Check internal _transport since httpx stores proxy configuration in the transport
    # layer
    transport = getattr(client, "_transport", None)
    assert transport is not None


def test_async_client_without_proxy() -> None:
    """Test async client creation without proxy."""
    client = _get_default_async_httpx_client(base_url="https://api.anthropic.com")

    assert not hasattr(client, "proxies") or client.proxies is None


def test_async_client_with_proxy() -> None:
    """Test async client creation with proxy."""
    proxy_url = "http://proxy.example.com:8080"
    client = _get_default_async_httpx_client(
        base_url="https://api.anthropic.com", anthropic_proxy=proxy_url
    )

    transport = getattr(client, "_transport", None)
    assert transport is not None


def test_client_proxy_none_value() -> None:
    """Test that explicitly passing None for proxy works correctly."""
    sync_client = _get_default_httpx_client(
        base_url="https://api.anthropic.com", anthropic_proxy=None
    )

    async_client = _get_default_async_httpx_client(
        base_url="https://api.anthropic.com", anthropic_proxy=None
    )

    # Both should be created successfully with None proxy
    assert sync_client is not None
    assert async_client is not None


def test_sync_client_wrapper_del_handles_uninitialized_client() -> None:
    """Test sync wrapper finalizer handles clients without initialized state."""
    client = _SyncHttpxClientWrapper.__new__(_SyncHttpxClientWrapper)

    client.__del__()


async def test_async_client_wrapper_del_handles_uninitialized_client() -> None:
    """Test async wrapper finalizer handles clients without initialized state."""
    client = _AsyncHttpxClientWrapper.__new__(_AsyncHttpxClientWrapper)

    client.__del__()


def test_async_client_not_shared_between_calls() -> None:
    """Async clients are per model; they must not be cached and shared."""
    first = _get_default_async_httpx_client(base_url="https://api.anthropic.com")
    second = _get_default_async_httpx_client(base_url="https://api.anthropic.com")
    assert first is not second


@pytest.mark.enable_socket
def test_new_model_per_event_loop_does_not_reuse_closed_loop_connection() -> None:
    """Regression: models must not share one async client across event loops.

    A shared client let a model created in a new event loop pick up a keep-alive
    connection opened in an earlier, closed loop and fail with
    `RuntimeError: Event loop is closed`. The SDK retries that on a fresh
    connection, so `max_retries=0` is needed to surface it.
    """
    body = json.dumps(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["content-length"]))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    ).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:

        async def invoke() -> Any:
            # A fresh model per loop, as in one test per event loop.
            llm = ChatAnthropic(
                model=MODEL, api_key="test", base_url=base_url, max_retries=0
            )
            return (await llm.ainvoke("hi")).content

        assert [asyncio.run(invoke()) for _ in range(3)] == ["hi", "hi", "hi"]
    finally:
        server.shutdown()
        server.server_close()
