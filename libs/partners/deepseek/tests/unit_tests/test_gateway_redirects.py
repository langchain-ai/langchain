"""Tests for gateway redirect isolation through the default HTTP clients."""

import socket
from contextlib import nullcontext
from typing import Any

import pytest
from langchain_core.utils._gateway import langsmith_gateway
from langchain_openai._compat import httpx
from langchain_openai.chat_models.base import OpenAIConnectionError

from langchain_deepseek import ChatDeepSeek


def _assert_gateway_rejected(
    error: BaseException | None, requests: list[Any], mode: str
) -> None:
    while error is not None and not isinstance(error, ValueError):
        error = error.__cause__
    assert isinstance(error, ValueError)
    assert "Cross-origin redirects" in str(error)
    gateway_requests = 2
    assert len(requests) == gateway_requests
    assert all(r.headers["authorization"] == "Bearer gateway-secret" for r in requests)
    if mode == "context":
        assert all(r.headers["x-service-key"] == "service-secret" for r in requests)


@pytest.mark.parametrize("mode", ["context", "empty", "env", "provider"])
@pytest.mark.parametrize("async_call", [False, True])
async def test_redirects_stay_on_gateway_origin(
    monkeypatch: pytest.MonkeyPatch, mode: str, *, async_call: bool
) -> None:
    """Block credential forwarding while preserving same-origin redirects."""

    def no_network(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError

    monkeypatch.setattr(socket.socket, "connect", no_network)
    for name in (
        "LANGSMITH_GATEWAY",
        "LANGSMITH_GATEWAY_API_KEY",
        "LANGSMITH_API_KEY",
        "DEEPSEEK_API_BASE",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    if mode == "env":
        monkeypatch.setenv("LANGSMITH_GATEWAY", "https://gateway.test")
        monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "gateway-secret")
    context = (
        langsmith_gateway(
            api_key="gateway-secret",
            base_url="https://gateway.test",
            headers={"X-Service-Key": "service-secret"} if mode == "context" else {},
        )
        if mode in {"context", "empty"}
        else nullcontext()
    )
    kwargs: dict[str, Any] = {"model": "test-model", "max_retries": 0}
    if mode == "provider":
        kwargs.update(base_url="https://gateway.test", api_key="provider-secret")
    with context:
        model = ChatDeepSeek(**kwargs)

    requests: list[Any] = []
    gateway_requests = 2
    all_requests = 3

    def handle(request: Any) -> Any:
        requests.append(request)
        locations = {1: "/same-origin", gateway_requests: "https://other.test/steal"}
        if location := locations.get(len(requests)):
            return httpx.Response(307, headers={"Location": location})
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
        )

    client = (model.root_async_client if async_call else model.root_client)._client
    monkeypatch.setattr(client, "_transport", httpx.MockTransport(handle))
    monkeypatch.setattr(client, "_mounts", {})

    async def invoke() -> None:
        if async_call:
            await model.ainvoke("hello")
        else:
            model.invoke("hello")

    if mode == "provider":
        await invoke()
        assert len(requests) == all_requests
    else:
        with pytest.raises((ValueError, OpenAIConnectionError)) as exc_info:
            await invoke()
        _assert_gateway_rejected(exc_info.value, requests, mode)
    assert all(r.method == "POST" for r in requests)
