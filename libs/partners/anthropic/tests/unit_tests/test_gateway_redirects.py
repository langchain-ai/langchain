"""Tests for gateway redirect isolation through the default HTTP clients."""

import socket
from contextlib import nullcontext
from typing import Any, cast

import pytest
from langchain_core.runnables import ConfigurableField
from langchain_core.utils._gateway import langsmith_gateway

from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import AnthropicConnectionError
from tests.unit_tests._httpx_compat import httpx


@pytest.mark.parametrize(
    "mode", ["context", "empty", "env", "env-reconstructed", "provider"]
)
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
        "ANTHROPIC_API_BASE",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    if mode in {"env", "env-reconstructed"}:
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
        model = ChatAnthropic(**kwargs)
    if mode == "env-reconstructed":
        monkeypatch.delenv("LANGSMITH_GATEWAY")
        monkeypatch.delenv("LANGSMITH_GATEWAY_API_KEY")
        configurable = model.configurable_fields(
            temperature=ConfigurableField(id="temperature")
        )
        model, _ = cast(Any, configurable).prepare(
            {"configurable": {"temperature": 0.4}}
        )

    requests: list[Any] = []
    gateway_requests = 2
    all_requests = 3

    def handle(request: Any) -> Any:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(307, headers={"Location": "/same-origin"})
        if len(requests) == gateway_requests:
            return httpx.Response(307, headers={"Location": "https://other.test/steal"})
        return httpx.Response(
            200,
            json={
                "id": "test",
                "type": "message",
                "model": "test-model",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = (model._async_client if async_call else model._client)._client
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
        with pytest.raises((ValueError, AnthropicConnectionError)) as exc_info:
            await invoke()
        error: BaseException | None = exc_info.value
        while error is not None and not isinstance(error, ValueError):
            error = error.__cause__
        assert isinstance(error, ValueError)
        assert "Cross-origin redirects" in str(error)
        assert len(requests) == gateway_requests
        assert requests[0].headers["x-api-key"] == "gateway-secret"
        assert requests[1].headers["x-api-key"] == "gateway-secret"
        if mode == "context":
            assert all(r.headers["x-service-key"] == "service-secret" for r in requests)
    assert all(r.method == "POST" for r in requests)
