"""Tests for construction-time gateway credentials on Anthropic requests."""

import json
from collections.abc import AsyncIterator
from contextlib import nullcontext
from typing import Any, cast
from unittest.mock import patch

import anthropic
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.load import dumpd
from langchain_core.messages import HumanMessage
from langchain_core.runnables import ConfigurableField
from langchain_core.utils._gateway import langsmith_gateway
from pydantic import BaseModel, SecretStr

from langchain_anthropic import ChatAnthropic
from tests.unit_tests._httpx_compat import httpx


class _CaptureStart(BaseCallbackHandler):
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: Any, **kwargs: Any
    ) -> None:
        self.calls.append((serialized, kwargs))


def _response(request: Any) -> Any:
    if request.url.path.endswith("/count_tokens"):
        return httpx.Response(200, json={"input_tokens": 7})
    message = {
        "id": "msg_1",
        "type": "message",
        "model": "test-model",
        "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    if json.loads(request.content).get("stream"):
        events: list[dict[str, Any]] = [
            {
                "type": "message_start",
                "message": {**message, "content": [], "stop_reason": None},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
        content = "".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
        )
        return httpx.Response(
            200, text=content, headers={"content-type": "text/event-stream"}
        )
    return httpx.Response(200, json=message)


@pytest.fixture
async def outbound_requests() -> AsyncIterator[list[Any]]:
    requests: list[Any] = []

    def handle(request: Any) -> Any:
        requests.append(request)
        return _response(request)

    with httpx.Client(transport=httpx.MockTransport(handle)) as sync_client:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle)
        ) as async_client:
            with (
                patch(
                    "langchain_anthropic.chat_models._get_default_httpx_client",
                    return_value=sync_client,
                ),
                patch(
                    "langchain_anthropic.chat_models._get_default_async_httpx_client",
                    return_value=async_client,
                ),
            ):
                yield requests


@pytest.mark.parametrize("initial_gateway", ["none", "empty", "headers"])
@pytest.mark.parametrize("async_call", [False, True])
async def test_gateway_credentials_survive_nested_revalidation(
    initial_gateway: str, outbound_requests: list[Any], *, async_call: bool
) -> None:
    context = (
        langsmith_gateway(
            api_key="original-api-secret",
            base_url="https://original-gateway.test",
            headers={"X-Service-Key": "original-header-secret"}
            if initial_gateway == "headers"
            else {},
        )
        if initial_gateway != "none"
        else nullcontext()
    )
    with context:
        llm = ChatAnthropic(
            model="test-model",
            api_key=SecretStr("provider-key"),
            base_url="https://provider.test",
        )
    with langsmith_gateway(
        api_key="other-api-secret",
        base_url="https://other-gateway.test",
        headers={"X-Service-Key": "other-secret", "X-Other-Key": "other-secret"},
    ):

        class Wrapper(BaseModel):
            model: ChatAnthropic

        assert Wrapper(model=llm).model is llm
        assert {"_client_params", "_client", "_async_client"}.isdisjoint(llm.__dict__)
        if async_call:
            await llm.ainvoke("hello")
        else:
            llm.invoke("hello")
    assert Wrapper(model=llm).model is llm
    if async_call:
        await llm.ainvoke("hello")
    else:
        llm.invoke("hello")

    assert len(outbound_requests) == 2
    for request in outbound_requests:
        assert str(request.url) == (
            "https://provider.test/v1/messages"
            if initial_gateway == "none"
            else "https://original-gateway.test/anthropic/v1/messages"
        )
        assert request.headers["x-api-key"] == (
            "provider-key" if initial_gateway == "none" else "original-api-secret"
        )
        assert request.headers.get("x-service-key") == (
            "original-header-secret" if initial_gateway == "headers" else None
        )
        assert "x-other-key" not in request.headers
        assert "other-secret" not in json.dumps(dict(request.headers))


@pytest.mark.parametrize("gateway", [False, True])
@pytest.mark.parametrize("betas", [None, ["test-beta"]])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("header_source", ["model", "invoke"])
async def test_gateway_headers_at_request_boundary(
    outbound_requests: list[Any],
    betas: list[str] | None,
    header_source: str,
    *,
    gateway: bool,
    stream: bool,
    async_call: bool,
) -> None:
    headers: dict[str, Any] = {"x-service-key": "forged", "X-CUSTOM": "kept"}
    if gateway:
        headers["X-REMOVE-KEY"] = anthropic.Omit()
    original_headers = headers.copy()
    capture = _CaptureStart()
    context = (
        langsmith_gateway(
            api_key="gateway-api-secret",
            headers={
                "X-Service-Key": "gateway-header-secret",
                "X-Remove-Key": "gateway-header-secret",
            },
        )
        if gateway
        else nullcontext()
    )
    with context:
        llm = ChatAnthropic(
            model="test-model",
            api_key=SecretStr("provider-key"),
            betas=betas,
            default_headers={"x-service-key": "default-forged", "X-Default": "kept"},
            model_kwargs={"extra_headers": headers} if header_source == "model" else {},
        )
    kwargs: dict[str, Any] = (
        {"extra_headers": headers} if header_source == "invoke" else {}
    )
    payload = llm._get_request_payload([HumanMessage("hello")], **kwargs)
    result: Any
    if async_call and stream:
        result = [
            chunk
            async for chunk in llm.astream(
                "hello", config={"callbacks": [capture]}, **kwargs
            )
        ]
    elif async_call:
        result = await llm.ainvoke("hello", config={"callbacks": [capture]}, **kwargs)
    elif stream:
        result = list(llm.stream("hello", config={"callbacks": [capture]}, **kwargs))
    else:
        result = llm.invoke("hello", config={"callbacks": [capture]}, **kwargs)

    assert result
    assert len(outbound_requests) == 1
    request = outbound_requests[0]
    assert request.headers["x-service-key"] == (
        "gateway-header-secret" if gateway else "forged"
    )
    assert request.headers.get("x-remove-key") == (
        "gateway-header-secret" if gateway else None
    )
    assert request.headers["x-custom"] == "kept"
    assert request.headers["x-default"] == "kept"
    assert request.headers.get("anthropic-beta") == ("test-beta" if betas else None)
    assert headers == original_headers
    assert payload["extra_headers"] == ({"X-CUSTOM": "kept"} if gateway else headers)
    assert capture.calls
    for secret in ("gateway-header-secret", "gateway-api-secret"):
        assert secret not in request.content.decode()
        assert secret not in str(payload)
        assert secret not in json.dumps(dumpd(llm))
        assert secret not in str(llm.model_dump())
        assert secret not in str(capture.calls)


@pytest.mark.parametrize("initial_gateway", ["none", "empty", "headers"])
@pytest.mark.parametrize("later_gateway", [False, True])
@pytest.mark.parametrize("async_call", [False, True])
async def test_configurable_preserves_gateway_context(
    initial_gateway: str,
    outbound_requests: list[Any],
    *,
    later_gateway: bool,
    async_call: bool,
) -> None:
    initial = (
        langsmith_gateway(
            api_key="original-api-secret",
            base_url="https://original.test",
            headers={"X-Service-Key": "original-header-secret"}
            if initial_gateway == "headers"
            else {},
        )
        if initial_gateway != "none"
        else nullcontext()
    )
    later = (
        langsmith_gateway(
            api_key="other-api-secret",
            base_url="https://other.test",
            headers={"X-Other": "other-header-secret"},
        )
        if later_gateway
        else nullcontext()
    )
    with initial:
        llm = ChatAnthropic(
            model="test-model",
            api_key=SecretStr("provider-key"),
            base_url="https://provider.test",
        )
    original_url, original_key = llm.anthropic_api_url, llm.anthropic_api_key
    runnable = llm.configurable_fields(temperature=ConfigurableField(id="temperature"))
    headers = {"x-service-key": "forged", "X-Custom": "kept"}
    with later:
        prepared, _ = cast(Any, runnable).prepare(
            {"configurable": {"temperature": 0.4}}
        )
        assert prepared is not llm
        assert prepared.anthropic_api_url == original_url
        assert prepared.anthropic_api_key == original_key
        if async_call:
            await prepared.ainvoke("hello", extra_headers=headers)
        else:
            prepared.invoke("hello", extra_headers=headers)
        if later_gateway:
            assert ChatAnthropic(model="test-model")._langsmith_gateway_headers == {
                "X-Other": "other-header-secret"
            }
    assert llm.anthropic_api_url == original_url
    assert llm.anthropic_api_key == original_key
    assert {"_client", "_async_client", "_client_params"}.isdisjoint(llm.__dict__)
    assert llm._langsmith_gateway_headers == (
        {"X-Service-Key": "original-header-secret"}
        if initial_gateway == "headers"
        else {}
    )
    for secret in ("original-header-secret", "original-api-secret"):
        assert secret not in json.dumps(dumpd(prepared))
        assert secret not in str(prepared.model_dump())
    assert len(outbound_requests) == 1
    request = outbound_requests[0]
    assert str(request.url).startswith(
        "https://provider.test"
        if initial_gateway == "none"
        else "https://original.test"
    )
    assert request.headers["x-api-key"] == (
        "provider-key" if initial_gateway == "none" else "original-api-secret"
    )
    assert request.headers["x-service-key"] == (
        "original-header-secret" if initial_gateway == "headers" else "forged"
    )
    assert "x-other" not in request.headers
    assert headers == {"x-service-key": "forged", "X-Custom": "kept"}


@pytest.mark.parametrize("gateway", [False, True])
@pytest.mark.parametrize("betas", [None, ["test-beta"]])
async def test_token_count_gateway_header_precedence(
    outbound_requests: list[Any], betas: list[str] | None, *, gateway: bool
) -> None:
    headers: dict[str, Any] = {"x-service-key": "forged", "X-CUSTOM": "kept"}
    if gateway:
        headers["X-REMOVE-KEY"] = anthropic.Omit()
    original_headers = headers.copy()
    context = (
        langsmith_gateway(
            api_key="gateway-api-secret",
            headers={
                "X-Service-Key": "gateway-header-secret",
                "X-Remove-Key": "gateway-remove-secret",
            },
        )
        if gateway
        else nullcontext()
    )
    with context:
        llm = ChatAnthropic(
            model="test-model",
            api_key=SecretStr("provider-key"),
            betas=betas,
            default_headers={"x-service-key": "default-forged", "X-Default": "kept"},
        )

    assert (
        llm.get_num_tokens_from_messages([HumanMessage("hello")], extra_headers=headers)
        == 7
    )
    assert len(outbound_requests) == 1
    request = outbound_requests[0]
    assert request.url.path.endswith("/v1/messages/count_tokens")
    assert request.headers["x-service-key"] == (
        "gateway-header-secret" if gateway else "forged"
    )
    assert request.headers.get("x-remove-key") == (
        "gateway-remove-secret" if gateway else None
    )
    assert request.headers["x-custom"] == "kept"
    assert request.headers["x-default"] == "kept"
    if betas:
        assert "test-beta" in request.headers["anthropic-beta"].split(",")
    assert headers == original_headers
    for secret in (
        "gateway-header-secret",
        "gateway-remove-secret",
        "gateway-api-secret",
    ):
        assert secret not in request.content.decode()
        assert secret not in json.dumps(dumpd(llm))
        assert secret not in str(llm.model_dump())
