"""Tests for gateway header precedence on Groq requests."""

import json
from contextlib import AsyncExitStack, nullcontext
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.load import dumpd
from langchain_core.runnables import ConfigurableField
from langchain_core.utils._gateway import langsmith_gateway
from pydantic import BaseModel, SecretStr

from langchain_groq import ChatGroq


class _CaptureStart(BaseCallbackHandler):
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: Any, **kwargs: Any
    ) -> None:
        self.calls.append((serialized, kwargs))


def _response(request: httpx.Request) -> httpx.Response:
    if json.loads(request.content).get("stream"):
        event = {
            "id": "chat_1",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "hello"}}],
        }
        return httpx.Response(
            200,
            text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )
    return httpx.Response(
        200,
        json={
            "id": "chat_1",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hello"},
                }
            ],
        },
    )


@pytest.fixture(autouse=True)
def _gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LANGSMITH_GATEWAY", "LANGSMITH_GATEWAY_API_KEY", "GROQ_API_BASE"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("gateway", [False, True])
@pytest.mark.parametrize("mode", ["invoke", "stream", "configured_stream"])
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("header_source", ["model", "invoke", "both"])
async def test_gateway_headers_at_request_boundary(
    mode: str, header_source: str, *, gateway: bool, async_call: bool
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(request)

    headers: dict[str, str | None] = {"x-service-key": "forged", "X-CUSTOM": "kept"}
    if gateway:
        headers["X-SERVICE-KEY"] = None
        headers["x-remove-key"] = None
    original_headers = headers.copy()
    capture = _CaptureStart()
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
    async with AsyncExitStack() as stack:
        sync_client = stack.enter_context(
            httpx.Client(transport=httpx.MockTransport(handle))
        )
        async_client = await stack.enter_async_context(
            httpx.AsyncClient(transport=httpx.MockTransport(handle))
        )
        with context:
            llm = ChatGroq(
                model="test-model",
                api_key=SecretStr("user-key"),
                default_headers={
                    "x-service-key": "default-forged",
                    "X-Default": "kept",
                },
                model_kwargs={"extra_headers": headers}
                if header_source == "model"
                else (
                    {"extra_headers": {"X-Service-Key": "model-forged"}}
                    if header_source == "both"
                    else {}
                ),
                streaming=mode == "configured_stream",
                http_client=sync_client,
                http_async_client=async_client,
            )
        kwargs: dict[str, Any] = (
            {"extra_headers": headers} if header_source != "model" else {}
        )
        sdk_client = llm.async_client if async_call else llm.client
        create = stack.enter_context(
            patch.object(sdk_client, "create", wraps=sdk_client.create)
        )
        result: Any
        if async_call and mode == "stream":
            result = [
                chunk
                async for chunk in llm.astream(
                    "hello", config={"callbacks": [capture]}, **kwargs
                )
            ]
        elif async_call:
            result = await llm.ainvoke(
                "hello", config={"callbacks": [capture]}, **kwargs
            )
        elif mode == "stream":
            result = list(
                llm.stream("hello", config={"callbacks": [capture]}, **kwargs)
            )
        else:
            result = llm.invoke("hello", config={"callbacks": [capture]}, **kwargs)

    assert result
    assert len(requests) == 1
    request = requests[0]
    assert request.headers["x-service-key"] == (
        "gateway-header-secret" if gateway else "forged"
    )
    assert request.headers.get("x-remove-key") == (
        "gateway-remove-secret" if gateway else None
    )
    assert request.headers["x-custom"] == "kept"
    assert request.headers["x-default"] == "kept"
    assert headers == original_headers
    assert create.call_count == 1
    sdk_params = create.call_args.kwargs
    assert sdk_params["extra_headers"] == ({"X-CUSTOM": "kept"} if gateway else headers)
    assert capture.calls
    for secret in (
        "gateway-header-secret",
        "gateway-remove-secret",
        "gateway-api-secret",
    ):
        assert secret not in request.content.decode()
        assert secret not in str(sdk_params)
        assert secret not in str(llm._get_invocation_params(**kwargs))
        assert secret not in json.dumps(dumpd(llm))
        assert secret not in str(llm.model_dump())
        assert secret not in str(capture.calls)


@pytest.mark.parametrize("initial_gateway", ["none", "empty", "headers"])
@pytest.mark.parametrize("async_call", [False, True])
async def test_gateway_headers_are_fixed_when_model_is_revalidated(
    initial_gateway: str, *, async_call: bool
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(request)

    context = (
        langsmith_gateway(
            api_key="original-api-secret",
            headers={"X-Service-Key": "original-secret"}
            if initial_gateway == "headers"
            else {},
        )
        if initial_gateway != "none"
        else nullcontext()
    )
    async with AsyncExitStack() as stack:
        sync_client = stack.enter_context(
            httpx.Client(transport=httpx.MockTransport(handle))
        )
        async_client = await stack.enter_async_context(
            httpx.AsyncClient(transport=httpx.MockTransport(handle))
        )
        with context:
            llm = ChatGroq(
                model="test-model",
                api_key=SecretStr("user-key"),
                http_client=sync_client,
                http_async_client=async_client,
            )
        original_clients = (llm.client, llm.async_client)
        headers = {"x-service-key": "forged", "x-other-key": "kept"}
        with langsmith_gateway(
            api_key="other-api-secret", headers={"X-Other-Key": "other-secret"}
        ):
            assert ChatGroq.model_validate(llm) is llm

            class Wrapper(BaseModel):
                model: ChatGroq

            assert Wrapper(model=llm).model is llm
            if async_call:
                await llm.ainvoke("hello", extra_headers=headers)
            else:
                llm.invoke("hello", extra_headers=headers)
        assert Wrapper(model=llm).model is llm
        assert (llm.client, llm.async_client) == original_clients
        if async_call:
            await llm.ainvoke("hello", extra_headers=headers)
        else:
            llm.invoke("hello", extra_headers=headers)
    assert len(requests) == 2
    for request in requests:
        assert request.headers["x-service-key"] == (
            "original-secret" if initial_gateway == "headers" else "forged"
        )
        assert request.headers["x-other-key"] == "kept"
        assert "other-secret" not in str(request.headers)


@pytest.mark.parametrize("initial_gateway", ["none", "empty", "headers"])
@pytest.mark.parametrize("rebuild_client", [False, True])
@pytest.mark.parametrize("async_call", [False, True])
async def test_configurable_preserves_gateway_context(
    initial_gateway: str, *, rebuild_client: bool, async_call: bool
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(request)

    context = (
        langsmith_gateway(
            api_key="original-api-secret",
            base_url="https://original.test",
            headers={"X-Service-Key": "original-secret"}
            if initial_gateway == "headers"
            else {},
        )
        if initial_gateway != "none"
        else nullcontext()
    )
    async with AsyncExitStack() as stack:
        sync_http = stack.enter_context(
            httpx.Client(transport=httpx.MockTransport(handle))
        )
        async_http = await stack.enter_async_context(
            httpx.AsyncClient(transport=httpx.MockTransport(handle))
        )
        with context:
            llm = ChatGroq(
                model="test-model",
                api_key=SecretStr("user-key"),
                base_url="https://provider.test",
                http_client=sync_http,
                http_async_client=async_http,
            )
        original_clients = (llm.client, llm.async_client)
        fields = {"temperature": ConfigurableField(id="temperature")}
        config: dict[str, Any] = {"temperature": 0.4}
        if rebuild_client:
            for name in ("client", "async_client"):
                fields[name] = ConfigurableField(id=name)
                config[name] = None
        runnable = llm.configurable_fields(**fields)
        with langsmith_gateway(
            api_key="other-api-secret",
            base_url="https://other.test",
            headers={"X-Other": "other-secret"},
        ):
            prepared, _ = cast("Any", runnable).prepare({"configurable": config})
            if async_call:
                await prepared.ainvoke(
                    "hello", extra_headers={"x-service-key": "forged"}
                )
            else:
                prepared.invoke("hello", extra_headers={"x-service-key": "forged"})
        assert (llm.client, llm.async_client) == original_clients
        for secret in ("original-api-secret", "original-secret"):
            assert secret not in json.dumps(dumpd(prepared))
            assert secret not in str(prepared.model_dump())
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url).startswith(
        "https://provider.test"
        if initial_gateway == "none"
        else "https://original.test"
    )
    assert request.headers["authorization"] == (
        "Bearer user-key" if initial_gateway == "none" else "Bearer original-api-secret"
    )
    assert request.headers["x-service-key"] == (
        "original-secret" if initial_gateway == "headers" else "forged"
    )
    assert "x-other" not in request.headers
