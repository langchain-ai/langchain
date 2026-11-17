"""Tests for gateway header precedence on OpenAI requests."""

import json
from contextlib import AsyncExitStack, nullcontext
from typing import Any, Self

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.load import dumpd
from langchain_core.utils._gateway import langsmith_gateway
from pydantic import BaseModel, SecretStr, model_validator

from langchain_openai import ChatOpenAI
from langchain_openai._compat import httpx


class _CaptureStart(BaseCallbackHandler):
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: Any, **kwargs: Any
    ) -> None:
        self.calls.append((serialized, kwargs))


def _response(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    responses_api = request.url.path.endswith("/responses")
    if body.get("stream"):
        if responses_api:
            event = {
                "type": "response.output_text.delta",
                "sequence_number": 0,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "hello",
            }
            content = (
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\ndata: [DONE]\n\n"
            )
        else:
            event = {
                "id": "chat_1",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": "hello"}}],
            }
            content = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
        return httpx.Response(
            200, text=content, headers={"content-type": "text/event-stream"}
        )
    if responses_api:
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "model": "test-model",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "hello", "annotations": []}
                        ],
                    }
                ],
            },
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


@pytest.mark.parametrize("gateway", [False, True])
@pytest.mark.parametrize("responses_api", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("header_source", ["model", "invoke"])
async def test_gateway_headers_at_request_boundary(
    gateway: bool,
    responses_api: bool,
    stream: bool,
    async_call: bool,
    header_source: str,
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(request)

    headers: dict[str, str | None] = {"x-service-key": "forged", "X-CUSTOM": "kept"}
    if gateway:
        headers["X-SERVICE-KEY"] = None
    capture = _CaptureStart()
    context = (
        langsmith_gateway(
            api_key="gateway-api-secret",
            headers={"X-Service-Key": "gateway-header-secret"},
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
            llm = ChatOpenAI(
                model="test-model",
                api_key=SecretStr("user-key"),
                use_responses_api=responses_api,
                model_kwargs={"extra_headers": headers}
                if header_source == "model"
                else {},
                http_client=sync_client,
                http_async_client=async_client,
            )
        kwargs: dict[str, Any] = (
            {"extra_headers": headers} if header_source == "invoke" else {}
        )
        result: Any
        if async_call and stream:
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
        elif stream:
            result = list(
                llm.stream("hello", config={"callbacks": [capture]}, **kwargs)
            )
        else:
            result = llm.invoke("hello", config={"callbacks": [capture]}, **kwargs)

    assert result
    assert len(requests) == 1
    assert requests[0].headers["x-service-key"] == (
        "gateway-header-secret" if gateway else "forged"
    )
    assert requests[0].headers["x-custom"] == "kept"
    assert headers["x-service-key"] == "forged"
    assert capture.calls
    for secret in ("gateway-header-secret", "gateway-api-secret"):
        assert secret not in requests[0].content.decode()
        assert secret not in json.dumps(dumpd(llm))
        assert secret not in str(llm.model_dump())
        assert secret not in str(capture.calls)


@pytest.mark.parametrize("initial_gateway", [False, True])
def test_gateway_headers_are_fixed_when_model_is_revalidated(
    initial_gateway: bool,
) -> None:
    context = (
        langsmith_gateway(
            api_key="gateway-api-secret", headers={"X-Service-Key": "original-secret"}
        )
        if initial_gateway
        else nullcontext()
    )
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(request)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with context:
            llm = ChatOpenAI(
                model="test-model", api_key=SecretStr("user-key"), http_client=client
            )
        original_client = llm.root_client
        headers = {"x-service-key": "forged", "x-other-key": "kept"}
        with langsmith_gateway(
            api_key="other-api-secret", headers={"X-Other-Key": "other-secret"}
        ):
            assert ChatOpenAI.model_validate(llm) is llm

            class Wrapper(BaseModel):
                model: ChatOpenAI

            Wrapper(model=llm)
            llm.invoke("hello", extra_headers=headers)
        assert llm.root_client is original_client
        payload = llm._get_request_payload("hello", extra_headers=headers)
        assert payload["extra_headers"] == (
            {"x-other-key": "kept"} if initial_gateway else headers
        )
        assert "original-secret" not in str(payload)
        assert "other-secret" not in str(payload)
        llm.invoke("hello", extra_headers=headers)
    assert len(requests) == 2
    for request in requests:
        assert request.headers["x-service-key"] == (
            "original-secret" if initial_gateway else "forged"
        )
        assert request.headers["x-other-key"] == "kept"


def test_subclass_with_own_environment_validator() -> None:
    class OtherOpenAI(ChatOpenAI):
        @model_validator(mode="after")
        def validate_environment(self) -> Self:
            return self

    llm = OtherOpenAI(model="test-model", api_key=SecretStr("user-key"))
    payload = llm._get_request_payload("hello", extra_headers={"X-Custom": "kept"})
    assert payload["extra_headers"] == {"X-Custom": "kept"}
