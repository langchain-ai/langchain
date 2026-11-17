"""Tests for construction-time gateway credentials on Anthropic requests."""

import json
from collections.abc import AsyncIterator
from contextlib import nullcontext
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.utils._gateway import langsmith_gateway
from pydantic import BaseModel, SecretStr

from langchain_anthropic import ChatAnthropic
from tests.unit_tests._httpx_compat import httpx


def _response(request: Any) -> Any:
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "model": "test-model",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


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
