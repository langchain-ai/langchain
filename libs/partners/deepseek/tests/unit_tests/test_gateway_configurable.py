"""Tests for configurable gateway construction state."""

import json
from contextlib import nullcontext
from typing import Any, cast

import httpx
import pytest
from langchain_core.load import dumpd
from langchain_core.runnables import ConfigurableField
from langchain_core.utils._gateway import langsmith_gateway
from pydantic import BaseModel, SecretStr

from langchain_deepseek import ChatDeepSeek
from langchain_deepseek.chat_models import DEFAULT_API_BASE, DEFAULT_BETA_API_BASE


@pytest.mark.parametrize(
    "initial_headers", [None, {}, {"X-Service-Key": "original-secret"}]
)
@pytest.mark.parametrize("rebuild", [False, True])
def test_configurable_preserves_gateway_context(
    initial_headers: dict[str, str] | None, *, rebuild: bool
) -> None:
    """Reconstruction keeps the original context and paired clients."""
    initial = (
        langsmith_gateway(
            api_key="original-api-secret",
            base_url="https://original.test",
            headers=initial_headers,
        )
        if initial_headers is not None
        else nullcontext()
    )
    with initial:
        model = ChatDeepSeek(
            model="test-model",
            api_key=SecretStr("user-key"),
            base_url="https://provider.test",
        )
    original_clients = model.root_client, model.root_async_client
    original_url, original_key = model.api_base, model.api_key
    fields = {"temperature": ConfigurableField(id="temperature")}
    config: dict[str, Any] = {"temperature": 0.4}
    if rebuild:
        for name in ("client", "root_client", "async_client", "root_async_client"):
            fields[name] = ConfigurableField(id=name)
            config[name] = None
    configurable = model.configurable_fields(**fields)
    with langsmith_gateway(
        api_key="other-api-secret",
        base_url="https://other.test",
        headers={"X-Other": "other-secret"},
    ):

        class Wrapper(BaseModel):
            wrapped: ChatDeepSeek

        assert Wrapper(wrapped=model).wrapped is model
        prepared, _ = cast("Any", configurable).prepare({"configurable": config})
        assert ChatDeepSeek(model="test-model")._langsmith_gateway_headers == {
            "X-Other": "other-secret"
        }
    assert (model.root_client, model.root_async_client) == original_clients
    assert (model.api_base, model.api_key) == (original_url, original_key)
    assert (prepared.api_base, prepared.api_key) == (original_url, original_key)
    assert prepared._langsmith_gateway_headers == (initial_headers or {})
    for client in (prepared.root_client, prepared.root_async_client):
        assert str(client.base_url).startswith(
            "https://original.test"
            if initial_headers is not None
            else "https://provider.test"
        )
        assert client.default_headers.get("X-Service-Key") == (
            initial_headers or {}
        ).get("X-Service-Key")
        assert "X-Other" not in client.default_headers
    headers = {"x-service-key": "forged", "X-Custom": "kept"}
    assert prepared._get_request_payload("hello", extra_headers=headers)[
        "extra_headers"
    ] == ({"X-Custom": "kept"} if initial_headers else headers)
    assert headers == {"x-service-key": "forged", "X-Custom": "kept"}
    for secret in ("original-api-secret", "original-secret"):
        assert secret not in json.dumps(dumpd(prepared))
        assert secret not in str(prepared.model_dump())


@pytest.mark.parametrize("gateway", ["foreign", "same-origin", "none"])
@pytest.mark.parametrize("wrapper", ["bind_tools", "with_structured_output"])
def test_strict_public_paths_keep_credentials_at_the_selected_origin(
    gateway: str, wrapper: str
) -> None:
    """Strict mode only selects beta when the provider origin was selected."""
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chat-1",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "Answer",
                                        "arguments": '{"value":"ok"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    context = (
        langsmith_gateway(
            api_key="gateway-key",
            base_url="https://gateway.test"
            if gateway == "foreign"
            else "https://api.deepseek.com",
            headers={"X-Service-Key": "gateway-header"},
        )
        if gateway != "none"
        else nullcontext()
    )
    schema = {
        "title": "Answer",
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        with context:
            model = ChatDeepSeek(
                model="test-model",
                api_key=SecretStr("provider-key"),
                base_url=DEFAULT_API_BASE,
                http_client=http_client,
            )
        if wrapper == "bind_tools":
            model.bind_tools([schema], strict=True).invoke("hello")
        else:
            model.with_structured_output(schema, strict=True).invoke("hello")
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url).startswith(
        "https://gateway.test/deepseek/v1"
        if gateway == "foreign"
        else DEFAULT_BETA_API_BASE
    )
    assert request.headers["authorization"] == (
        "Bearer provider-key" if gateway == "none" else "Bearer gateway-key"
    )
    assert request.headers.get("x-service-key") == (
        None if gateway == "none" else "gateway-header"
    )
