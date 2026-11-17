import json
from contextlib import nullcontext
from typing import Any, cast

import pytest
from langchain_core.load import dumpd
from langchain_core.runnables import ConfigurableField
from langchain_core.utils._gateway import langsmith_gateway
from pydantic import BaseModel, SecretStr

from langchain_xai import ChatXAI


@pytest.mark.parametrize(
    "initial_headers", [None, {}, {"X-Service-Key": "original-secret"}]
)
@pytest.mark.parametrize("rebuild", [False, True])
def test_configurable_preserves_gateway_context(
    initial_headers: dict[str, str] | None, *, rebuild: bool
) -> None:
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
        model = ChatXAI(
            model="test-model",
            api_key=SecretStr("user-key"),
            base_url="https://provider.test",
        )
    original_clients = model.root_client, model.root_async_client
    original_url, original_key = model.xai_api_base, model.xai_api_key
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
            wrapped: ChatXAI

        assert Wrapper(wrapped=model).wrapped is model
        prepared, _ = cast("Any", configurable).prepare({"configurable": config})
        assert ChatXAI(model="test-model")._langsmith_gateway_headers == {
            "X-Other": "other-secret"
        }
    assert (model.root_client, model.root_async_client) == original_clients
    assert (model.xai_api_base, model.xai_api_key) == (original_url, original_key)
    assert (prepared.xai_api_base, prepared.xai_api_key) == (original_url, original_key)
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
