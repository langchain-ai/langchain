import asyncio
from unittest.mock import Mock

import pytest

from langchain_ollama.chat_models import ChatOllama


def test_max_retries_field_present():
    model = ChatOllama(model="llama3", max_retries=3)
    # model_fields exists on Pydantic models
    assert "max_retries" in ChatOllama.model_fields
    assert getattr(model, "max_retries") == 3


def test_sync_retries_on_transient_error():
    llm = ChatOllama(model="llama3", max_retries=2)

    class DummyClient:
        def __init__(self):
            self.count = 0

        def chat(self, **params):
            self.count += 1
            if self.count < 2:
                raise RuntimeError("transient")
            return "ok"

    llm._client = DummyClient()
    res = llm._client_chat_with_retries(stream=False, messages=[])
    assert res == "ok"
    assert llm._client.count == 2


@pytest.mark.asyncio
async def test_async_retries_on_transient_error():
    llm = ChatOllama(model="llama3", max_retries=2)

    class DummyAsyncClient:
        def __init__(self):
            self.count = 0

        async def chat(self, **params):
            self.count += 1
            if self.count < 2:
                raise RuntimeError("transient-async")
            return "ok-async"

    llm._async_client = DummyAsyncClient()
    res = await llm._async_client_chat_with_retries(stream=False, messages=[])
    assert res == "ok-async"
    assert llm._async_client.count == 2
