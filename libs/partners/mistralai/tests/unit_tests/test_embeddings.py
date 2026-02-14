import asyncio
import os
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from httpx import Response
from pydantic import SecretStr

from langchain_mistralai import MistralAIEmbeddings

os.environ["MISTRAL_API_KEY"] = "foo"


def test_mistral_init() -> None:
    for model in [
        MistralAIEmbeddings(model="mistral-embed", mistral_api_key="test"),  # type: ignore[call-arg]
        MistralAIEmbeddings(model="mistral-embed", api_key="test"),  # type: ignore[arg-type]
    ]:
        assert model.model == "mistral-embed"
        assert cast("SecretStr", model.mistral_api_key).get_secret_value() == "test"


def test_embeddings_uses_max_concurrent_requests_for_limits() -> None:
    """Ensure HTTPX client limits respect max_concurrent_requests configuration."""

    with (
        patch("langchain_mistralai.embeddings.httpx.Client") as mock_client,
        patch(
            "langchain_mistralai.embeddings.httpx.AsyncClient",
        ) as mock_async_client,
    ):
        mock_client.return_value = MagicMock()
        mock_async_client.return_value = MagicMock()

        max_concurrent = 5
        MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key="test",  # type: ignore[call-arg]
            max_concurrent_requests=max_concurrent,
        )

        _, client_kwargs = mock_client.call_args
        _, async_kwargs = mock_async_client.call_args

        client_limits = client_kwargs["limits"]
        async_limits = async_kwargs["limits"]

        assert client_limits.max_connections == max_concurrent
        assert client_limits.max_keepalive_connections == max_concurrent
        assert async_limits.max_connections == max_concurrent
        assert async_limits.max_keepalive_connections == max_concurrent


@pytest.mark.asyncio
async def test_aembed_documents_respects_max_concurrent_requests() -> None:
    """Verify that concurrent async embedding requests are bounded by a semaphore."""

    embed = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key="test",  # type: ignore[call-arg]
        max_concurrent_requests=1,
    )

    class _SimpleTokenizer:
        @staticmethod
        def encode_batch(texts: list[str]) -> list[list[str]]:
            return [list(text) for text in texts]

    embed.tokenizer = _SimpleTokenizer()  # type: ignore[assignment]

    active = 0
    max_active = 0

    async def fake_post(*args: object, **kwargs: object) -> Response:  # type: ignore[override]
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        response = MagicMock(spec=Response)
        response.json.return_value = {
            "data": [
                {"embedding": [0.0]}
                for _ in kwargs["json"]["input"]  # type: ignore[index]
            ]
        }
        active -= 1
        return response

    embed.async_client.post = fake_post  # type: ignore[assignment]

    texts = ["a", "b", "c"]
    result = await embed.aembed_documents(texts)

    assert len(result) == len(texts)
    assert max_active <= 1
