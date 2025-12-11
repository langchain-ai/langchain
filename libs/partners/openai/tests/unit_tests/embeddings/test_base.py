import os
from typing import Any
from unittest.mock import Mock, patch

import pytest
from pydantic import SecretStr

from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "foo"


def test_openai_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        OpenAIEmbeddings(model_kwargs={"model": "foo"})


def test_openai_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = OpenAIEmbeddings(foo="bar")  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}


def test_embed_documents_with_custom_chunk_size() -> None:
    embeddings = OpenAIEmbeddings(chunk_size=2)
    texts = ["text1", "text2", "text3", "text4"]
    custom_chunk_size = 3

    with patch.object(embeddings.client, "create") as mock_create:
        mock_create.side_effect = [
            {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]},
            {"data": [{"embedding": [0.5, 0.6]}, {"embedding": [0.7, 0.8]}]},
        ]

        result = embeddings.embed_documents(texts, chunk_size=custom_chunk_size)
        _, tokens, __, ___ = embeddings._tokenize(texts, custom_chunk_size)
        mock_create.call_args
        mock_create.assert_any_call(input=tokens[0:3], **embeddings._invocation_params)
        mock_create.assert_any_call(input=tokens[3:4], **embeddings._invocation_params)

    assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]


def test_embed_documents_with_custom_chunk_size_no_check_ctx_length() -> None:
    embeddings = OpenAIEmbeddings(chunk_size=2, check_embedding_ctx_length=False)
    texts = ["text1", "text2", "text3", "text4"]
    custom_chunk_size = 3

    with patch.object(embeddings.client, "create") as mock_create:
        mock_create.side_effect = [
            {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]},
            {"data": [{"embedding": [0.5, 0.6]}, {"embedding": [0.7, 0.8]}]},
        ]

        result = embeddings.embed_documents(texts, chunk_size=custom_chunk_size)

        mock_create.call_args
        mock_create.assert_any_call(input=texts[0:3], **embeddings._invocation_params)
        mock_create.assert_any_call(input=texts[3:4], **embeddings._invocation_params)

    assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]


def test_embed_with_kwargs() -> None:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", check_embedding_ctx_length=False
    )
    texts = ["text1", "text2"]
    with patch.object(embeddings.client, "create") as mock_create:
        mock_create.side_effect = [
            {"data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]}
        ]

        result = embeddings.embed_documents(texts, dimensions=3)
        mock_create.assert_any_call(
            input=texts, dimensions=3, **embeddings._invocation_params
        )

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


async def test_embed_with_kwargs_async() -> None:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        check_embedding_ctx_length=False,
        dimensions=4,  # also check that runtime kwargs take precedence
    )
    texts = ["text1", "text2"]
    with patch.object(embeddings.async_client, "create") as mock_create:
        mock_create.side_effect = [
            {"data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]}
        ]

        result = await embeddings.aembed_documents(texts, dimensions=3)
        client_kwargs = embeddings._invocation_params.copy()
        assert client_kwargs["dimensions"] == 4
        client_kwargs["dimensions"] = 3
        mock_create.assert_any_call(input=texts, **client_kwargs)

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_embeddings_respects_token_limit() -> None:
    """Test that embeddings respect the 300k token per request limit."""
    # Create embeddings instance
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", api_key=SecretStr("test-key")
    )

    call_counts = []

    def mock_create(**kwargs: Any) -> Mock:
        input_ = kwargs["input"]
        # Track how many tokens in this call
        if isinstance(input_, list):
            total_tokens = sum(
                len(t) if isinstance(t, list) else len(t.split()) for t in input_
            )
            call_counts.append(total_tokens)
            # Verify this call doesn't exceed limit
            assert total_tokens <= 300000, (
                f"Batch exceeded token limit: {total_tokens} tokens"
            )

        # Return mock response
        mock_response = Mock()
        mock_response.model_dump.return_value = {
            "data": [
                {"embedding": [0.1] * 1536}
                for _ in range(len(input_) if isinstance(input_, list) else 1)
            ]
        }
        return mock_response

    embeddings.client.create = mock_create

    # Create a scenario that would exceed 300k tokens in a single batch
    # with default chunk_size=1000
    # Simulate 500 texts with ~1000 tokens each = 500k tokens total
    large_texts = ["word " * 1000 for _ in range(500)]

    # This should not raise an error anymore
    embeddings.embed_documents(large_texts)

    # Verify we made multiple API calls to respect the limit
    assert len(call_counts) > 1, "Should have split into multiple batches"

    # Verify each call respected the limit
    for count in call_counts:
        assert count <= 300000, f"Batch exceeded limit: {count}"
