import os
from typing import cast
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai.embeddings import (
    DEFAULT_MAX_TOKENS,
    DummyTokenizer,
    _is_retryable_error,
)

os.environ["MISTRAL_API_KEY"] = "foo"


@patch("langchain_mistralai.embeddings.Tokenizer.from_pretrained")
def test_mistral_init(mock_tokenizer: MagicMock) -> None:
    mock_tokenizer.side_effect = OSError(
        "No HuggingFace access"
    )  # Force DummyTokenizer
    for model in [
        MistralAIEmbeddings(model="mistral-embed", mistral_api_key="test"),  # type: ignore[call-arg]
        MistralAIEmbeddings(model="mistral-embed", api_key="test"),  # type: ignore[arg-type]
    ]:
        assert model.model == "mistral-embed"
        assert cast("SecretStr", model.mistral_api_key).get_secret_value() == "test"


@patch("langchain_mistralai.embeddings.Tokenizer.from_pretrained")
def test_max_tokens_configurable(mock_tokenizer: MagicMock) -> None:
    """Test that max_tokens parameter is configurable."""
    mock_tokenizer.side_effect = OSError("No HuggingFace access")
    model = MistralAIEmbeddings(model="mistral-embed", api_key="test", max_tokens=8000)  # type: ignore[arg-type]
    assert model.max_tokens == 8000


@patch("langchain_mistralai.embeddings.Tokenizer.from_pretrained")
def test_default_max_tokens(mock_tokenizer: MagicMock) -> None:
    """Test that default max_tokens is set correctly."""
    mock_tokenizer.side_effect = OSError("No HuggingFace access")
    model = MistralAIEmbeddings(model="mistral-embed", api_key="test")  # type: ignore[arg-type]
    assert model.max_tokens == DEFAULT_MAX_TOKENS


@patch("langchain_mistralai.embeddings.Tokenizer.from_pretrained")
def test_get_batches_normal_documents(mock_tokenizer: MagicMock) -> None:
    """Test batching with normal-sized documents."""
    mock_tokenizer.side_effect = OSError("No HuggingFace access")
    model = MistralAIEmbeddings(model="mistral-embed", api_key="test", max_tokens=100)  # type: ignore[arg-type]
    # Override the tokenizer to return specific token counts
    model.tokenizer = MagicMock()
    model.tokenizer.encode_batch.return_value = [
        list(range(30)),  # 30 tokens
        list(range(30)),  # 30 tokens
        list(range(30)),  # 30 tokens
        list(range(30)),  # 30 tokens
    ]

    texts = ["doc1", "doc2", "doc3", "doc4"]
    batches = list(model._get_batches(texts))

    # With max_tokens=100 and 95% safety margin (effective=95),
    # each batch can hold up to 3 documents of 30 tokens each (90 tokens)
    assert len(batches) == 2
    assert batches[0] == ["doc1", "doc2", "doc3"]
    assert batches[1] == ["doc4"]


@patch("langchain_mistralai.embeddings.Tokenizer.from_pretrained")
def test_get_batches_oversized_document_raises_error(mock_tokenizer: MagicMock) -> None:
    """Test that a document exceeding max tokens raises ValueError."""
    mock_tokenizer.side_effect = OSError("No HuggingFace access")
    model = MistralAIEmbeddings(model="mistral-embed", api_key="test", max_tokens=100)  # type: ignore[arg-type]
    # Override the tokenizer
    model.tokenizer = MagicMock()
    model.tokenizer.encode_batch.return_value = [
        list(range(200)),  # 200 tokens - exceeds 95 (100 * 0.95)
    ]

    texts = ["oversized_document"]

    with pytest.raises(ValueError) as exc_info:
        list(model._get_batches(texts))

    assert "Document at index 0 has 200 tokens" in str(exc_info.value)
    assert "exceeds the maximum" in str(exc_info.value)
    assert "split your document" in str(exc_info.value)


@patch("langchain_mistralai.embeddings.Tokenizer.from_pretrained")
def test_get_batches_second_document_oversized(mock_tokenizer: MagicMock) -> None:
    """Test that error includes correct index for oversized document."""
    mock_tokenizer.side_effect = OSError("No HuggingFace access")
    model = MistralAIEmbeddings(model="mistral-embed", api_key="test", max_tokens=100)  # type: ignore[arg-type]
    model.tokenizer = MagicMock()
    model.tokenizer.encode_batch.return_value = [
        list(range(30)),  # 30 tokens - OK
        list(range(200)),  # 200 tokens - exceeds limit
    ]

    texts = ["normal_doc", "oversized_document"]

    with pytest.raises(ValueError) as exc_info:
        list(model._get_batches(texts))

    assert "Document at index 1 has 200 tokens" in str(exc_info.value)


def test_is_retryable_error_timeout() -> None:
    """Test that timeout exceptions are retryable."""
    exc = httpx.TimeoutException("timeout")
    assert _is_retryable_error(exc) is True


def test_is_retryable_error_rate_limit() -> None:
    """Test that 429 errors are retryable."""
    response = MagicMock()
    response.status_code = 429
    exc = httpx.HTTPStatusError("rate limit", request=MagicMock(), response=response)
    assert _is_retryable_error(exc) is True


def test_is_retryable_error_server_error() -> None:
    """Test that 5xx errors are retryable."""
    for status_code in [500, 502, 503, 504]:
        response = MagicMock()
        response.status_code = status_code
        exc = httpx.HTTPStatusError(
            "server error", request=MagicMock(), response=response
        )
        assert _is_retryable_error(exc) is True


def test_is_retryable_error_bad_request_not_retryable() -> None:
    """Test that 400 errors are NOT retryable."""
    response = MagicMock()
    response.status_code = 400
    exc = httpx.HTTPStatusError("bad request", request=MagicMock(), response=response)
    assert _is_retryable_error(exc) is False


def test_is_retryable_error_other_4xx_not_retryable() -> None:
    """Test that other 4xx errors are NOT retryable."""
    for status_code in [401, 403, 404, 422]:
        response = MagicMock()
        response.status_code = status_code
        exc = httpx.HTTPStatusError(
            "client error", request=MagicMock(), response=response
        )
        assert _is_retryable_error(exc) is False


def test_is_retryable_error_other_exceptions() -> None:
    """Test that other exceptions are not retryable."""
    assert _is_retryable_error(ValueError("test")) is False
    assert _is_retryable_error(RuntimeError("test")) is False


def test_dummy_tokenizer() -> None:
    """Test that DummyTokenizer returns character lists."""
    tokenizer = DummyTokenizer()
    result = tokenizer.encode_batch(["hello", "world"])
    assert result == [["h", "e", "l", "l", "o"], ["w", "o", "r", "l", "d"]]
