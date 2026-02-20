"""Unit tests for SentenceTransformersTokenTextSplitter."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain_text_splitters import SentenceTransformersTokenTextSplitter


def _make_mock_model(max_seq_length: int = 512) -> MagicMock:
    """Create a mock SentenceTransformer model."""
    mock_model = MagicMock()
    mock_model.max_seq_length = max_seq_length
    mock_model.tokenizer = MagicMock()
    mock_model.tokenizer.encode.return_value = list(range(10))
    mock_model.tokenizer.decode.return_value = "decoded text"
    return mock_model


@pytest.fixture
def mock_sentence_transformer() -> MagicMock:
    """Patch SentenceTransformer so that import checks pass."""
    mock_cls = MagicMock()
    mock_cls.return_value = _make_mock_model()
    with (
        patch(
            "langchain_text_splitters.sentence_transformers.SentenceTransformer",
            mock_cls,
            create=True,
        ),
        patch(
            "langchain_text_splitters.sentence_transformers._HAS_SENTENCE_TRANSFORMERS",
            new=True,
        ),
    ):
        yield mock_cls


class TestSentenceTransformersTokenTextSplitter:
    """Tests for SentenceTransformersTokenTextSplitter model_kwargs support."""

    def test_model_kwargs_passed_to_sentence_transformer(
        self, mock_sentence_transformer: MagicMock
    ) -> None:
        """Test that model_kwargs are forwarded to the SentenceTransformer."""
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device": "cpu",
        }
        SentenceTransformersTokenTextSplitter(
            model_name="test-model",
            model_kwargs=model_kwargs,
        )

        mock_sentence_transformer.assert_called_once_with(
            "test-model", trust_remote_code=True, device="cpu"
        )

    def test_default_model_kwargs_is_none(
        self, mock_sentence_transformer: MagicMock
    ) -> None:
        """Test that no extra kwargs are passed when model_kwargs is None."""
        SentenceTransformersTokenTextSplitter(model_name="test-model")

        mock_sentence_transformer.assert_called_once_with("test-model")

    def test_empty_model_kwargs(self, mock_sentence_transformer: MagicMock) -> None:
        """Test that an empty model_kwargs dict passes no extra args."""
        SentenceTransformersTokenTextSplitter(
            model_name="test-model",
            model_kwargs={},
        )

        mock_sentence_transformer.assert_called_once_with("test-model")

    def test_split_text_with_model_kwargs(
        self, mock_sentence_transformer: MagicMock
    ) -> None:
        """Test that split_text works when model_kwargs are provided."""
        mock_model = _make_mock_model(max_seq_length=20)
        # encode returns token IDs including start/stop tokens
        mock_model.tokenizer.encode.return_value = [101, 1, 2, 3, 4, 5, 102]
        mock_model.tokenizer.decode.side_effect = lambda ids: f"chunk({ids})"
        mock_sentence_transformer.return_value = mock_model

        splitter = SentenceTransformersTokenTextSplitter(
            model_name="test-model",
            tokens_per_chunk=10,
            chunk_overlap=0,
            model_kwargs={"trust_remote_code": True},
        )

        result = splitter.split_text("some text")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_count_tokens_with_model_kwargs(
        self, mock_sentence_transformer: MagicMock
    ) -> None:
        """Test that count_tokens works when model_kwargs are provided."""
        mock_model = _make_mock_model()
        mock_model.tokenizer.encode.return_value = [101, 1, 2, 3, 102]
        mock_sentence_transformer.return_value = mock_model

        splitter = SentenceTransformersTokenTextSplitter(
            model_name="test-model",
            model_kwargs={"local_files_only": True},
        )

        token_count = splitter.count_tokens(text="hello world")
        assert token_count == 5
