"""Unit tests for TransformersTokenTextSplitter."""

import pytest

from langchain_text_splitters.transformers_token import TransformersTokenTextSplitter


class TestTransformersTokenTextSplitter:
    """Test TransformersTokenTextSplitter."""

    def test_initialization(self) -> None:
        """Test that the splitter can be initialized."""
        try:
            splitter = TransformersTokenTextSplitter(
                model_name="sentence-transformers/all-mpnet-base-v2",
                chunk_overlap=10,
                tokens_per_chunk=100,
            )
            assert splitter.model_name == "sentence-transformers/all-mpnet-base-v2"
            assert splitter.tokens_per_chunk == 100
            assert splitter._chunk_overlap == 10
        except ImportError:
            pytest.skip("transformers not available")

    def test_split_text(self) -> None:
        """Test basic text splitting functionality."""
        try:
            splitter = TransformersTokenTextSplitter(
                model_name="sentence-transformers/all-mpnet-base-v2",
                tokens_per_chunk=10,
            )
            text = "This is a test sentence. " * 20
            chunks = splitter.split_text(text)
            assert isinstance(chunks, list)
            assert len(chunks) > 1
            assert all(isinstance(chunk, str) for chunk in chunks)
        except ImportError:
            pytest.skip("transformers not available")

    def test_count_tokens(self) -> None:
        """Test token counting functionality."""
        try:
            splitter = TransformersTokenTextSplitter(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
            text = "This is a test sentence."
            token_count = splitter.count_tokens(text=text)
            assert isinstance(token_count, int)
            assert token_count > 0
        except ImportError:
            pytest.skip("transformers not available")

    def test_tokens_per_chunk_validation(self) -> None:
        """Test that tokens_per_chunk is validated against model limits."""
        try:
            with pytest.raises(ValueError, match="maximum token limit"):
                TransformersTokenTextSplitter(
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    tokens_per_chunk=100000,  # Way too large
                )
        except ImportError:
            pytest.skip("transformers not available")
