"""Unit tests for TransformersEmbeddings."""

from typing import Any
from unittest.mock import MagicMock, patch

from langchain_huggingface.embeddings.transformers_embeddings import (
    TransformersEmbeddings,
)


class TestTransformersEmbeddings:
    """Test TransformersEmbeddings."""

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    @patch("torch.no_grad")
    def test_initialization_success(
        self, mock_no_grad: Any, mock_tokenizer: Any, mock_model: Any
    ) -> None:
        """Test successful initialization with mocked dependencies."""
        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock model
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        embeddings = TransformersEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            normalize_embeddings=True,
        )

        assert embeddings.model_name == "sentence-transformers/all-mpnet-base-v2"
        assert embeddings.normalize_embeddings is True
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    @patch("torch.no_grad")
    def test_configuration_properties(
        self, mock_no_grad: Any, mock_tokenizer: Any, mock_model: Any
    ) -> None:
        """Test that configuration properties are set correctly."""
        # Mock tokenizer and model
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        embeddings = TransformersEmbeddings(
            model_name="test-model",
            cache_dir="./test_cache",
            normalize_embeddings=False,
            show_progress=True,
        )

        assert embeddings.model_name == "test-model"
        assert embeddings.cache_dir == "./test_cache"
        assert embeddings.normalize_embeddings is False
        assert embeddings.show_progress is True

    def test_model_config(self) -> None:
        """Test that model configuration is set correctly."""
        # No need to initialize the actual model, just test the class attributes
        config = TransformersEmbeddings.model_config
        assert config["extra"] == "forbid"
        assert config["populate_by_name"] is True

    def test_default_values(self) -> None:
        """Test default field values without initializing."""
        from langchain_huggingface.embeddings.transformers_embeddings import (
            DEFAULT_MODEL_NAME,
        )

        # Test that default values are set correctly at class level
        assert DEFAULT_MODEL_NAME == "sentence-transformers/all-mpnet-base-v2"
