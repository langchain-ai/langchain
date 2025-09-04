"""Unit tests for LlamaStack embeddings."""

from typing import Any, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from langchain_llamastack.embeddings import LlamaStackEmbeddings


class TestLlamaStackEmbeddings:
    """Test cases for LlamaStackEmbeddings."""

    def setup_method(self):
        """Set up test fixtures."""
        self.base_url = "http://test-server:8321"
        self.model = "test-model"

    @patch("langchain_llamastack.embeddings.LlamaStackClient")
    def test_init_default_params(self, mock_client_class):
        """Test initialization with default parameters."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        embeddings = LlamaStackEmbeddings()

        assert embeddings.model == "all-minilm"
        assert embeddings.base_url == "http://localhost:8321"
        assert embeddings.chunk_size == 1000
        assert embeddings.max_retries == 3
        assert embeddings.request_timeout == 30.0
        assert embeddings.client is not None

    @patch("langchain_llamastack.embeddings.LlamaStackClient")
    def test_init_custom_params(self, mock_client_class):
        """Test initialization with custom parameters."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        embeddings = LlamaStackEmbeddings(
            model=self.model,
            base_url=self.base_url,
            chunk_size=500,
            max_retries=5,
            request_timeout=60.0,
        )

        assert embeddings.model == self.model
        assert embeddings.base_url == self.base_url
        assert embeddings.chunk_size == 500
        assert embeddings.max_retries == 5
        assert embeddings.request_timeout == 60.0

    @patch("langchain_llamastack.embeddings.LlamaStackClient")
    def test_setup_client_success(self, mock_client_class):
        """Test successful client setup."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        embeddings = LlamaStackEmbeddings(base_url=self.base_url)

        mock_client_class.assert_called_once_with(base_url=self.base_url)
        assert embeddings.client == mock_client

    @patch("langchain_llamastack.embeddings.LlamaStackClient")
    def test_setup_client_failure(self, mock_client_class):
        """Test client setup failure."""
        mock_client_class.side_effect = Exception("Connection failed")

        with pytest.raises(ValueError, match="Failed to initialize Llama Stack client"):
            LlamaStackEmbeddings(base_url=self.base_url)

    def test_llm_type(self):
        """Test _llm_type property."""
        with patch("langchain_llamastack.embeddings.LlamaStackClient"):
            embeddings = LlamaStackEmbeddings()
            assert embeddings._llm_type == "llamastack-embeddings"

    @patch("httpx.Client")
    @patch("langchain_llamastack.embeddings.LlamaStackClient")
    def test_embed_texts_success(self, mock_client_class, mock_httpx_client):
        """Test successful text embedding."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]
        }
        mock_response.raise_for_status.return_value = None

        mock_http_client = Mock()
        mock_http_client.post.return_value = mock_response
        mock_httpx_client.return_value.__enter__.return_value = mock_http_client

        embeddings = LlamaStackEmbeddings(model=self.model, base_url=self.base_url)

        # Test embedding
        texts = ["hello", "world"]
        result = embeddings._embed_texts(texts)

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

        # Verify API call
        mock_http_client.post.assert_called_once()
        call_args = mock_http_client.post.call_args
        assert f"{self.base_url}/v1/openai/v1/embeddings" in call_args[0]
        assert call_args[1]["json"]["model"] == self.model
        assert call_args[1]["json"]["input"] == texts

    @patch("httpx.Client")
    @patch("langchain_llamastack.embeddings.LlamaStackClient")
    def test_embed_texts_http_error(self, mock_client_class, mock_httpx_client):
        """Test embedding with HTTP error."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("Server error")

        mock_http_client = Mock()
        mock_http_client.post.return_value = mock_response
        mock_httpx_client.return_value.__enter__.return_value = mock_http_client

        embeddings = LlamaStackEmbeddings(base_url=self.base_url)

        with pytest.raises(Exception):
            embeddings._embed_texts(["test"])

    @patch("httpx.Client")
    @patch("langchain_llamastack.embeddings.LlamaStackClient")
    def test_embed_texts_invalid_response(self, mock_client_class, mock_httpx_client):
        """Test embedding with invalid response format."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_response = Mock()
        mock_response.json.return_value = {"error": "Invalid request"}
        mock_response.raise_for_status.return_value = None

        mock_http_client = Mock()
        mock_http_client.post.return_value = mock_response
        mock_httpx_client.return_value.__enter__.return_value = mock_http_client

        embeddings = LlamaStackEmbeddings(base_url=self.base_url)

        with pytest.raises(ValueError, match="No data in embeddings response"):
            embeddings._embed_texts(["test"])

    # @patch.object(LlamaStackEmbeddings, "_embed_with_retry")
    # @patch("langchain_llamastack.embeddings.LlamaStackClient")
    # def test_embed_documents(self, mock_client_class, mock_embed):
    #     """Test embedding multiple documents."""
    #     mock_client = Mock()
    #     mock_client_class.return_value = mock_client

    #     mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    #     embeddings = LlamaStackEmbeddings(chunk_size=2)
    #     texts = ["doc1", "doc2", "doc3"]

    #     result = embeddings.embed_documents(texts)

    #     assert len(result) == 3
    #     assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    @patch.object(LlamaStackEmbeddings, "_embed_with_retry")
    @patch("langchain_llamastack.embeddings.LlamaStackClient")
    def test_embed_query(self, mock_client_class, mock_embed):
        """Test embedding a single query."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_embed.return_value = [[0.1, 0.2, 0.3]]

        embeddings = LlamaStackEmbeddings()
        result = embeddings.embed_query("test query")

        assert result == [0.1, 0.2, 0.3]
        mock_embed.assert_called_once_with(["test query"])

    @patch("langchain_llamastack.embeddings.LlamaStackClient")
    def test_identifying_params(self, mock_client_class):
        """Test identifying parameters property."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        embeddings = LlamaStackEmbeddings(
            model=self.model,
            base_url=self.base_url,
            chunk_size=500,
            max_retries=5,
            request_timeout=60.0,
        )

        params = embeddings._identifying_params

        expected = {
            "model": self.model,
            "base_url": self.base_url,
            "chunk_size": 500,
            "max_retries": 5,
            "request_timeout": 60.0,
        }
        assert params == expected

    @patch.object(LlamaStackEmbeddings, "_embed_texts")
    @patch("langchain_llamastack.embeddings.LlamaStackClient")
    def test_embed_with_retry_success(self, mock_client_class, mock_embed_texts):
        """Test successful embedding with retry logic."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_embed_texts.return_value = [[0.1, 0.2, 0.3]]

        embeddings = LlamaStackEmbeddings()
        result = embeddings._embed_with_retry(["test"])

        assert result == [[0.1, 0.2, 0.3]]
        mock_embed_texts.assert_called_once_with(["test"])

    @patch.object(LlamaStackEmbeddings, "_embed_texts")
    @patch("langchain_llamastack.embeddings.LlamaStackClient")
    def test_embed_with_retry_failure_then_success(
        self, mock_client_class, mock_embed_texts
    ):
        """Test embedding with retry after failure."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # First call fails, second succeeds
        mock_embed_texts.side_effect = [Exception("Network error"), [[0.1, 0.2, 0.3]]]

        embeddings = LlamaStackEmbeddings()
        result = embeddings._embed_with_retry(["test"])

        assert result == [[0.1, 0.2, 0.3]]
        assert mock_embed_texts.call_count == 2

    @patch.object(LlamaStackEmbeddings, "_embed_texts")
    @patch("langchain_llamastack.embeddings.LlamaStackClient")
    def test_embed_with_retry_max_retries_exceeded(
        self, mock_client_class, mock_embed_texts
    ):
        """Test embedding failure after max retries."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_embed_texts.side_effect = Exception("Persistent error")

        embeddings = LlamaStackEmbeddings(max_retries=2)

        with pytest.raises(Exception, match="Persistent error"):
            embeddings._embed_with_retry(["test"])

        assert mock_embed_texts.call_count == 2

    @patch("langchain_llamastack.embeddings.LlamaStackClient")
    def test_get_model_info_with_client_success(self, mock_client_class):
        """Test getting model info successfully."""
        mock_client = Mock()
        mock_model = Mock()
        mock_model.identifier = "test-model"
        mock_model.provider_resource_id = "provider-resource"
        mock_model.provider_id = "test-provider"
        mock_model.model_type = "embedding"
        mock_model.metadata = {"embedding_dimension": 768, "context_length": 512}

        mock_client.models.list.return_value = [mock_model]
        mock_client_class.return_value = mock_client

        embeddings = LlamaStackEmbeddings(model="test-model")
        result = embeddings.get_model_info()

        expected = {
            "identifier": "test-model",
            "provider_resource_id": "provider-resource",
            "provider_id": "test-provider",
            "model_type": "embedding",
            "metadata": {"embedding_dimension": 768, "context_length": 512},
            "embedding_dimension": 768,
            "context_length": 512,
        }
        assert result == expected

    @patch("langchain_llamastack.embeddings.LlamaStackClient")
    def test_get_model_info_model_not_found(self, mock_client_class):
        """Test getting model info when model not found."""
        mock_client = Mock()
        mock_client.models.list.return_value = []
        mock_client_class.return_value = mock_client

        embeddings = LlamaStackEmbeddings(model="nonexistent-model")
        result = embeddings.get_model_info()

        assert result == {"error": "Model nonexistent-model not found"}

    @patch("langchain_llamastack.embeddings.LlamaStackClient")
    def test_get_model_info_client_error(self, mock_client_class):
        """Test getting model info with client error."""
        mock_client = Mock()
        mock_client.models.list.side_effect = Exception("API error")
        mock_client_class.return_value = mock_client

        embeddings = LlamaStackEmbeddings()
        result = embeddings.get_model_info()

        assert result == {"error": "API error"}

    @patch.object(LlamaStackEmbeddings, "get_model_info")
    @patch("langchain_llamastack.embeddings.LlamaStackClient")
    def test_get_embedding_dimension(self, mock_client_class, mock_get_model_info):
        """Test getting embedding dimension."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_get_model_info.return_value = {"embedding_dimension": 768}

        embeddings = LlamaStackEmbeddings()
        result = embeddings.get_embedding_dimension()

        assert result == 768

    # @patch.object(LlamaStackEmbeddings, "embed_documents")
    # @patch("langchain_llamastack.embeddings.LlamaStackClient")
    # def test_similarity_search_by_vector_success(
    #     self, mock_client_class, mock_embed_docs
    # ):
    #     """Test similarity search by vector."""
    #     mock_client = Mock()
    #     mock_client_class.return_value = mock_client

    #     # Mock document embeddings
    #     mock_embed_docs.return_value = [
    #         [0.1, 0.2, 0.3],  # doc1
    #         [0.4, 0.5, 0.6],  # doc2
    #         [0.7, 0.8, 0.9],  # doc3
    #     ]

    #     with patch("numpy.argsort") as mock_argsort, patch(
    #         "sklearn.metrics.pairwise.cosine_similarity"
    #     ) as mock_cosine:

    #         # Mock cosine similarity scores
    #         mock_cosine.return_value = [[0.9, 0.7, 0.5]]  # similarity scores
    #         mock_argsort.return_value = [0, 1, 2]  # indices sorted by similarity

    #         embeddings = LlamaStackEmbeddings()

    #         documents = ["doc1", "doc2", "doc3"]
    #         query_embedding = [0.1, 0.2, 0.3]

    #         results = embeddings.similarity_search_by_vector(
    #             query_embedding, documents, k=2
    #         )

    #         assert len(results) == 2
    #         assert results[0] == ("doc1", 0.9)
    #         assert results[1] == ("doc2", 0.7)

    # @patch("langchain_llamastack.embeddings.LlamaStackClient")
    # def test_similarity_search_missing_sklearn(self, mock_client_class):
    #     """Test similarity search when sklearn is not available."""
    #     mock_client = Mock()
    #     mock_client_class.return_value = mock_client
    #     embeddings = LlamaStackEmbeddings()

    #     # We need to patch the import inside the similarity_search_by_vector method
    #     # Since the import happens inside a try/except, we need to ensure it fails
    #     import sys

    #     # Temporarily remove sklearn from modules if it exists
    #     sklearn_modules = [k for k in sys.modules.keys() if k.startswith("sklearn")]
    #     removed_modules = {}
    #     for module_name in sklearn_modules:
    #         removed_modules[module_name] = sys.modules.pop(module_name, None)

    #     try:
    #         # Now when the method tries to import sklearn, it should fail
    #         with pytest.raises(ImportError, match="scikit-learn is required"):
    #             embeddings.similarity_search_by_vector([0.1, 0.2], ["doc1"], k=1)
    #     finally:
    #         # Restore the modules
    #         for module_name, module in removed_modules.items():
    #             if module is not None:
    #                 sys.modules[module_name] = module

    # @patch("langchain_llamastack.embeddings.LlamaStackClient")
    # def test_get_available_models_success(self, mock_client_class):
    #     """Test getting available models successfully."""
    #     mock_client = Mock()
    #     mock_client_class.return_value = mock_client

    #     with patch(
    #         "langchain_llamastack.embeddings.list_available_models"
    #     ) as mock_list:
    #         mock_list.return_value = ["model1", "model2", "model3"]

    #         embeddings = LlamaStackEmbeddings()
    #         result = embeddings.get_available_models()

    #         assert result == ["model1", "model2", "model3"]
    #         mock_list.assert_called_once_with(
    #             embeddings.base_url, model_type="embedding"
    #         )

    # @patch("langchain_llamastack.embeddings.LlamaStackClient")
    # def test_get_available_models_error(self, mock_client_class):
    #     """Test getting available models with error."""
    #     mock_client = Mock()
    #     mock_client_class.return_value = mock_client

    #     with patch(
    #         "langchain_llamastack.embeddings.list_available_models"
    #     ) as mock_list:
    #         mock_list.side_effect = Exception("API error")

    #         embeddings = LlamaStackEmbeddings()
    #         result = embeddings.get_available_models()

    #         assert result == []
