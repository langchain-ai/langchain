"""Unit tests for LlamaStack utils module."""

import os
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from langchain_llamastack.utils import check_llamastack_connection


class TestUtils:
    """Test cases for utils module."""

    def setup_method(self):
        """Set up test fixtures."""
        self.base_url = "http://test-server:8321"
        self.api_key = "test-api-key"

    @patch("langchain_llamastack.utils.LlamaStackClient")
    def test_check_llamastack_connection_success(self, mock_client_class):
        """Test successful LlamaStack connection check."""
        # Setup mock client and models
        mock_model1 = Mock()
        mock_model1.identifier = "model1"
        mock_model1.model_type = "inference"

        mock_model2 = Mock()
        mock_model2.identifier = "model2"
        mock_model2.model_type = "embedding"

        mock_model3 = Mock()
        mock_model3.identifier = "model3"
        mock_model3.model_type = "inference"

        mock_client = Mock()
        mock_client.models.list.return_value = [mock_model1, mock_model2, mock_model3]
        mock_client_class.return_value = mock_client

        result = check_llamastack_connection(self.base_url)

        expected = {
            "connected": True,
            "models": ["model1", "model3"],  # Only inference models
            "models_count": 2,
            "base_url": self.base_url,
        }
        assert result == expected
        mock_client_class.assert_called_once_with(base_url=self.base_url)

    @patch("langchain_llamastack.utils.LlamaStackClient")
    def test_check_llamastack_connection_success_with_api_key(self, mock_client_class):
        """Test successful LlamaStack connection check with API key."""
        mock_model = Mock()
        mock_model.identifier = "test-model"
        mock_model.model_type = "inference"

        mock_client = Mock()
        mock_client.models.list.return_value = [mock_model]
        mock_client_class.return_value = mock_client

        result = check_llamastack_connection(
            self.base_url, llamastack_api_key=self.api_key
        )

        expected = {
            "connected": True,
            "models": ["test-model"],
            "models_count": 1,
            "base_url": self.base_url,
        }
        assert result == expected
        mock_client_class.assert_called_once_with(
            base_url=self.base_url, api_key=self.api_key
        )

    @patch("langchain_llamastack.utils.LlamaStackClient")
    def test_check_llamastack_connection_filter_by_model_type(self, mock_client_class):
        """Test connection check filtered by specific model type."""
        # Setup mock models of different types
        mock_inference = Mock()
        mock_inference.identifier = "inference-model"
        mock_inference.model_type = "inference"

        mock_embedding = Mock()
        mock_embedding.identifier = "embedding-model"
        mock_embedding.model_type = "embedding"

        mock_safety = Mock()
        mock_safety.identifier = "safety-model"
        mock_safety.model_type = "safety"

        mock_client = Mock()
        mock_client.models.list.return_value = [
            mock_inference,
            mock_embedding,
            mock_safety,
        ]
        mock_client_class.return_value = mock_client

        # Test filtering for embedding models
        result = check_llamastack_connection(self.base_url, model_type="embedding")

        expected = {
            "connected": True,
            "models": ["embedding-model"],
            "models_count": 1,
            "base_url": self.base_url,
        }
        assert result == expected

    @patch("langchain_llamastack.utils.LlamaStackClient")
    def test_check_llamastack_connection_no_models(self, mock_client_class):
        """Test connection check when no models are available."""
        mock_client = Mock()
        mock_client.models.list.return_value = []
        mock_client_class.return_value = mock_client

        result = check_llamastack_connection(self.base_url)

        expected = {
            "connected": True,
            "models": [],
            "models_count": 0,
            "base_url": self.base_url,
        }
        assert result == expected

    @patch("langchain_llamastack.utils.LlamaStackClient")
    def test_check_llamastack_connection_no_matching_models(self, mock_client_class):
        """Test connection check when no models match the filter."""
        mock_model = Mock()
        mock_model.identifier = "inference-model"
        mock_model.model_type = "inference"

        mock_client = Mock()
        mock_client.models.list.return_value = [mock_model]
        mock_client_class.return_value = mock_client

        # Look for embedding models but only inference models exist
        result = check_llamastack_connection(self.base_url, model_type="embedding")

        expected = {
            "connected": True,
            "models": [],
            "models_count": 0,
            "base_url": self.base_url,
        }
        assert result == expected

    # @patch("langchain_llamastack.utils.LlamaStackClient")
    # def test_check_llamastack_connection_client_creation_error(self, mock_client_class):
    #     """Test connection check when client creation fails."""
    #     mock_client_class.side_effect = Exception("Connection failed")

    #     result = check_llamastack_connection(self.base_url)

    #     expected = {
    #         "connected": False,
    #         "error": "Connection failed",
    #         "models": [],
    #         "models_count": 0,
    #         "base_url": self.base_url,
    #     }
    #     assert result == expected

    @patch("langchain_llamastack.utils.LlamaStackClient")
    def test_check_llamastack_connection_models_list_error(self, mock_client_class):
        """Test connection check when models.list() fails."""
        mock_client = Mock()
        mock_client.models.list.side_effect = Exception("Models API error")
        mock_client_class.return_value = mock_client

        result = check_llamastack_connection(self.base_url)

        expected = {
            "connected": False,
            "error": "Models API error",
            "models": [],
            "models_count": 0,
            "base_url": self.base_url,
        }
        assert result == expected

    # def test_check_llamastack_connection_no_client_available(self):
    #     """Test connection check when LlamaStackClient is not available."""
    #     with patch("langchain_llamastack.utils.LlamaStackClient", None):
    #         result = check_llamastack_connection(self.base_url)

    #         expected = {
    #             "connected": False,
    #             "error": "llama-stack-client not available. Install with: pip install llama-stack-client",
    #             "models": [],
    #             "models_count": 0,
    #             "base_url": self.base_url,
    #         }
    #         assert result == expected

    @patch("langchain_llamastack.utils.LlamaStackClient")
    def test_check_llamastack_connection_model_without_identifier(
        self, mock_client_class
    ):
        """Test connection check with model missing identifier attribute."""
        # Setup mock model without identifier
        mock_model = Mock()
        del mock_model.identifier  # Remove identifier attribute
        mock_model.model_type = "inference"

        mock_client = Mock()
        mock_client.models.list.return_value = [mock_model]
        mock_client_class.return_value = mock_client

        result = check_llamastack_connection(self.base_url)

        # Should handle gracefully and exclude models without identifiers
        expected = {
            "connected": True,
            "models": [],
            "models_count": 0,
            "base_url": self.base_url,
        }
        assert result == expected

    @patch("langchain_llamastack.utils.LlamaStackClient")
    def test_check_llamastack_connection_model_without_model_type(
        self, mock_client_class
    ):
        """Test connection check with model missing model_type attribute."""
        # Setup mock model without model_type
        mock_model = Mock()
        mock_model.identifier = "test-model"
        # Don't set model_type - it will be None when accessed with getattr

        mock_client = Mock()
        mock_client.models.list.return_value = [mock_model]
        mock_client_class.return_value = mock_client

        result = check_llamastack_connection(self.base_url)

        # Should handle gracefully and exclude models without model_type
        expected = {
            "connected": True,
            "models": [],
            "models_count": 0,
            "base_url": self.base_url,
        }
        assert result == expected

    @patch("langchain_llamastack.utils.LlamaStackClient")
    def test_check_llamastack_connection_mixed_valid_invalid_models(
        self, mock_client_class
    ):
        """Test connection check with mix of valid and invalid models."""
        # Setup valid model
        mock_valid_model = Mock()
        mock_valid_model.identifier = "valid-model"
        mock_valid_model.model_type = "inference"

        # Setup invalid model (missing identifier)
        mock_invalid_model = Mock()
        del mock_invalid_model.identifier
        mock_invalid_model.model_type = "inference"

        mock_client = Mock()
        mock_client.models.list.return_value = [mock_valid_model, mock_invalid_model]
        mock_client_class.return_value = mock_client

        result = check_llamastack_connection(self.base_url)

        # Should only include valid models
        expected = {
            "connected": True,
            "models": ["valid-model"],
            "models_count": 1,
            "base_url": self.base_url,
        }
        assert result == expected

    @patch("langchain_llamastack.utils.LlamaStackClient")
    def test_check_llamastack_connection_default_model_type_inference(
        self, mock_client_class
    ):
        """Test that default model_type is 'inference'."""
        mock_inference = Mock()
        mock_inference.identifier = "inference-model"
        mock_inference.model_type = "inference"

        mock_embedding = Mock()
        mock_embedding.identifier = "embedding-model"
        mock_embedding.model_type = "embedding"

        mock_client = Mock()
        mock_client.models.list.return_value = [mock_inference, mock_embedding]
        mock_client_class.return_value = mock_client

        # Call without specifying model_type (should default to "inference")
        result = check_llamastack_connection(self.base_url)

        expected = {
            "connected": True,
            "models": ["inference-model"],  # Only inference model included
            "models_count": 1,
            "base_url": self.base_url,
        }
        assert result == expected

    @patch("langchain_llamastack.utils.LlamaStackClient")
    def test_check_llamastack_connection_case_insensitive_model_type(
        self, mock_client_class
    ):
        """Test that model type filtering is case insensitive."""
        mock_model = Mock()
        mock_model.identifier = "test-model"
        mock_model.model_type = "INFERENCE"  # Uppercase model type

        mock_client = Mock()
        mock_client.models.list.return_value = [mock_model]
        mock_client_class.return_value = mock_client

        # Search with lowercase model type
        result = check_llamastack_connection(self.base_url, model_type="inference")

        expected = {
            "connected": True,
            "models": ["test-model"],
            "models_count": 1,
            "base_url": self.base_url,
        }
        assert result == expected

    @patch("langchain_llamastack.utils.LlamaStackClient")
    def test_check_llamastack_connection_empty_string_model_type(
        self, mock_client_class
    ):
        """Test connection check with empty string model_type."""
        mock_model = Mock()
        mock_model.identifier = "test-model"
        mock_model.model_type = "inference"

        mock_client = Mock()
        mock_client.models.list.return_value = [mock_model]
        mock_client_class.return_value = mock_client

        result = check_llamastack_connection(self.base_url, model_type="")

        # Empty model type should not match any models
        expected = {
            "connected": True,
            "models": [],
            "models_count": 0,
            "base_url": self.base_url,
        }
        assert result == expected
