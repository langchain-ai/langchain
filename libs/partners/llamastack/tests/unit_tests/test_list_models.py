"""Unit tests for LlamaStack list models module."""

import os
import sys
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from langchain_llamastack.list_models import (
    classify_model,
    get_detailed_model_info,
    main,
)


class TestListModels:
    """Test cases for list models module."""

    def setup_method(self):
        """Set up test fixtures."""
        self.base_url = "http://test-server:8321"

    # def test_classify_model_llama(self):
    #     """Test classifying Llama models."""
    #     test_cases = [
    #         ("llama-2-7b", "Llama"),
    #         ("llama3-8b-instruct", "Llama"),
    #         ("meta-llama/Llama-3.2-1B-Instruct", "Llama"),
    #         ("accounts/fireworks/models/llama-v3p1-8b-instruct", "Llama"),
    #     ]

    #     for model_name, expected in test_cases:
    #         result = classify_model(model_name)
    #         assert result == expected

    # def test_classify_model_codellama(self):
    #     """Test classifying CodeLlama models."""
    #     test_cases = [
    #         ("code-llama-7b", "CodeLlama"),
    #         ("codellama-13b-instruct", "CodeLlama"),
    #     ]

    #     for model_name, expected in test_cases:
    #         result = classify_model(model_name)
    #         assert result == expected

    # def test_classify_model_embedding(self):
    #     """Test classifying embedding models."""
    #     test_cases = [
    #         ("all-minilm", "Embedding"),
    #         ("sentence-transformer", "Embedding"),
    #         ("bge-large", "Embedding"),
    #     ]

    #     for model_name, expected in test_cases:
    #         result = classify_model(model_name)
    #         assert result == expected

    # def test_classify_model_openai(self):
    #     """Test classifying OpenAI models."""
    #     test_cases = [
    #         ("gpt-3.5-turbo", "OpenAI"),
    #         ("gpt-4", "OpenAI"),
    #         ("gpt-4-turbo", "OpenAI"),
    #     ]

    #     for model_name, expected in test_cases:
    #         result = classify_model(model_name)
    #         assert result == expected

    # def test_classify_model_anthropic(self):
    #     """Test classifying Anthropic models."""
    #     test_cases = [
    #         ("claude-3-sonnet", "Anthropic"),
    #         ("claude-3-haiku", "Anthropic"),
    #         ("claude-instant", "Anthropic"),
    #     ]

    #     for model_name, expected in test_cases:
    #         result = classify_model(model_name)
    #         assert result == expected

    # def test_classify_model_unknown(self):
    #     """Test classifying unknown models."""
    #     test_cases = [
    #         ("unknown-model", "Other"),
    #         ("custom-model-v1", "Other"),
    #         ("random-name", "Other"),
    #     ]

    #     for model_name, expected in test_cases:
    #         result = classify_model(model_name)
    #         assert result == expected

    # @patch("langchain_llamastack.list_models.check_llamastack_connection")
    # def test_get_detailed_model_info_success(self, mock_check_connection):
    #     """Test getting detailed model info successfully."""
    #     mock_models = ["llama-2-7b", "gpt-4", "claude-3-sonnet", "all-minilm"]
    #     mock_check_connection.return_value = {"connected": True, "models": mock_models}

    #     models, error = get_detailed_model_info()

    #     assert error is None
    #     assert len(models) == 4

    #     # Verify model structure
    #     llama_model = next(m for m in models if m["name"] == "llama-2-7b")
    #     assert llama_model["category"] == "Llama"

    #     gpt_model = next(m for m in models if m["name"] == "gpt-4")
    #     assert gpt_model["category"] == "OpenAI"

    #     claude_model = next(m for m in models if m["name"] == "claude-3-sonnet")
    #     assert claude_model["category"] == "Anthropic"

    #     embedding_model = next(m for m in models if m["name"] == "all-minilm")
    #     assert embedding_model["category"] == "Embedding"

    # @patch("langchain_llamastack.list_models.check_llamastack_connection")
    # def test_get_detailed_model_info_connection_failed(self, mock_check_connection):
    #     """Test getting model info when connection fails."""
    #     mock_check_connection.return_value = {
    #         "connected": False,
    #         "error": "Connection refused",
    #     }

    #     models, error = get_detailed_model_info()

    #     assert models is None
    #     assert error == "Connection refused"

    # @patch("langchain_llamastack.list_models.check_llamastack_connection")
    # def test_get_detailed_model_info_no_models(self, mock_check_connection):
    #     """Test getting model info when no models are available."""
    #     mock_check_connection.return_value = {"connected": True, "models": []}

    #     models, error = get_detailed_model_info()

    #     assert error is None
    #     assert models == []

    # @patch("langchain_llamastack.list_models.check_llamastack_connection")
    # def test_get_detailed_model_info_exception(self, mock_check_connection):
    #     """Test getting model info with exception."""
    #     mock_check_connection.side_effect = Exception("Unexpected error")

    #     models, error = get_detailed_model_info()

    #     assert models is None
    #     assert "Unexpected error" in error

    # @patch("langchain_llamastack.list_models.get_detailed_model_info")
    # @patch("builtins.print")
    # def test_main_success(self, mock_print, mock_get_info):
    #     """Test main function with successful execution."""
    #     mock_models = [
    #         {"name": "llama-2-7b", "category": "Llama"},
    #         {"name": "gpt-4", "category": "OpenAI"},
    #         {"name": "claude-3-sonnet", "category": "Anthropic"},
    #         {"name": "all-minilm", "category": "Embedding"},
    #     ]
    #     mock_get_info.return_value = (mock_models, None)

    #     result = main()

    #     assert result == 0  # Success exit code
    #     mock_get_info.assert_called_once()

    #     # Verify that print was called (output was generated)
    #     assert mock_print.call_count > 0

    # @patch("langchain_llamastack.list_models.get_detailed_model_info")
    # @patch("builtins.print")
    # def test_main_connection_error(self, mock_print, mock_get_info):
    #     """Test main function with connection error."""
    #     mock_get_info.return_value = (None, "Connection failed")

    #     result = main()

    #     assert result == 1  # Error exit code

    #     # Verify error message was printed
    #     error_calls = [
    #         call
    #         for call in mock_print.call_args_list
    #         if "Error connecting to LlamaStack" in str(call)
    #     ]
    #     assert len(error_calls) > 0

    # @patch("langchain_llamastack.list_models.get_detailed_model_info")
    # @patch("builtins.print")
    # def test_main_no_models(self, mock_print, mock_get_info):
    #     """Test main function with no models available."""
    #     mock_get_info.return_value = ([], None)

    #     result = main()

    #     assert result == 0  # Success exit code even with no models

    #     # Verify no models message was printed
    #     no_models_calls = [
    #         call for call in mock_print.call_args_list if "No models found" in str(call)
    #     ]
    #     assert len(no_models_calls) > 0

    # @patch("langchain_llamastack.list_models.get_detailed_model_info")
    # @patch("builtins.print")
    # def test_main_grouped_output(self, mock_print, mock_get_info):
    #     """Test main function generates properly grouped output."""
    #     mock_models = [
    #         {"name": "llama-2-7b", "category": "Llama"},
    #         {"name": "llama-3-8b", "category": "Llama"},
    #         {"name": "gpt-4", "category": "OpenAI"},
    #         {"name": "gpt-3.5-turbo", "category": "OpenAI"},
    #         {"name": "claude-3-sonnet", "category": "Anthropic"},
    #         {"name": "all-minilm", "category": "Embedding"},
    #         {"name": "unknown-model", "category": "Other"},
    #     ]
    #     mock_get_info.return_value = (mock_models, None)

    #     result = main()

    #     assert result == 0

    #     # Check that all categories are represented in output
    #     all_output = " ".join(str(call) for call in mock_print.call_args_list)

    #     assert "Llama" in all_output
    #     assert "OpenAI" in all_output
    #     assert "Anthropic" in all_output
    #     assert "Embedding" in all_output
    #     assert "Other" in all_output

    #     # Check model names are in output
    #     assert "llama-2-7b" in all_output
    #     assert "gpt-4" in all_output
    #     assert "claude-3-sonnet" in all_output
    #     assert "all-minilm" in all_output

    # def test_classify_model_case_insensitive(self):
    #     """Test that model classification is case insensitive."""
    #     test_cases = [
    #         ("LLAMA-2-7B", "Llama"),
    #         ("GPT-4", "OpenAI"),
    #         ("CLAUDE-3-SONNET", "Anthropic"),
    #         ("ALL-MINILM", "Embedding"),
    #     ]

    #     for model_name, expected in test_cases:
    #         result = classify_model(model_name)
    #         assert result == expected

    # def test_classify_model_partial_matches(self):
    #     """Test model classification with partial pattern matches."""
    #     test_cases = [
    #         ("custom-llama-model", "Llama"),
    #         ("my-gpt-model", "OpenAI"),
    #         ("claude-custom", "Anthropic"),
    #         ("embedding-all-minilm", "Embedding"),
    #     ]

    #     for model_name, expected in test_cases:
    #         result = classify_model(model_name)
    #         assert result == expected

    # @patch("langchain_llamastack.list_models.check_llamastack_connection")
    # def test_get_detailed_model_info_mixed_models(self, mock_check_connection):
    #     """Test getting detailed model info with mixed model types."""
    #     mock_models = [
    #         "llama-2-7b",
    #         "accounts/fireworks/models/mixtral-8x7b",
    #         "gpt-4-turbo",
    #         "claude-3-haiku",
    #         "sentence-transformers/all-minilm-l6-v2",
    #         "code-llama-13b",
    #         "unknown-custom-model",
    #     ]
    #     mock_check_connection.return_value = {"connected": True, "models": mock_models}

    #     models, error = get_detailed_model_info()

    #     assert error is None
    #     assert len(models) == 7

    #     # Verify all categories are represented
    #     categories = {m["category"] for m in models}
    #     expected_categories = {
    #         "Llama",
    #         "Mistral",
    #         "OpenAI",
    #         "Anthropic",
    #         "Embedding",
    #         "CodeLlama",
    #         "Other",
    #     }
    #     assert categories == expected_categories

    # @patch("sys.argv", ["list_models.py"])
    # @patch("langchain_llamastack.list_models.main")
    # def test_main_entry_point(self, mock_main):
    #     """Test that main can be called as entry point."""
    #     mock_main.return_value = 0

    #     # Import and test the module's main execution
    #     import langchain_llamastack.list_models

    #     # The module should be importable without errors
    #     assert hasattr(langchain_llamastack.list_models, "main")
    #     assert callable(langchain_llamastack.list_models.main)
