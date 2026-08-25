"""Tests for cache token extraction in Perplexity usage metadata."""

from langchain_perplexity.chat_models import _create_usage_metadata


def test_create_usage_metadata_extracts_cache_tokens():
    token_usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 50,
        "total_tokens": 1050,
        "input_tokens_details": {
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 150,
        },
    }
    result = _create_usage_metadata(token_usage)
    assert result["input_token_details"]["cache_read"] == 800
    assert result["input_token_details"]["cache_creation"] == 150


def test_create_usage_metadata_without_cache_details():
    token_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    result = _create_usage_metadata(token_usage)
    assert "input_token_details" not in result
