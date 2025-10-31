"""Test provider and model ID mappings."""

import pytest

from langchain_model_profiles.model_profile import get_model_profile


@pytest.mark.parametrize(
    ("provider", "model_id"),
    [
        ("openai-chat", "gpt-5"),
        ("anthropic-chat", "claude-sonnet-4-5"),
        ("vertexai", "models/gemini-2.0-flash-001"),
        ("chat-google-generative-ai", "models/gemini-2.0-flash-001"),
        ("amazon_bedrock_chat", "anthropic.claude-sonnet-4-20250514-v1:0"),
        ("amazon_bedrock_converse_chat", "anthropic.claude-sonnet-4-20250514-v1:0"),
    ],
)
def test_id_translation(provider: str, model_id: str) -> None:
    """Test translation from LangChain to model / provider IDs."""
    assert get_model_profile(provider, model_id)
