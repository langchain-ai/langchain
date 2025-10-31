"""Test provider and model ID mappings."""

from langchain_model_profiles.model_profile import get_model_profile


def test_id_translation() -> None:
    """Test translation from LangChain to model / provider IDs."""
    assert get_model_profile("vertexai", "models/gemini-2.0-flash-001")
    assert get_model_profile("chat-google-generative-ai", "models/gemini-2.0-flash-001")
