"""End to end test for fetching model profiles from a chat model."""

from langchain.chat_models import init_chat_model


def test_chat_model() -> None:
    """Test that chat model gets profile data correctly."""
    model = init_chat_model("openai:gpt-5", api_key="foo")
    assert model.profile
    assert model.profile["max_input_tokens"] == 400000
    assert model.profile["structured_output"]


def test_chat_model_no_data() -> None:
    """Test that chat model handles missing profile data."""
    model = init_chat_model("openai:gpt-fake", api_key="foo")
    assert model.profile == {}
