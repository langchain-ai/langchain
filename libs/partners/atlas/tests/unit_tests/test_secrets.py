"""Secret handling tests for `ChatAtlas`."""

from langchain_atlas import ChatAtlas

MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"


def test_chat_atlas_secrets() -> None:
    """Test that the API key is masked in string output."""
    model = ChatAtlas(model=MODEL_NAME, atlas_api_key="foo")  # type: ignore[call-arg]
    assert "foo" not in str(model)
