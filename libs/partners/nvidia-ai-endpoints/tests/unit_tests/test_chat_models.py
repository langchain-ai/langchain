"""Test chat model integration."""


from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA


def test_integration_initialization() -> None:
    """Test chat model initialization."""
    ChatNVIDIA(
        model="llama2_13b",
        nvidia_api_key="nvapi-...",
        temperature=0.5,
        top_p=0.9,
        max_tokens=50,
    )
    ChatNVIDIA(model="mistral", nvidia_api_key="nvapi-...")
