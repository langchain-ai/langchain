"""Test chat model integration."""


from langchain_nvidia_aiplay.chat_models import ChatNVAIPlay


def test_integration_initialization() -> None:
    """Test chat model initialization."""
    ChatNVAIPlay(
        model="llama2_13b",
        nvidia_api_key="nvapi-...",
        temperature=0.5,
        top_p=0.9,
        max_tokens=50,
    )
    ChatNVAIPlay(model="mistral", nvidia_api_key="nvapi-...")
