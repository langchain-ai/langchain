"""Test chat model integration."""


from langchain_google_genai.chat_models import ChatGoogleGenerativeAI


def test_integration_initialization() -> None:
    """Test chat model initialization."""
    ChatGoogleGenerativeAI(model="gemini-nano", google_api_key="...")
