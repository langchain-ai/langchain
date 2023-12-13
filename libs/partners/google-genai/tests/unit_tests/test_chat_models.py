"""Test chat model integration."""


from langchain_google_genai.chat_models import ChatGoogleGenerativeAI


def test_integration_initialization() -> None:
    """Test chat model initialization."""
    ChatGoogleGenerativeAI(
        model="gemini-nano",
        google_api_key="...",
        top_k=2,
        top_p=1,
        temperature=0.7,
        n=2,
    )
    ChatGoogleGenerativeAI(
        model="gemini-nano",
        google_api_key="...",
        top_k=2,
        top_p=1,
        temperature=0.7,
        candidate_count=2,
    )
