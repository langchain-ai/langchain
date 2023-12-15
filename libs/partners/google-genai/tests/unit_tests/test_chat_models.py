"""Test chat model integration."""
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture

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


def test_api_key_is_string() -> None:
    chat = ChatGoogleGenerativeAI(model="gemini-nano", google_api_key="secret-api-key")
    assert isinstance(chat.google_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(capsys: CaptureFixture) -> None:
    chat = ChatGoogleGenerativeAI(model="gemini-nano", google_api_key="secret-api-key")
    print(chat.google_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"
