"""Test embeddings model integration."""
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture

from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings


def test_integration_initialization() -> None:
    """Test chat model initialization."""
    GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key="...",
    )
    GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key="...",
        task_type="retrieval_document",
    )


def test_api_key_is_string() -> None:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key="secret-api-key",
    )
    assert isinstance(embeddings.google_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(capsys: CaptureFixture) -> None:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key="secret-api-key",
    )
    print(embeddings.google_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"
