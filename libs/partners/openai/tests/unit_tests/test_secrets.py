from typing import cast

from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAI,
    OpenAIEmbeddings,
)


def test_chat_openai_secrets() -> None:
    o = ChatOpenAI(openai_api_key="foo")
    s = str(o)
    assert "foo" not in s


def test_openai_secrets() -> None:
    o = OpenAI(openai_api_key="foo")
    s = str(o)
    assert "foo" not in s


def test_openai_embeddings_secrets() -> None:
    o = OpenAIEmbeddings(openai_api_key="foo")
    s = str(o)
    assert "foo" not in s


def test_azure_chat_openai_secrets() -> None:
    o = AzureChatOpenAI(
        openai_api_key="foo1",
        azure_endpoint="endpoint",
        azure_ad_token="foo2",
        api_version="version",
    )
    s = str(o)
    assert "foo1" not in s
    assert "foo2" not in s


def test_azure_openai_secrets() -> None:
    o = AzureOpenAI(
        openai_api_key="foo1",
        azure_endpoint="endpoint",
        azure_ad_token="foo2",
        api_version="version",
    )
    s = str(o)
    assert "foo1" not in s
    assert "foo2" not in s


def test_azure_openai_embeddings_secrets() -> None:
    o = AzureOpenAIEmbeddings(
        openai_api_key="foo1",
        azure_endpoint="endpoint",
        azure_ad_token="foo2",
        api_version="version",
    )
    s = str(o)
    assert "foo1" not in s
    assert "foo2" not in s


# azure chat openai
def test_azure_chat_openai_api_key_is_secret_string() -> None:
    """Test that the API key is stored as a SecretStr."""
    chat_model = AzureChatOpenAI(
        openai_api_key="secret-api-key",
        azure_endpoint="endpoint",
        azure_ad_token="secret-ad-token",
        api_version="version",
    )
    assert isinstance(chat_model.openai_api_key, SecretStr)
    assert isinstance(chat_model.azure_ad_token, SecretStr)


def test_azure_chat_openai_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test that the API key is masked when passed from an environment variable."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret-api-key")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "secret-ad-token")
    chat_model = AzureChatOpenAI(
        openai_api_key="secret-api-key",
        azure_endpoint="endpoint",
        azure_ad_token="secret-ad-token",
        api_version="version",
    )
    print(chat_model.openai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"

    print(chat_model.azure_ad_token, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_azure_chat_openai_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test that the API key is masked when passed via the constructor."""
    chat_model = AzureChatOpenAI(
        openai_api_key="secret-api-key",
        azure_endpoint="endpoint",
        azure_ad_token="secret-ad-token",
        api_version="version",
    )
    print(chat_model.openai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"

    print(chat_model.azure_ad_token, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_azure_chat_openai_uses_actual_secret_value_from_secretstr() -> None:
    """Test that the actual secret value is correctly retrieved."""
    chat_model = AzureChatOpenAI(
        openai_api_key="secret-api-key",
        azure_endpoint="endpoint",
        azure_ad_token="secret-ad-token",
        api_version="version",
    )
    assert (
        cast(SecretStr, chat_model.openai_api_key).get_secret_value()
        == "secret-api-key"
    )
    assert (
        cast(SecretStr, chat_model.azure_ad_token).get_secret_value()
        == "secret-ad-token"
    )


# azure openai
def test_azure_openai_api_key_is_secret_string() -> None:
    """Test that the API key is stored as a SecretStr."""
    llm = AzureOpenAI(
        openai_api_key="secret-api-key",
        azure_endpoint="endpoint",
        azure_ad_token="secret-ad-token",
        api_version="version",
    )
    assert isinstance(llm.openai_api_key, SecretStr)
    assert isinstance(llm.azure_ad_token, SecretStr)


def test_azure_openai_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test that the API key is masked when passed from an environment variable."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret-api-key")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "secret-ad-token")
    llm = AzureOpenAI(
        openai_api_key="secret-api-key",
        azure_endpoint="endpoint",
        azure_ad_token="secret-ad-token",
        api_version="version",
    )
    print(llm.openai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"

    print(llm.azure_ad_token, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_azure_openai_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test that the API key is masked when passed via the constructor."""
    llm = AzureOpenAI(
        openai_api_key="secret-api-key",
        azure_endpoint="endpoint",
        azure_ad_token="secret-ad-token",
        api_version="version",
    )
    print(llm.openai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"

    print(llm.azure_ad_token, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_azure_openai_uses_actual_secret_value_from_secretstr() -> None:
    """Test that the actual secret value is correctly retrieved."""
    llm = AzureOpenAI(
        openai_api_key="secret-api-key",
        azure_endpoint="endpoint",
        azure_ad_token="secret-ad-token",
        api_version="version",
    )
    assert cast(SecretStr, llm.openai_api_key).get_secret_value() == "secret-api-key"
    assert cast(SecretStr, llm.azure_ad_token).get_secret_value() == "secret-ad-token"


# azure openai embeddings
def test_azure_openai_embeddings_api_key_is_secret_string() -> None:
    """Test that the API key is stored as a SecretStr."""
    embeddings_model = AzureOpenAIEmbeddings(
        openai_api_key="secret-api-key",
        azure_endpoint="endpoint",
        azure_ad_token="secret-ad-token",
        api_version="version",
    )
    assert isinstance(embeddings_model.openai_api_key, SecretStr)
    assert isinstance(embeddings_model.azure_ad_token, SecretStr)


def test_azure_openai_embeddings_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test that the API key is masked when passed from an environment variable."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret-api-key")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "secret-ad-token")
    embeddings_model = AzureOpenAIEmbeddings(
        openai_api_key="secret-api-key",
        azure_endpoint="endpoint",
        azure_ad_token="secret-ad-token",
        api_version="version",
    )
    print(embeddings_model.openai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"

    print(embeddings_model.azure_ad_token, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_azure_openai_embeddings_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test that the API key is masked when passed via the constructor."""
    embeddings_model = AzureOpenAIEmbeddings(
        openai_api_key="secret-api-key",
        azure_endpoint="endpoint",
        azure_ad_token="secret-ad-token",
        api_version="version",
    )
    print(embeddings_model.openai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"

    print(embeddings_model.azure_ad_token, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_azure_openai_embeddings_uses_actual_secret_value_from_secretstr() -> None:
    """Test that the actual secret value is correctly retrieved."""
    embeddings_model = AzureOpenAIEmbeddings(
        openai_api_key="secret-api-key",
        azure_endpoint="endpoint",
        azure_ad_token="secret-ad-token",
        api_version="version",
    )
    assert (
        cast(SecretStr, embeddings_model.openai_api_key).get_secret_value()
        == "secret-api-key"
    )
    assert (
        cast(SecretStr, embeddings_model.azure_ad_token).get_secret_value()
        == "secret-ad-token"
    )


# openai chat
def test_chat_openai_api_key_is_secret_string() -> None:
    """Test that the API key is stored as a SecretStr."""
    chat_model = ChatOpenAI(openai_api_key="secret-api-key")
    assert isinstance(chat_model.openai_api_key, SecretStr)


def test_chat_openai_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test that the API key is masked when passed from an environment variable."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret-api-key")
    chat_model = ChatOpenAI(openai_api_key="secret-api-key")
    print(chat_model.openai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_chat_openai_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test that the API key is masked when passed via the constructor."""
    chat_model = ChatOpenAI(openai_api_key="secret-api-key")
    print(chat_model.openai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_chat_openai_uses_actual_secret_value_from_secretstr() -> None:
    """Test that the actual secret value is correctly retrieved."""
    chat_model = ChatOpenAI(openai_api_key="secret-api-key")
    assert (
        cast(SecretStr, chat_model.openai_api_key).get_secret_value()
        == "secret-api-key"
    )


# openai
def test_openai_api_key_is_secret_string() -> None:
    """Test that the API key is stored as a SecretStr."""
    llm = OpenAI(openai_api_key="secret-api-key")
    assert isinstance(llm.openai_api_key, SecretStr)


def test_openai_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test that the API key is masked when passed from an environment variable."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret-api-key")
    llm = OpenAI(openai_api_key="secret-api-key")
    print(llm.openai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_openai_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test that the API key is masked when passed via the constructor."""
    llm = OpenAI(openai_api_key="secret-api-key")
    print(llm.openai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_openai_uses_actual_secret_value_from_secretstr() -> None:
    """Test that the actual secret value is correctly retrieved."""
    llm = OpenAI(openai_api_key="secret-api-key")
    assert cast(SecretStr, llm.openai_api_key).get_secret_value() == "secret-api-key"


# openai embeddings
def test_openai_embeddings_api_key_is_secret_string() -> None:
    """Test that the API key is stored as a SecretStr."""
    embeddings_model = OpenAIEmbeddings(openai_api_key="secret-api-key")
    assert isinstance(embeddings_model.openai_api_key, SecretStr)


def test_openai_embeddings_api_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test that the API key is masked when passed from an environment variable."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret-api-key")
    embeddings_model = OpenAIEmbeddings(openai_api_key="secret-api-key")
    print(embeddings_model.openai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_openai_embeddings_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test that the API key is masked when passed via the constructor."""
    embeddings_model = OpenAIEmbeddings(openai_api_key="secret-api-key")
    print(embeddings_model.openai_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_openai_embeddings_uses_actual_secret_value_from_secretstr() -> None:
    """Test that the actual secret value is correctly retrieved."""
    embeddings_model = OpenAIEmbeddings(openai_api_key="secret-api-key")
    assert (
        cast(SecretStr, embeddings_model.openai_api_key).get_secret_value()
        == "secret-api-key"
    )
