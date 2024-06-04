import pytest

from langchain.chat_models.base import __all__, init_chat_model

EXPECTED_ALL = [
    "BaseChatModel",
    "SimpleChatModel",
    "agenerate_from_stream",
    "generate_from_stream",
    "init_chat_model",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)


@pytest.mark.requires(
    "langchain_openai",
    "langchain_anthropic",
    "langchain_fireworks",
    "langchain_together",
    "langchain_mistralai",
    "langchain_groq",
)
@pytest.mark.parametrize(
    ["model_name", "model_provider"],
    [
        ("gpt-4o", "openai"),
        ("claude-3-opus-20240229", "anthropic"),
        ("accounts/fireworks/models/mixtral-8x7b-instruct", "fireworks"),
        ("meta-llama/Llama-3-8b-chat-hf", "together"),
        ("mixtral-8x7b-32768", "groq"),
    ],
)
def test_init_chat_model(model_name: str, model_provider: str) -> None:
    init_chat_model(model_name, model_provider=model_provider, api_key="foo")


def test_init_missing_dep() -> None:
    with pytest.raises(ImportError):
        init_chat_model("gpt-4o", model_provider="openai")


def test_init_unknown_provider() -> None:
    with pytest.raises(ValueError):
        init_chat_model("foo", model_provider="bar")
