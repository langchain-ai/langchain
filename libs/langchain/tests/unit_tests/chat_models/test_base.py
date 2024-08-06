import os
from unittest import mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableSequence

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
    _: BaseChatModel = init_chat_model(
        model_name, model_provider=model_provider, api_key="foo"
    )


def test_init_missing_dep() -> None:
    with pytest.raises(ImportError):
        init_chat_model("mixtral-8x7b-32768", model_provider="groq")


def test_init_unknown_provider() -> None:
    with pytest.raises(ValueError):
        init_chat_model("foo", model_provider="bar")


@pytest.mark.requires("langchain_openai")
@mock.patch.dict(
    os.environ, {"OPENAI_API_KEY": "foo", "ANTHROPIC_API_KEY": "foo"}, clear=True
)
def test_configurable() -> None:
    model = init_chat_model()

    for method in (
        "invoke",
        "ainvoke",
        "batch",
        "abatch",
        "stream",
        "astream",
        "batch_as_completed",
        "abatch_as_completed",
    ):
        assert hasattr(model, method)

    # Doesn't have access non-configurable, non-declarative methods until a config is
    # provided.
    for method in ("get_num_tokens", "get_num_tokens_from_messages"):
        with pytest.raises(AttributeError):
            getattr(model, method)

    # Can call declarative methods even without a default model.
    model_with_tools = model.bind_tools(
        [{"name": "foo", "description": "foo", "parameters": {}}]
    )

    # Check that original model wasn't mutated by declarative operation.
    assert model._queued_declarative_operations == []

    # Can iteratively call declarative methods.
    model_with_config = model_with_tools.with_config(
        RunnableConfig(tags=["foo"]), configurable={"model": "gpt-4o"}
    )
    assert model_with_config.model_name == "gpt-4o"  # type: ignore[attr-defined]

    for method in ("get_num_tokens", "get_num_tokens_from_messages"):
        assert hasattr(model_with_config, method)

    assert model_with_config.dict() == {  # type: ignore[attr-defined]
        "name": None,
        "bound": {
            "model_name": "gpt-4o",
            "model": "gpt-4o",
            "stream": False,
            "n": 1,
            "logprobs": False,
            "temperature": 0.7,
            "_type": "openai-chat",
        },
        "kwargs": {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "foo",
                        "description": "foo",
                        "parameters": {},
                    },
                }
            ]
        },
        "config": {"tags": ["foo"], "configurable": {}},
        "config_factories": [],
        "custom_input_type": None,
        "custom_output_type": None,
    }


@pytest.mark.requires("langchain_openai", "langchain_anthropic")
@mock.patch.dict(
    os.environ, {"OPENAI_API_KEY": "foo", "ANTHROPIC_API_KEY": "foo"}, clear=True
)
def test_configurable_with_default() -> None:
    model = init_chat_model("gpt-4o", configurable_fields="any", config_prefix="bar")
    for method in (
        "invoke",
        "ainvoke",
        "batch",
        "abatch",
        "stream",
        "astream",
        "batch_as_completed",
        "abatch_as_completed",
    ):
        assert hasattr(model, method)

    # Does have access non-configurable, non-declarative methods since default params
    # are provided.
    for method in ("get_num_tokens", "get_num_tokens_from_messages", "dict"):
        assert hasattr(model, method)

    assert model.model_name == "gpt-4o"  # type: ignore[attr-defined]

    model_with_tools = model.bind_tools(
        [{"name": "foo", "description": "foo", "parameters": {}}]
    )

    model_with_config = model_with_tools.with_config(
        RunnableConfig(tags=["foo"]),
        configurable={"bar_model": "claude-3-sonnet-20240229"},
    )

    assert model_with_config.model == "claude-3-sonnet-20240229"  # type: ignore[attr-defined]
    # Anthropic defaults to using `transformers` for token counting.
    with pytest.raises(ImportError):
        model_with_config.get_num_tokens_from_messages([(HumanMessage("foo"))])  # type: ignore[attr-defined]

    assert model_with_config.dict() == {  # type: ignore[attr-defined]
        "name": None,
        "bound": {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024,
            "temperature": None,
            "top_k": None,
            "top_p": None,
            "model_kwargs": {},
            "streaming": False,
            "max_retries": 2,
            "default_request_timeout": None,
            "_type": "anthropic-chat",
        },
        "kwargs": {
            "tools": [{"name": "foo", "description": "foo", "input_schema": {}}]
        },
        "config": {"tags": ["foo"], "configurable": {}},
        "config_factories": [],
        "custom_input_type": None,
        "custom_output_type": None,
    }
    prompt = ChatPromptTemplate.from_messages([("system", "foo")])
    chain = prompt | model_with_config
    assert isinstance(chain, RunnableSequence)
