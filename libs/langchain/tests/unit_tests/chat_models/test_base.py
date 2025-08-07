import os
from typing import Optional
from unittest import mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableSequence
from langchain_core.v1.chat_models import BaseChatModel as BaseChatModelV1
from pydantic import SecretStr

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
    "langchain_groq",
)
@pytest.mark.parametrize(
    ("model_name", "model_provider"),
    [
        ("gpt-4o", "openai"),
        ("claude-3-opus-20240229", "anthropic"),
        ("accounts/fireworks/models/mixtral-8x7b-instruct", "fireworks"),
        ("mixtral-8x7b-32768", "groq"),
    ],
)
def test_init_chat_model(model_name: str, model_provider: Optional[str]) -> None:
    llm1: BaseChatModel = init_chat_model(
        model_name,
        model_provider=model_provider,
        api_key="foo",
    )
    llm2: BaseChatModel = init_chat_model(
        f"{model_provider}:{model_name}",
        api_key="foo",
    )
    assert llm1.dict() == llm2.dict()


@pytest.mark.requires("langchain_openai")
def test_message_version() -> None:
    model = init_chat_model("openai:gpt-4.1", api_key="foo")
    assert isinstance(model, BaseChatModel)

    model_v1 = init_chat_model("openai:gpt-4.1", api_key="foo", message_version="v1")
    assert isinstance(model_v1, BaseChatModelV1)

    # Test we emit a warning for unsupported providers
    with (
        pytest.warns(match="Model provider bar does not support message_version=v1"),
        pytest.raises(ValueError, match="Unsupported model_provider='bar'."),
    ):
        init_chat_model("foo", model_provider="bar", message_version="v1")


def test_init_missing_dep() -> None:
    with pytest.raises(ImportError):
        init_chat_model("mixtral-8x7b-32768", model_provider="groq")


def test_init_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported model_provider='bar'."):
        init_chat_model("foo", model_provider="bar")


@pytest.mark.requires("langchain_openai")
@mock.patch.dict(
    os.environ,
    {"OPENAI_API_KEY": "foo", "ANTHROPIC_API_KEY": "bar"},
    clear=True,
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
        [{"name": "foo", "description": "foo", "parameters": {}}],
    )

    # Check that original model wasn't mutated by declarative operation.
    assert model._queued_declarative_operations == []

    # Can iteratively call declarative methods.
    model_with_config = model_with_tools.with_config(
        RunnableConfig(tags=["foo"]),
        configurable={"model": "gpt-4o"},
    )
    assert model_with_config.model_name == "gpt-4o"  # type: ignore[attr-defined]

    for method in ("get_num_tokens", "get_num_tokens_from_messages"):
        assert hasattr(model_with_config, method)

    assert model_with_config.model_dump() == {  # type: ignore[attr-defined]
        "name": None,
        "bound": {
            "name": None,
            "disable_streaming": False,
            "disabled_params": None,
            "model_name": "gpt-4o",
            "temperature": None,
            "model_kwargs": {},
            "openai_api_key": SecretStr("foo"),
            "openai_api_base": None,
            "openai_organization": None,
            "openai_proxy": None,
            "output_version": "v0",
            "request_timeout": None,
            "max_retries": None,
            "presence_penalty": None,
            "reasoning": None,
            "reasoning_effort": None,
            "frequency_penalty": None,
            "include": None,
            "seed": None,
            "service_tier": None,
            "logprobs": None,
            "top_logprobs": None,
            "logit_bias": None,
            "streaming": False,
            "n": None,
            "top_p": None,
            "truncation": None,
            "max_tokens": None,
            "tiktoken_model_name": None,
            "default_headers": None,
            "default_query": None,
            "stop": None,
            "store": None,
            "extra_body": None,
            "include_response_headers": False,
            "stream_usage": False,
            "use_previous_response_id": False,
            "use_responses_api": None,
        },
        "kwargs": {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "foo", "description": "foo", "parameters": {}},
                },
            ],
        },
        "config": {"tags": ["foo"], "configurable": {}},
        "config_factories": [],
        "custom_input_type": None,
        "custom_output_type": None,
    }


@pytest.mark.requires("langchain_openai", "langchain_anthropic")
@mock.patch.dict(
    os.environ,
    {"OPENAI_API_KEY": "foo", "ANTHROPIC_API_KEY": "bar"},
    clear=True,
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

    assert model.model_name == "gpt-4o"

    model_with_tools = model.bind_tools(
        [{"name": "foo", "description": "foo", "parameters": {}}],
    )

    model_with_config = model_with_tools.with_config(
        RunnableConfig(tags=["foo"]),
        configurable={"bar_model": "claude-3-7-sonnet-20250219"},
    )

    assert model_with_config.model == "claude-3-7-sonnet-20250219"  # type: ignore[attr-defined]

    assert model_with_config.model_dump() == {  # type: ignore[attr-defined]
        "name": None,
        "bound": {
            "name": None,
            "disable_streaming": False,
            "model": "claude-3-7-sonnet-20250219",
            "mcp_servers": None,
            "max_tokens": 1024,
            "temperature": None,
            "thinking": None,
            "top_k": None,
            "top_p": None,
            "default_request_timeout": None,
            "max_retries": 2,
            "stop_sequences": None,
            "anthropic_api_url": "https://api.anthropic.com",
            "anthropic_api_key": SecretStr("bar"),
            "betas": None,
            "default_headers": None,
            "model_kwargs": {},
            "streaming": False,
            "stream_usage": True,
        },
        "kwargs": {
            "tools": [{"name": "foo", "description": "foo", "input_schema": {}}],
        },
        "config": {"tags": ["foo"], "configurable": {}},
        "config_factories": [],
        "custom_input_type": None,
        "custom_output_type": None,
    }
    prompt = ChatPromptTemplate.from_messages([("system", "foo")])
    chain = prompt | model_with_config
    assert isinstance(chain, RunnableSequence)
