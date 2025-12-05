import os
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableSequence
from pydantic import SecretStr

from langchain.chat_models import __all__, init_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

EXPECTED_ALL = [
    "init_chat_model",
    "BaseChatModel",
]


def test_all_imports() -> None:
    """Test that all expected imports are present in the module's __all__."""
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
        ("claude-opus-4-1", "anthropic"),
        ("accounts/fireworks/models/mixtral-8x7b-instruct", "fireworks"),
        ("mixtral-8x7b-32768", "groq"),
    ],
)
def test_init_chat_model(model_name: str, model_provider: str | None) -> None:
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
    """Test configurable chat model behavior without default parameters.

    Verifies that a configurable chat model initialized without default parameters:
    - Has access to all standard runnable methods (`invoke`, `stream`, etc.)
    - Blocks access to non-configurable methods until configuration is provided
    - Supports declarative operations (`bind_tools`) without mutating original model
    - Can chain declarative operations and configuration to access full functionality
    - Properly resolves to the configured model type when parameters are provided

    Example:
    ```python
    # This creates a configurable model without specifying which model
    model = init_chat_model()

    # This will FAIL - no model specified yet
    model.get_num_tokens("hello")  # AttributeError!

    # This works - provides model at runtime
    response = model.invoke("Hello", config={"configurable": {"model": "gpt-4o"}})
    ```
    """
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
            "output_version": None,
            "request_timeout": None,
            "max_retries": None,
            "presence_penalty": None,
            "reasoning": None,
            "reasoning_effort": None,
            "verbosity": None,
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
            "stream_usage": True,
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
    """Test configurable chat model behavior with default parameters.

    Verifies that a configurable chat model initialized with default parameters:
    - Has access to all standard runnable methods (`invoke`, `stream`, etc.)
    - Provides immediate access to non-configurable methods (e.g. `get_num_tokens`)
    - Supports model switching through runtime configuration using `config_prefix`
    - Maintains proper model identity and attributes when reconfigured
    - Can be used in chains with different model providers via configuration

    Example:
    ```python
    # This creates a configurable model with default parameters (model)
    model = init_chat_model("gpt-4o", configurable_fields="any", config_prefix="bar")

    # This works immediately - uses default gpt-4o
    tokens = model.get_num_tokens("hello")

    # This also works - switches to Claude at runtime
    response = model.invoke(
        "Hello", config={"configurable": {"my_model_model": "claude-3-sonnet-20240229"}}
    )
    ```
    """
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
        configurable={"bar_model": "claude-sonnet-4-5-20250929"},
    )

    assert model_with_config.model == "claude-sonnet-4-5-20250929"  # type: ignore[attr-defined]

    assert model_with_config.model_dump() == {  # type: ignore[attr-defined]
        "name": None,
        "bound": {
            "name": None,
            "disable_streaming": False,
            "effort": None,
            "model": "claude-sonnet-4-5-20250929",
            "mcp_servers": None,
            "max_tokens": 64000,
            "temperature": None,
            "thinking": None,
            "top_k": None,
            "top_p": None,
            "default_request_timeout": None,
            "max_retries": 2,
            "stop_sequences": None,
            "anthropic_api_url": "https://api.anthropic.com",
            "anthropic_proxy": None,
            "context_management": None,
            "anthropic_api_key": SecretStr("bar"),
            "betas": None,
            "default_headers": None,
            "model_kwargs": {},
            "streaming": False,
            "stream_usage": True,
            "output_version": None,
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


def test_init_chat_model_input_validation() -> None:
    """Test input validation for init_chat_model function.

    Validates that the function properly checks input types and values:
    - model parameter must be a string or None
    - model_provider parameter must be a string or None
    - Empty strings are rejected
    - configurable_fields must be properly typed
    """
    # Test invalid model type
    with pytest.raises(TypeError, match="model must be a string"):
        init_chat_model(model=123)  # type: ignore[arg-type]

    # Test invalid model_provider type
    with pytest.raises(TypeError, match="model_provider must be a string"):
        init_chat_model(model="gpt-4o", model_provider=456)  # type: ignore[arg-type]

    # Test empty model string
    with pytest.raises(ValueError, match="model cannot be an empty string"):
        init_chat_model(model="")

    # Test empty model_provider string
    with pytest.raises(ValueError, match="model_provider cannot be an empty string"):
        init_chat_model(model="gpt-4o", model_provider="")

    # Test model string with only whitespace
    with pytest.raises(ValueError, match="model cannot be an empty string"):
        init_chat_model(model="   ")

    # Test invalid configurable_fields type
    with pytest.raises(TypeError, match="configurable_fields must be 'any', list, or tuple"):
        init_chat_model(model="gpt-4o", configurable_fields="invalid")  # type: ignore[arg-type]

    # Test configurable_fields with non-string elements
    with pytest.raises(TypeError, match="All configurable_fields must be strings"):
        init_chat_model(model="gpt-4o", configurable_fields=["model", 123])  # type: ignore[list-item]

    # Test configurable_fields with empty string elements
    with pytest.raises(ValueError, match="configurable_fields cannot contain empty strings"):
        init_chat_model(model="gpt-4o", configurable_fields=["model", ""])


def test_model_inference_patterns() -> None:
    """Test enhanced model inference patterns.
    
    Validates model inference by testing the public init_chat_model function
    with various model naming patterns to ensure providers are correctly inferred.
    """
    # Test that model inference works correctly through the public API
    # We test error messages to verify inference is working
    
    # OpenAI models - should be inferred correctly (won't raise ValueError for provider)
    # We expect ImportError since we don't have the actual packages installed
    with pytest.raises(ImportError, match="langchain_openai"):
        init_chat_model("gpt-4o")
    
    with pytest.raises(ImportError, match="langchain_openai"):
        init_chat_model("o1-preview")
    
    with pytest.raises(ImportError, match="langchain_openai"):  
        init_chat_model("o3-mini")
    
    # Anthropic models
    with pytest.raises(ImportError, match="langchain_anthropic"):
        init_chat_model("claude-3-sonnet-20240229")
    
    # Mistral models (including mixtral)
    with pytest.raises(ImportError, match="langchain_mistralai"):
        init_chat_model("mistral-7b-instruct")
    
    with pytest.raises(ImportError, match="langchain_mistralai"):
        init_chat_model("mixtral-8x7b-instruct")
    
    # Bedrock models (multiple patterns)
    with pytest.raises(ImportError, match="langchain_aws"):
        init_chat_model("amazon.titan-text-lite-v1")
    
    # Other providers
    with pytest.raises(ImportError, match="langchain_cohere"):
        init_chat_model("command-light")
    
    with pytest.raises(ImportError, match="langchain_google_vertexai"):
        init_chat_model("gemini-pro")
    
    with pytest.raises(ImportError, match="langchain_deepseek"):
        init_chat_model("deepseek-chat")
    
    with pytest.raises(ImportError, match="langchain_xai"):
        init_chat_model("grok-1")
    
    with pytest.raises(ImportError, match="langchain_perplexity"):
        init_chat_model("sonar-medium-online")
    
    with pytest.raises(ImportError, match="langchain_upstage"):
        init_chat_model("solar-1-mini-chat")
    
    # Unknown models should raise ValueError for provider inference
    with pytest.raises(ValueError, match="Unable to infer model provider"):
        init_chat_model("unknown-model-123")


def test_enhanced_error_messages() -> None:
    """Test enhanced error messages for better user experience.

    Validates that error messages:
    - Provide helpful suggestions and examples
    - Include links to documentation
    - List supported providers in sorted order
    - Give specific guidance for common issues
    """
    # Test error message for unknown model provider
    with pytest.raises(ValueError) as exc_info:
        init_chat_model("unknown-model-xyz")

    error_msg = str(exc_info.value)
    # Check that error message contains helpful information
    assert "Unable to infer model provider" in error_msg
    assert "Supported providers:" in error_msg
    assert "Examples:" in error_msg
    assert "https://docs.langchain.com" in error_msg
    assert "init_chat_model('unknown-model-xyz', model_provider=" in error_msg

    # Test error message for unsupported provider
    with pytest.raises(ValueError) as exc_info:
        init_chat_model("test-model", model_provider="unsupported_provider")

    error_msg = str(exc_info.value)
    assert "Unsupported model_provider='unsupported_provider'" in error_msg
    assert "Supported providers are:" in error_msg
    assert "https://docs.langchain.com/oss/python/integrations/providers/" in error_msg
    assert "https://docs.langchain.com/oss/python/contributing/" in error_msg


def test_provider_colon_format_parsing() -> None:
    """Test parsing of provider:model format with various edge cases.
    
    Validates that provider:model parsing works through the public API:
    - Works with standard formats
    - Handles models with colons in their names  
    - Properly validates provider names
    """
    # Standard format - should work (will fail at ImportError, not ValueError)
    with pytest.raises(ImportError, match="langchain_openai"):
        init_chat_model("openai:gpt-4o")
    
    # Model with multiple colons - should work
    with pytest.raises(ImportError, match="langchain_openai"):
        init_chat_model("openai:custom:model:v1")
    
    # Provider normalization (dash to underscore) should work
    with pytest.raises(ImportError, match="langchain_openai"):
        init_chat_model("azure-openai:gpt-4")
    
    # Invalid provider should raise ValueError
    with pytest.raises(ValueError, match="Unsupported model_provider"):
        init_chat_model("invalid_provider:some-model")
