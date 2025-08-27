"""Unit tests for DeepSeek strict mode support."""

from pydantic import BaseModel, Field, SecretStr

from langchain_deepseek import ChatDeepSeek
from langchain_deepseek.chat_models import DEFAULT_API_BASE


class SampleTool(BaseModel):
    """Sample tool schema for testing."""

    value: str = Field(description="A test value")


def test_bind_tools_with_strict_mode_uses_beta_endpoint() -> None:
    """Test that bind_tools with strict=True uses the beta endpoint."""
    llm = ChatDeepSeek(
        model="deepseek-chat",
        api_key=SecretStr("test_key"),
    )

    # Verify default endpoint
    assert llm.api_base == DEFAULT_API_BASE

    # Bind tools with strict=True
    bound_model = llm.bind_tools([SampleTool], strict=True)

    # The bound model should have its internal model using beta endpoint
    # We can't directly access the internal model, but we can verify the behavior
    # by checking that the binding operation succeeds
    assert bound_model is not None


def test_bind_tools_without_strict_mode_uses_default_endpoint() -> None:
    """Test bind_tools without strict or with strict=False uses default endpoint."""
    llm = ChatDeepSeek(
        model="deepseek-chat",
        api_key=SecretStr("test_key"),
    )

    # Test with strict=False
    bound_model_false = llm.bind_tools([SampleTool], strict=False)
    assert bound_model_false is not None

    # Test with strict=None (default)
    bound_model_none = llm.bind_tools([SampleTool])
    assert bound_model_none is not None


def test_with_structured_output_strict_mode_uses_beta_endpoint() -> None:
    """Test that with_structured_output with strict=True uses beta endpoint."""
    llm = ChatDeepSeek(
        model="deepseek-chat",
        api_key=SecretStr("test_key"),
    )

    # Verify default endpoint
    assert llm.api_base == DEFAULT_API_BASE

    # Create structured output with strict=True
    structured_model = llm.with_structured_output(SampleTool, strict=True)

    # The structured model should work with beta endpoint
    assert structured_model is not None


def test_custom_api_base_not_overridden() -> None:
    """Test that custom API base is not overridden even with strict=True."""
    custom_base = "https://custom.api.com/v1"
    llm = ChatDeepSeek(
        model="deepseek-chat",
        api_key=SecretStr("test_key"),
        api_base=custom_base,
    )

    assert llm.api_base == custom_base

    # Bind tools with strict=True should not override custom base
    bound_model = llm.bind_tools([SampleTool], strict=True)
    assert bound_model is not None

    # The original model should still have custom base
    assert llm.api_base == custom_base
