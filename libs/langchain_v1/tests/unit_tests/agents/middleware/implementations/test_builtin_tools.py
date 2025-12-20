"""Unit tests for BuiltinToolsMiddleware."""

from __future__ import annotations

import warnings
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from langchain.agents.middleware.builtin_tools import (
    BuiltinToolsMiddleware,
    _detect_provider,
)
from langchain.agents.middleware.types import ModelRequest, ModelResponse


class FakeAnthropicModel(BaseChatModel):
    """Fake Anthropic model for testing."""

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        return AIMessage(content="test")

    def _llm_type(self) -> str:
        return "fake-anthropic"


class FakeOpenAIModel(BaseChatModel):
    """Fake OpenAI model for testing."""

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        return AIMessage(content="test")

    def _llm_type(self) -> str:
        return "fake-openai"


class FakeGoogleModel(BaseChatModel):
    """Fake Google model for testing."""

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        return AIMessage(content="test")

    def _llm_type(self) -> str:
        return "fake-google"


class FakeXAIModel(BaseChatModel):
    """Fake xAI model for testing."""

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        return AIMessage(content="test")

    def _llm_type(self) -> str:
        return "fake-xai"


class FakeUnknownModel(BaseChatModel):
    """Fake unknown model for testing."""

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        return AIMessage(content="test")

    def _llm_type(self) -> str:
        return "fake-unknown"


class ChatAnthropic(FakeAnthropicModel):
    """Mock ChatAnthropic for provider detection."""


class ChatOpenAI(FakeOpenAIModel):
    """Mock ChatOpenAI for provider detection."""


class AzureChatOpenAI(FakeOpenAIModel):
    """Mock AzureChatOpenAI for provider detection."""


class ChatGoogleGenerativeAI(FakeGoogleModel):
    """Mock ChatGoogleGenerativeAI for provider detection."""


class ChatVertexAI(FakeGoogleModel):
    """Mock ChatVertexAI for provider detection."""


class ChatXAI(FakeXAIModel):
    """Mock ChatXAI for provider detection."""


def _mock_convert_to_openai(tool: dict[str, Any]) -> dict[str, Any] | None:
    """Mock OpenAI converter."""
    tool_type = tool.get("type")
    if tool_type == "web_search":
        result: dict[str, Any] = {"type": "web_search"}
        if "user_location" in tool:
            result["user_location"] = tool["user_location"]
        return result
    if tool_type == "code_execution":
        result = {"type": "code_interpreter"}
        result["container"] = tool.get("container", {"type": "auto"})
        return result
    if tool_type == "file_search":
        result = {"type": "file_search"}
        if "vector_store_ids" in tool:
            result["vector_store_ids"] = tool["vector_store_ids"]
        return result
    if tool_type == "image_generation":
        return {"type": "image_generation"}
    return None


def _mock_convert_to_anthropic(tool: dict[str, Any]) -> dict[str, Any] | None:
    """Mock Anthropic converter."""
    tool_type = tool.get("type")
    if tool_type == "web_search":
        result: dict[str, Any] = {"type": "web_search_20250305", "name": "web_search"}
        if "max_uses" in tool:
            result["max_uses"] = tool["max_uses"]
        if "user_location" in tool:
            result["user_location"] = tool["user_location"]
        return result
    if tool_type == "code_execution":
        return {"type": "code_execution_20250825", "name": "code_execution"}
    if tool_type == "web_fetch":
        result = {"type": "web_fetch_20250910", "name": "web_fetch"}
        if "max_uses" in tool:
            result["max_uses"] = tool["max_uses"]
        return result
    if tool_type == "memory":
        return {"type": "memory_20250818", "name": "memory"}
    if tool_type == "text_editor":
        return {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}
    if tool_type == "bash":
        return {"type": "bash_20250124", "name": "bash"}
    return None


def _mock_convert_to_xai(tool: dict[str, Any]) -> dict[str, Any] | None:
    """Mock xAI converter."""
    tool_type = tool.get("type")
    if tool_type == "web_search":
        return {"type": "web_search"}
    if tool_type == "code_execution":
        return {"type": "code_interpreter"}
    if tool_type == "x_search":
        return {"type": "x_search"}
    return None


@pytest.fixture(autouse=True)
def mock_conversion_utilities(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the conversion utilities from partner packages."""
    import sys

    # Mock the modules to make _check_pkg pass
    mock_openai = MagicMock()
    mock_openai.__spec__ = MagicMock()
    mock_openai_utils = MagicMock()
    mock_openai_utils_builtin = MagicMock()
    mock_openai_utils_builtin.convert_standard_to_openai = _mock_convert_to_openai

    mock_anthropic = MagicMock()
    mock_anthropic.__spec__ = MagicMock()
    mock_anthropic_utils = MagicMock()
    mock_anthropic_utils_builtin = MagicMock()
    mock_anthropic_utils_builtin.convert_standard_to_anthropic = _mock_convert_to_anthropic

    mock_xai = MagicMock()
    mock_xai.__spec__ = MagicMock()
    mock_xai_utils = MagicMock()
    mock_xai_utils_builtin = MagicMock()
    mock_xai_utils_builtin.convert_standard_to_xai = _mock_convert_to_xai

    monkeypatch.setitem(sys.modules, "langchain_openai", mock_openai)
    monkeypatch.setitem(sys.modules, "langchain_openai.utils", mock_openai_utils)
    monkeypatch.setitem(
        sys.modules, "langchain_openai.utils.builtin_tools", mock_openai_utils_builtin
    )

    monkeypatch.setitem(sys.modules, "langchain_anthropic", mock_anthropic)
    monkeypatch.setitem(sys.modules, "langchain_anthropic.utils", mock_anthropic_utils)
    monkeypatch.setitem(
        sys.modules, "langchain_anthropic.utils.builtin_tools", mock_anthropic_utils_builtin
    )

    monkeypatch.setitem(sys.modules, "langchain_xai", mock_xai)
    monkeypatch.setitem(sys.modules, "langchain_xai.utils", mock_xai_utils)
    monkeypatch.setitem(sys.modules, "langchain_xai.utils.builtin_tools", mock_xai_utils_builtin)


def test_detect_provider_anthropic() -> None:
    """Test provider detection for Anthropic models."""
    model = ChatAnthropic()
    assert _detect_provider(model) == "anthropic"


def test_detect_provider_openai() -> None:
    """Test provider detection for OpenAI models."""
    model = ChatOpenAI()
    assert _detect_provider(model) == "openai"


def test_detect_provider_azure_openai() -> None:
    """Test provider detection for Azure OpenAI models."""
    model = AzureChatOpenAI()
    assert _detect_provider(model) == "openai"


def test_detect_provider_google() -> None:
    """Test provider detection for Google models."""
    model1 = ChatGoogleGenerativeAI()
    model2 = ChatVertexAI()
    assert _detect_provider(model1) == "google_genai"
    assert _detect_provider(model2) == "google_genai"


def test_detect_provider_xai() -> None:
    """Test provider detection for xAI models."""
    model = ChatXAI()
    assert _detect_provider(model) == "xai"


def test_detect_provider_unknown() -> None:
    """Test provider detection for unknown models."""
    model = FakeUnknownModel()
    assert _detect_provider(model) == "unknown"


def test_builtin_tools_middleware_anthropic_web_search() -> None:
    """Test adding web_search tool for Anthropic."""
    middleware = BuiltinToolsMiddleware(include_tools=["web_search"])

    model = ChatAnthropic()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        # Verify the tool was added
        assert len(req.tools) == 1
        tool = req.tools[0]
        assert isinstance(tool, dict)
        assert tool["type"] == "web_search_20250305"
        assert tool["name"] == "web_search"
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_openai_web_search() -> None:
    """Test adding web_search tool for OpenAI."""
    middleware = BuiltinToolsMiddleware(include_tools=["web_search"])

    model = ChatOpenAI()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        assert len(req.tools) == 1
        tool = req.tools[0]
        assert isinstance(tool, dict)
        assert tool["type"] == "web_search"
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_anthropic_code_execution() -> None:
    """Test adding code_execution tool for Anthropic."""
    middleware = BuiltinToolsMiddleware(include_tools=["code_execution"])

    model = ChatAnthropic()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        assert len(req.tools) == 1
        tool = req.tools[0]
        assert isinstance(tool, dict)
        assert tool["type"] == "code_execution_20250825"
        assert tool["name"] == "code_execution"
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_openai_code_execution() -> None:
    """Test adding code_execution tool for OpenAI."""
    middleware = BuiltinToolsMiddleware(include_tools=["code_execution"])

    model = ChatOpenAI()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        assert len(req.tools) == 1
        tool = req.tools[0]
        assert isinstance(tool, dict)
        assert tool["type"] == "code_interpreter"
        assert tool["container"] == {"type": "auto"}
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_xai_web_search() -> None:
    """Test adding web_search tool for xAI/Grok."""
    middleware = BuiltinToolsMiddleware(include_tools=["web_search"])

    model = ChatXAI()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        assert len(req.tools) == 1
        tool = req.tools[0]
        assert isinstance(tool, dict)
        assert tool["type"] == "web_search"
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_xai_code_interpreter() -> None:
    """Test adding code_execution tool for xAI/Grok."""
    middleware = BuiltinToolsMiddleware(include_tools=["code_execution"])

    model = ChatXAI()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        assert len(req.tools) == 1
        tool = req.tools[0]
        assert isinstance(tool, dict)
        assert tool["type"] == "code_interpreter"
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_xai_x_search() -> None:
    """Test adding x_search tool for xAI/Grok."""
    middleware = BuiltinToolsMiddleware(include_tools=["x_search"])

    model = ChatXAI()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        assert len(req.tools) == 1
        tool = req.tools[0]
        assert isinstance(tool, dict)
        assert tool["type"] == "x_search"
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_xai_multiple_tools() -> None:
    """Test adding multiple tools for xAI/Grok including x_search."""
    middleware = BuiltinToolsMiddleware(include_tools=["web_search", "x_search", "code_execution"])

    model = ChatXAI()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        assert len(req.tools) == 3
        # Check web_search
        assert req.tools[0]["type"] == "web_search"
        # Check x_search
        assert req.tools[1]["type"] == "x_search"
        # Check code_execution
        assert req.tools[2]["type"] == "code_interpreter"
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_x_search_unsupported_on_other_providers() -> None:
    """Test that x_search is not supported on non-xAI providers."""
    middleware = BuiltinToolsMiddleware(
        include_tools=["x_search"],
        unsupported_behavior="remove",
    )

    # Test with OpenAI
    model = ChatOpenAI()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        # x_search should be removed for non-xAI providers
        assert len(req.tools) == 0
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_multiple_tools() -> None:
    """Test adding multiple builtin tools."""
    middleware = BuiltinToolsMiddleware(include_tools=["web_search", "code_execution"])

    model = ChatAnthropic()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        assert len(req.tools) == 2
        # Check web_search
        assert req.tools[0]["type"] == "web_search_20250305"
        # Check code_execution
        assert req.tools[1]["type"] == "code_execution_20250825"
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_with_existing_tools() -> None:
    """Test that builtin tools are added to existing tools."""
    from langchain_core.tools import tool

    @tool
    def my_custom_tool(query: str) -> str:
        """A custom tool."""
        return "result"

    middleware = BuiltinToolsMiddleware(include_tools=["web_search"])

    model = ChatAnthropic()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[my_custom_tool],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        # Should have custom tool + web_search
        assert len(req.tools) == 2
        assert req.tools[0] == my_custom_tool
        assert isinstance(req.tools[1], dict)
        assert req.tools[1]["type"] == "web_search_20250305"
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_with_options() -> None:
    """Test adding tools with custom options."""
    middleware = BuiltinToolsMiddleware(
        include_tools=[
            {"name": "web_fetch", "max_uses": 5},
        ]
    )

    model = ChatAnthropic()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        assert len(req.tools) == 1
        tool = req.tools[0]
        assert isinstance(tool, dict)
        assert tool["type"] == "web_fetch_20250910"
        assert tool["name"] == "web_fetch"
        assert tool["max_uses"] == 5
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_openai_file_search_with_options() -> None:
    """Test adding file_search with vector_store_ids for OpenAI."""
    middleware = BuiltinToolsMiddleware(
        include_tools=[
            {"name": "file_search", "vector_store_ids": ["vs_123", "vs_456"]},
        ]
    )

    model = ChatOpenAI()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        assert len(req.tools) == 1
        tool = req.tools[0]
        assert isinstance(tool, dict)
        assert tool["type"] == "file_search"
        assert tool["vector_store_ids"] == ["vs_123", "vs_456"]
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_unsupported_remove() -> None:
    """Test unsupported_behavior='remove' silently skips unsupported tools."""
    middleware = BuiltinToolsMiddleware(
        include_tools=["web_fetch"],  # Anthropic only
        unsupported_behavior="remove",
    )

    model = ChatOpenAI()  # OpenAI doesn't support web_fetch
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        # Should have no tools added
        assert len(req.tools) == 0
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_unsupported_warn() -> None:
    """Test unsupported_behavior='warn' logs warning and skips unsupported tools."""
    middleware = BuiltinToolsMiddleware(
        include_tools=["web_fetch"],  # Anthropic only
        unsupported_behavior="warn",
    )

    model = ChatOpenAI()  # OpenAI doesn't support web_fetch
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        assert len(req.tools) == 0
        return ModelResponse(result=[AIMessage(content="test")])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = middleware.wrap_model_call(request, handler)
        assert len(w) == 1
        assert "web_fetch" in str(w[0].message)
        assert "not supported" in str(w[0].message)

    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_unsupported_error() -> None:
    """Test unsupported_behavior='error' raises ValueError for unsupported tools."""
    middleware = BuiltinToolsMiddleware(
        include_tools=["web_fetch"],  # Anthropic only
        unsupported_behavior="error",
    )

    model = ChatOpenAI()  # OpenAI doesn't support web_fetch
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="test")])

    with pytest.raises(ValueError, match="not supported by openai"):
        middleware.wrap_model_call(request, handler)


def test_builtin_tools_middleware_partial_support() -> None:
    """Test that supported tools are added even when some are unsupported."""
    middleware = BuiltinToolsMiddleware(
        include_tools=["web_search", "web_fetch"],  # web_fetch is Anthropic only
        unsupported_behavior="remove",
    )

    model = ChatOpenAI()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        # Should only have web_search (web_fetch removed)
        assert len(req.tools) == 1
        assert req.tools[0]["type"] == "web_search"
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_unknown_provider() -> None:
    """Test that unknown providers don't add any tools."""
    middleware = BuiltinToolsMiddleware(include_tools=["web_search"])

    model = FakeUnknownModel()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        # Should have no tools added
        assert len(req.tools) == 0
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_google_tools() -> None:
    """Test adding tools for Google models."""
    middleware = BuiltinToolsMiddleware(include_tools=["web_search", "code_execution"])

    model = ChatGoogleGenerativeAI()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        assert len(req.tools) == 2
        # Check web_search
        assert "google_search" in req.tools[0]
        # Check code_execution
        assert "code_execution" in req.tools[1]
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_all_anthropic_tools() -> None:
    """Test all Anthropic-specific tools."""
    tools = ["web_search", "code_execution", "web_fetch", "memory", "text_editor", "bash"]
    middleware = BuiltinToolsMiddleware(include_tools=tools)

    model = ChatAnthropic()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        assert len(req.tools) == 6
        tool_types = [t["type"] for t in req.tools]
        assert "web_search_20250305" in tool_types
        assert "code_execution_20250825" in tool_types
        assert "web_fetch_20250910" in tool_types
        assert "memory_20250818" in tool_types
        assert "text_editor_20250728" in tool_types
        assert "bash_20250124" in tool_types
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_all_openai_tools() -> None:
    """Test all OpenAI-specific tools."""
    tools = ["web_search", "code_execution", "file_search", "image_generation"]
    middleware = BuiltinToolsMiddleware(include_tools=tools)

    model = ChatOpenAI()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        assert len(req.tools) == 4
        tool_types = [t["type"] for t in req.tools]
        assert "web_search" in tool_types
        assert "code_interpreter" in tool_types
        assert "file_search" in tool_types
        assert "image_generation" in tool_types
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_all_xai_tools() -> None:
    """Test all xAI/Grok-specific tools."""
    tools = ["web_search", "code_execution", "x_search"]
    middleware = BuiltinToolsMiddleware(include_tools=tools)

    model = ChatXAI()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        assert len(req.tools) == 3
        tool_types = [t["type"] for t in req.tools]
        assert "web_search" in tool_types
        assert "code_interpreter" in tool_types
        assert "x_search" in tool_types
        return ModelResponse(result=[AIMessage(content="test")])

    result = middleware.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


def test_builtin_tools_middleware_invalid_tool_spec() -> None:
    """Test that invalid tool spec raises ValueError."""
    middleware = BuiltinToolsMiddleware(
        include_tools=[{"invalid": "spec"}]  # Missing 'name' key
    )

    model = ChatAnthropic()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    def handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="test")])

    with pytest.raises(ValueError, match="must have 'name' key"):
        middleware.wrap_model_call(request, handler)


@pytest.mark.asyncio
async def test_builtin_tools_middleware_async() -> None:
    """Test async version of wrap_model_call."""
    middleware = BuiltinToolsMiddleware(include_tools=["web_search"])

    model = ChatAnthropic()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    async def handler(req: ModelRequest) -> ModelResponse:
        assert len(req.tools) == 1
        assert req.tools[0]["type"] == "web_search_20250305"
        return ModelResponse(result=[AIMessage(content="test")])

    result = await middleware.awrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)


@pytest.mark.asyncio
async def test_builtin_tools_middleware_async_unsupported_error() -> None:
    """Test async version raises error for unsupported tools."""
    middleware = BuiltinToolsMiddleware(
        include_tools=["web_fetch"],
        unsupported_behavior="error",
    )

    model = ChatOpenAI()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    async def handler(req: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="test")])

    with pytest.raises(ValueError, match="not supported by openai"):
        await middleware.awrap_model_call(request, handler)


@pytest.mark.asyncio
async def test_builtin_tools_middleware_async_xai() -> None:
    """Test async version with xAI/Grok including x_search."""
    middleware = BuiltinToolsMiddleware(include_tools=["web_search", "x_search", "code_execution"])

    model = ChatXAI()
    request = ModelRequest(
        model=model,
        messages=[],
        tools=[],
    )

    async def handler(req: ModelRequest) -> ModelResponse:
        assert len(req.tools) == 3
        assert req.tools[0]["type"] == "web_search"
        assert req.tools[1]["type"] == "x_search"
        assert req.tools[2]["type"] == "code_interpreter"
        return ModelResponse(result=[AIMessage(content="test")])

    result = await middleware.awrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)
