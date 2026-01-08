"""Tests for the `LargeToolResultMiddleware`."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from langchain_core.language_models import ModelProfile
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)

from langchain.agents.middleware.large_tool_result import (
    _OFFLOAD_METADATA_KEY,
    LargeToolResultMiddleware,
)
from langchain.agents.middleware.types import ModelRequest, ModelResponse

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


class ProfileChatModel(FakeChatModel):
    """Fake chat model with profile for testing."""

    profile: ModelProfile | None = ModelProfile(max_input_tokens=10000)


def _fake_runtime() -> Runtime:
    """Create a fake runtime for testing."""
    return None  # type: ignore[return-value]


def _make_request(
    messages: list[AnyMessage],
    *,
    model: FakeChatModel | None = None,
) -> ModelRequest:
    """Create a `ModelRequest` for testing."""
    model = model or ProfileChatModel()
    state = cast("dict", {"messages": messages})
    return ModelRequest(
        model=model,
        messages=list(messages),
        system_prompt=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state=state,
        runtime=_fake_runtime(),
        model_settings={},
    )


def test_initialization_with_defaults() -> None:
    """Test `LargeToolResultMiddleware` initialization with default values."""
    middleware = LargeToolResultMiddleware()
    assert middleware.threshold == ("fraction", 0.10)
    assert middleware.preview_length == 500
    assert middleware.user_temp_dir is None
    assert middleware.cleanup_on_end is True


def test_initialization_with_custom_values() -> None:
    """Test `LargeToolResultMiddleware` initialization with custom values."""
    middleware = LargeToolResultMiddleware(
        threshold=("tokens", 5000),
        preview_length=200,
        cleanup_on_end=False,
    )
    assert middleware.threshold == ("tokens", 5000)
    assert middleware.preview_length == 200
    assert middleware.cleanup_on_end is False


def test_initialization_with_temp_dir() -> None:
    """Test initialization with custom temp directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        middleware = LargeToolResultMiddleware(temp_dir=tmp_dir)
        assert middleware.user_temp_dir == Path(tmp_dir)


def test_validation_invalid_fraction_threshold() -> None:
    """Test validation of invalid fractional threshold."""
    with pytest.raises(ValueError, match="Fractional threshold must be between 0 and 1"):
        LargeToolResultMiddleware(threshold=("fraction", 0.0))

    with pytest.raises(ValueError, match="Fractional threshold must be between 0 and 1"):
        LargeToolResultMiddleware(threshold=("fraction", 1.5))


def test_validation_invalid_token_threshold() -> None:
    """Test validation of invalid token threshold."""
    with pytest.raises(ValueError, match="Token threshold must be greater than 0"):
        LargeToolResultMiddleware(threshold=("tokens", 0))

    with pytest.raises(ValueError, match="Token threshold must be greater than 0"):
        LargeToolResultMiddleware(threshold=("tokens", -100))


def test_no_offload_below_threshold() -> None:
    """Test that small tool results are not offloaded."""
    middleware = LargeToolResultMiddleware(threshold=("tokens", 1000))

    tool_message = ToolMessage(content="Small result", tool_call_id="call_123")
    messages: list[AnyMessage] = [
        HumanMessage(content="Hello"),
        AIMessage(
            content="I'll use a tool", tool_calls=[{"name": "test", "args": {}, "id": "call_123"}]
        ),
        tool_message,
    ]

    request = _make_request(messages)

    modified_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock")])

    middleware.wrap_model_call(request, mock_handler)

    assert modified_request is not None
    # Tool message should be unchanged
    result_tool_msg = modified_request.messages[2]
    assert isinstance(result_tool_msg, ToolMessage)
    assert result_tool_msg.content == "Small result"
    assert not result_tool_msg.response_metadata.get(_OFFLOAD_METADATA_KEY)


def test_offload_large_tool_result() -> None:
    """Test that large tool results are offloaded to disk."""
    middleware = LargeToolResultMiddleware(
        threshold=("tokens", 10),  # Very low threshold
        preview_length=20,
    )

    # Create a large tool result (way over 10 tokens)
    large_content = "A" * 1000
    tool_message = ToolMessage(content=large_content, tool_call_id="call_abc123")
    messages: list[AnyMessage] = [
        HumanMessage(content="Hello"),
        AIMessage(
            content="I'll use a tool",
            tool_calls=[{"name": "test", "args": {}, "id": "call_abc123"}],
        ),
        tool_message,
    ]

    request = _make_request(messages)

    modified_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock")])

    middleware.wrap_model_call(request, mock_handler)

    assert modified_request is not None
    result_tool_msg = modified_request.messages[2]
    assert isinstance(result_tool_msg, ToolMessage)

    # Content should be truncated
    assert "[TRUNCATED - Full result saved to:" in result_tool_msg.content
    assert "Preview (first 20 chars):" in result_tool_msg.content
    assert "A" * 20 in result_tool_msg.content

    # Metadata should be set
    offload_metadata = result_tool_msg.response_metadata.get(_OFFLOAD_METADATA_KEY)
    assert offload_metadata is not None
    assert offload_metadata["offloaded"] is True
    assert "file_path" in offload_metadata
    assert offload_metadata["original_size_chars"] == 1000

    # Verify file exists and contains original content
    file_path = Path(offload_metadata["file_path"])
    assert file_path.exists()
    assert file_path.read_text() == large_content

    # Cleanup
    middleware.after_agent({"messages": []}, _fake_runtime())


def test_preview_content_preserved() -> None:
    """Test that preview contains the beginning of the content."""
    middleware = LargeToolResultMiddleware(
        threshold=("tokens", 10),
        preview_length=50,
    )

    content = "This is the beginning of a very long message." + "X" * 1000
    tool_message = ToolMessage(content=content, tool_call_id="call_xyz")
    messages: list[AnyMessage] = [tool_message]

    request = _make_request(messages)

    modified_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock")])

    middleware.wrap_model_call(request, mock_handler)

    assert modified_request is not None
    result_tool_msg = modified_request.messages[0]
    assert isinstance(result_tool_msg, ToolMessage)

    # Preview should contain the beginning of the content
    assert "This is the beginning of a very long message." in result_tool_msg.content

    # Cleanup
    middleware.after_agent({"messages": []}, _fake_runtime())


def test_temp_dir_cleanup_on_agent_end() -> None:
    """Test that temp directory is cleaned up when agent ends."""
    middleware = LargeToolResultMiddleware(
        threshold=("tokens", 10),
        cleanup_on_end=True,
    )

    # Trigger offload to create temp dir
    large_content = "A" * 1000
    tool_message = ToolMessage(content=large_content, tool_call_id="call_cleanup")
    messages: list[AnyMessage] = [tool_message]

    request = _make_request(messages)

    modified_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock")])

    middleware.wrap_model_call(request, mock_handler)

    assert modified_request is not None
    result_tool_msg = modified_request.messages[0]
    offload_metadata = result_tool_msg.response_metadata.get(_OFFLOAD_METADATA_KEY)
    file_path = Path(offload_metadata["file_path"])
    temp_dir = file_path.parent

    # Verify file exists before cleanup
    assert file_path.exists()
    assert temp_dir.exists()

    # Call after_agent to trigger cleanup
    middleware.after_agent({"messages": []}, _fake_runtime())

    # Temp dir should be cleaned up
    assert not temp_dir.exists()


def test_custom_temp_dir_not_deleted() -> None:
    """Test that user-provided temp dir is not deleted on cleanup."""
    with tempfile.TemporaryDirectory() as user_dir:
        user_path = Path(user_dir)
        middleware = LargeToolResultMiddleware(
            threshold=("tokens", 10),
            temp_dir=user_dir,
            cleanup_on_end=True,
        )

        # Trigger offload
        large_content = "A" * 1000
        tool_message = ToolMessage(content=large_content, tool_call_id="call_custom")
        messages: list[AnyMessage] = [tool_message]

        request = _make_request(messages)

        def mock_handler(req: ModelRequest) -> ModelResponse:
            return ModelResponse(result=[AIMessage(content="mock")])

        middleware.wrap_model_call(request, mock_handler)

        # Call after_agent
        middleware.after_agent({"messages": []}, _fake_runtime())

        # User-provided dir should still exist
        assert user_path.exists()


def test_multiple_large_results_in_turn() -> None:
    """Test handling of multiple large tool results in parallel tool calls."""
    middleware = LargeToolResultMiddleware(
        threshold=("tokens", 10),
        preview_length=10,
    )

    messages: list[AnyMessage] = [
        HumanMessage(content="Process these files"),
        AIMessage(
            content="I'll process all files",
            tool_calls=[
                {"name": "read_file", "args": {"path": "file1.txt"}, "id": "call_1"},
                {"name": "read_file", "args": {"path": "file2.txt"}, "id": "call_2"},
                {"name": "read_file", "args": {"path": "file3.txt"}, "id": "call_3"},
            ],
        ),
        ToolMessage(content="X" * 500, tool_call_id="call_1"),
        ToolMessage(content="Y" * 500, tool_call_id="call_2"),
        ToolMessage(content="Z" * 500, tool_call_id="call_3"),
    ]

    request = _make_request(messages)

    modified_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock")])

    middleware.wrap_model_call(request, mock_handler)

    assert modified_request is not None

    # All three tool messages should be offloaded
    for i in range(2, 5):
        tool_msg = modified_request.messages[i]
        assert isinstance(tool_msg, ToolMessage)
        assert "[TRUNCATED - Full result saved to:" in tool_msg.content
        offload_metadata = tool_msg.response_metadata.get(_OFFLOAD_METADATA_KEY)
        assert offload_metadata is not None
        assert offload_metadata["offloaded"] is True

    # Each should have a unique file
    file_paths = set()
    for i in range(2, 5):
        tool_msg = modified_request.messages[i]
        offload_metadata = tool_msg.response_metadata.get(_OFFLOAD_METADATA_KEY)
        file_paths.add(offload_metadata["file_path"])

    assert len(file_paths) == 3  # Three unique files

    # Cleanup
    middleware.after_agent({"messages": []}, _fake_runtime())


def test_already_offloaded_skipped() -> None:
    """Test that already offloaded messages are not re-processed."""
    middleware = LargeToolResultMiddleware(
        threshold=("tokens", 10),
    )

    # Pre-offloaded message
    tool_message = ToolMessage(
        content="[TRUNCATED - Full result saved to: /path/to/file.txt]\n\nPreview...",
        tool_call_id="call_already",
        response_metadata={
            _OFFLOAD_METADATA_KEY: {
                "offloaded": True,
                "file_path": "/path/to/file.txt",
                "original_size_chars": 10000,
            }
        },
    )

    messages: list[AnyMessage] = [tool_message]
    request = _make_request(messages)

    modified_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock")])

    middleware.wrap_model_call(request, mock_handler)

    assert modified_request is not None
    result_tool_msg = modified_request.messages[0]
    assert isinstance(result_tool_msg, ToolMessage)

    # Content should be unchanged (not re-processed)
    assert result_tool_msg.content == tool_message.content


def test_fraction_threshold_with_model_profile() -> None:
    """Test fractional threshold calculation with model profile."""
    middleware = LargeToolResultMiddleware(
        threshold=("fraction", 0.10),  # 10% of 10000 = 1000 tokens
    )

    # Content that is about 1200 tokens (4800 chars / 4 chars per token estimate)
    # This should exceed 10% of 10000 tokens
    large_content = "A" * 4800
    tool_message = ToolMessage(content=large_content, tool_call_id="call_frac")
    messages: list[AnyMessage] = [tool_message]

    # Use model with profile
    model = ProfileChatModel()
    request = _make_request(messages, model=model)

    modified_request = None

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock")])

    middleware.wrap_model_call(request, mock_handler)

    assert modified_request is not None
    result_tool_msg = modified_request.messages[0]
    assert isinstance(result_tool_msg, ToolMessage)

    # Should be offloaded due to exceeding 10% threshold
    assert "[TRUNCATED - Full result saved to:" in result_tool_msg.content

    # Cleanup
    middleware.after_agent({"messages": []}, _fake_runtime())


async def test_async_wrap_model_call() -> None:
    """Test async version of `wrap_model_call`."""
    middleware = LargeToolResultMiddleware(
        threshold=("tokens", 10),
        preview_length=20,
    )

    large_content = "A" * 1000
    tool_message = ToolMessage(content=large_content, tool_call_id="call_async")
    messages: list[AnyMessage] = [tool_message]

    request = _make_request(messages)

    modified_request = None

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal modified_request
        modified_request = req
        return ModelResponse(result=[AIMessage(content="mock")])

    await middleware.awrap_model_call(request, mock_handler)

    assert modified_request is not None
    result_tool_msg = modified_request.messages[0]
    assert isinstance(result_tool_msg, ToolMessage)
    assert "[TRUNCATED - Full result saved to:" in result_tool_msg.content

    # Cleanup
    await middleware.aafter_agent({"messages": []}, _fake_runtime())


def test_empty_messages_passthrough() -> None:
    """Test that empty messages list is handled correctly."""
    middleware = LargeToolResultMiddleware(threshold=("tokens", 10))

    request = _make_request([])

    called = False

    def mock_handler(req: ModelRequest) -> ModelResponse:
        nonlocal called
        called = True
        return ModelResponse(result=[AIMessage(content="mock")])

    middleware.wrap_model_call(request, mock_handler)

    assert called
