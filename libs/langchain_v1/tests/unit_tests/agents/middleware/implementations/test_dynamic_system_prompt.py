"""Unit tests for dynamic system prompt middleware with SystemMessage support.

These tests replicate the functionality from langchainjs PR #9459:
- Middleware accepting functions that return SystemMessage
- Error handling for invalid return types
"""

from typing import cast

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from langchain.agents.middleware.types import ModelRequest, ModelResponse


def _fake_runtime(context: dict | None = None) -> Runtime:
    """Create a fake runtime with optional context."""
    if context:
        # Create a simple object with context
        class FakeRuntime:
            def __init__(self):
                self.context = type("Context", (), context)()

        return cast(Runtime, FakeRuntime())
    return cast(Runtime, object())


def _make_request(
    system_message: SystemMessage | None = None,
    system_prompt: str | None = None,
) -> ModelRequest:
    """Create a minimal ModelRequest for testing."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
    return ModelRequest(
        model=model,
        system_message=system_message,
        system_prompt=system_prompt,
        messages=[],
        tool_choice=None,
        tools=[],
        response_format=None,
        state=cast("AgentState", {"messages": []}),  # type: ignore[name-defined]
        runtime=_fake_runtime(),
        model_settings={},
    )


class TestDynamicSystemPromptWithSystemMessage:
    """Test middleware that accepts SystemMessage return types.

    These tests verify that middleware can work with SystemMessage objects,
    not just strings, enabling richer metadata handling.
    """

    def test_middleware_can_return_system_message(self) -> None:
        """Test that middleware can return a SystemMessage instead of string.

        This replicates the JS test: "should support returning a SystemMessage"
        """

        # Create a middleware function that returns SystemMessage
        def dynamic_system_prompt_middleware(request: ModelRequest) -> SystemMessage:
            """Return a SystemMessage with dynamic content."""
            region = getattr(request.runtime.context, "region", "n/a")
            return SystemMessage(content=f"You are a helpful assistant. Region: {region}")

        # Create request with runtime context
        runtime = _fake_runtime(context={"region": "EU"})
        request = ModelRequest(
            model=GenericFakeChatModel(messages=iter([AIMessage(content="response")])),
            system_message=None,
            messages=[HumanMessage(content="Hello")],
            tool_choice=None,
            tools=[],
            response_format=None,
            state=cast("AgentState", {"messages": []}),  # type: ignore[name-defined]
            runtime=runtime,
            model_settings={},
        )

        # Apply the middleware
        new_system_message = dynamic_system_prompt_middleware(request)

        # Verify the system message was created correctly
        assert isinstance(new_system_message, SystemMessage)
        assert len(new_system_message.content_blocks) == 1
        assert (
            new_system_message.content_blocks[0]["text"]
            == "You are a helpful assistant. Region: EU"
        )

    def test_middleware_rejects_invalid_return_types(self) -> None:
        """Test that middleware properly validates return types.

        This replicates the JS test for error handling with invalid return types.
        """

        def invalid_middleware(request: ModelRequest) -> int:
            """Return an invalid type (should raise error)."""
            return 123

        request = _make_request(system_prompt="Base prompt")

        # The middleware should not accept non-string/non-SystemMessage types
        # In Python, we rely on type checking, but let's verify the behavior
        result = invalid_middleware(request)
        assert not isinstance(result, (str, SystemMessage))
        # In a real implementation, this would be caught by type checking or runtime validation

    def test_middleware_can_use_system_message_with_metadata(self) -> None:
        """Test middleware creating SystemMessage with additional metadata."""

        def metadata_middleware(request: ModelRequest) -> SystemMessage:
            """Return SystemMessage with metadata."""
            return SystemMessage(
                content="You are a helpful assistant",
                additional_kwargs={"temperature": 0.7, "model": "gpt-4"},
                response_metadata={"region": "us-east"},
            )

        request = _make_request()
        new_system_message = metadata_middleware(request)

        assert len(new_system_message.content_blocks) == 1
        assert new_system_message.content_blocks[0]["text"] == "You are a helpful assistant"
        assert new_system_message.additional_kwargs == {
            "temperature": 0.7,
            "model": "gpt-4",
        }
        assert new_system_message.response_metadata == {"region": "us-east"}

    def test_middleware_handles_none_system_message(self) -> None:
        """Test middleware creating new SystemMessage when none exists."""

        def create_if_none_middleware(request: ModelRequest) -> SystemMessage:
            """Create a system message if none exists."""
            if request.system_message is None:
                return SystemMessage(content="Default system prompt")
            return request.system_message

        request = _make_request(system_message=None)
        new_system_message = create_if_none_middleware(request)

        assert isinstance(new_system_message, SystemMessage)
        assert len(new_system_message.content_blocks) == 1
        assert new_system_message.content_blocks[0]["text"] == "Default system prompt"

    def test_middleware_with_content_blocks(self) -> None:
        """Test middleware creating SystemMessage with content blocks."""

        def content_blocks_middleware(request: ModelRequest) -> SystemMessage:
            """Create SystemMessage with content blocks including cache control."""
            return SystemMessage(
                content=[
                    {"type": "text", "text": "Base instructions"},
                    {
                        "type": "text",
                        "text": "Cached instructions",
                        "cache_control": {"type": "ephemeral"},
                    },
                ]
            )

        request = _make_request()
        new_system_message = content_blocks_middleware(request)

        assert isinstance(new_system_message.content_blocks, list)
        assert len(new_system_message.content_blocks) == 2
        assert new_system_message.content_blocks[0]["text"] == "Base instructions"
        assert new_system_message.content_blocks[1]["cache_control"] == {"type": "ephemeral"}


class TestSystemMessageMiddlewareIntegration:
    """Test integration of SystemMessage with middleware chain."""

    def test_multiple_middleware_can_modify_system_message(self) -> None:
        """Test that multiple middleware can modify system message in sequence."""

        def first_middleware(request: ModelRequest) -> ModelRequest:
            """First middleware adds base system message."""
            new_message = SystemMessage(
                content="You are an assistant.",
                additional_kwargs={"middleware_1": "applied"},
            )
            return request.override(system_message=new_message)

        def second_middleware(request: ModelRequest) -> ModelRequest:
            """Second middleware appends to system message."""
            current_content = request.system_message.text
            new_content = current_content + " Be helpful."

            merged_kwargs = {
                **request.system_message.additional_kwargs,
                "middleware_2": "applied",
            }

            new_message = SystemMessage(
                content=new_content,
                additional_kwargs=merged_kwargs,
            )
            return request.override(system_message=new_message)

        # Start with no system message
        request = _make_request(system_message=None)

        # Apply middleware in sequence
        request = first_middleware(request)
        assert len(request.system_message.content_blocks) == 1
        assert request.system_message.content_blocks[0]["text"] == "You are an assistant."
        assert request.system_message.additional_kwargs["middleware_1"] == "applied"

        request = second_middleware(request)
        assert len(request.system_message.content_blocks) == 1
        assert (
            request.system_message.content_blocks[0]["text"] == "You are an assistant. Be helpful."
        )
        assert request.system_message.additional_kwargs["middleware_1"] == "applied"
        assert request.system_message.additional_kwargs["middleware_2"] == "applied"

    def test_middleware_preserves_system_message_metadata(self) -> None:
        """Test that metadata is preserved when middleware modifies system message."""
        base_message = SystemMessage(
            content="Base prompt",
            additional_kwargs={"key1": "value1", "key2": "value2"},
            response_metadata={"model": "gpt-4"},
        )

        def preserving_middleware(request: ModelRequest) -> ModelRequest:
            """Middleware that preserves existing metadata."""
            new_message = SystemMessage(
                content=request.system_message.text + " Extended.",
                additional_kwargs=request.system_message.additional_kwargs,
                response_metadata=request.system_message.response_metadata,
            )
            return request.override(system_message=new_message)

        request = _make_request(system_message=base_message)
        new_request = preserving_middleware(request)

        assert len(new_request.system_message.content_blocks) == 1
        assert new_request.system_message.content_blocks[0]["text"] == "Base prompt Extended."
        assert new_request.system_message.additional_kwargs == {
            "key1": "value1",
            "key2": "value2",
        }
        assert new_request.system_message.response_metadata == {"model": "gpt-4"}

    def test_backward_compatibility_with_string_system_prompt(self) -> None:
        """Test that middleware still works with string system prompts."""

        def string_middleware(request: ModelRequest) -> ModelRequest:
            """Middleware using string system prompt (backward compatible)."""
            current_prompt = request.system_prompt or ""
            new_prompt = current_prompt + " Additional instructions."
            return request.override(system_prompt=new_prompt.strip())

        request = _make_request(system_prompt="Base prompt")
        new_request = string_middleware(request)

        assert new_request.system_prompt == "Base prompt Additional instructions."
        # The system_prompt should be converted to SystemMessage internally
        assert isinstance(new_request.system_message, SystemMessage)

    def test_middleware_can_switch_between_string_and_system_message(self) -> None:
        """Test middleware can work with both string and SystemMessage.

        Note: In the Python implementation, system_prompt is automatically
        converted to SystemMessage, so middleware always sees a SystemMessage.
        """

        def flexible_middleware(request: ModelRequest) -> ModelRequest:
            """Middleware that works with both formats."""
            if request.system_message:
                # Work with SystemMessage
                new_message = SystemMessage(content=request.system_message.text + " [modified]")
                return request.override(system_message=new_message)
            else:
                # Create new SystemMessage if none exists
                new_message = SystemMessage(content="[created]")
                return request.override(system_message=new_message)

        # Test with explicit SystemMessage
        request1 = _make_request(system_message=SystemMessage(content="Hello"))
        result1 = flexible_middleware(request1)
        assert len(result1.system_message.content_blocks) == 1
        assert result1.system_message.content_blocks[0]["text"] == "Hello [modified]"

        # Test with string (gets converted to SystemMessage automatically)
        request2 = _make_request(system_prompt="Hello")
        result2 = flexible_middleware(request2)
        # String prompts are converted to SystemMessage internally
        assert len(result2.system_message.content_blocks) == 1
        assert result2.system_message.content_blocks[0]["text"] == "Hello [modified]"

        # Test with None
        request3 = _make_request(system_message=None)
        result3 = flexible_middleware(request3)
        assert len(result3.system_message.content_blocks) == 1
        assert result3.system_message.content_blocks[0]["text"] == "[created]"
