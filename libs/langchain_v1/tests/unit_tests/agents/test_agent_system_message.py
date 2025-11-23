"""Comprehensive unit tests for agent system message handling.

These tests replicate the functionality from langchainjs PR #9459:
- Basic system message scenarios (none, string, SystemMessage)
- System message updates via middleware
- Multiple middleware chaining
- Cache control preservation
- Metadata merging
- Edge cases and error handling
"""

from typing import cast

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import ModelRequest, ModelResponse

from .model import FakeToolCallingModel


def _fake_runtime(context: dict | None = None) -> Runtime:
    """Create a fake runtime with optional context."""
    if context:
        # Create a simple object with context
        class FakeRuntime:
            def __init__(self):
                self.context = type("Context", (), context)()

        return cast(Runtime, FakeRuntime())
    return cast(Runtime, object())


class TestBasicSystemMessageScenarios:
    """Test basic scenarios for system message handling in agents."""

    def test_agent_with_no_system_message(self) -> None:
        """Test creating an agent with no system message."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))

        agent = create_agent(
            model=model,
            system_prompt=None,
        )

        assert agent is not None

    def test_agent_with_string_system_prompt(self) -> None:
        """Test creating an agent with a string system prompt (backward compat)."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))

        agent = create_agent(
            model=model,
            system_prompt="You are a helpful assistant",
        )

        assert agent is not None

    def test_agent_with_system_message_object(self) -> None:
        """Test creating an agent with a SystemMessage object."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        system_msg = SystemMessage(content="You are a helpful assistant")

        agent = create_agent(
            model=model,
            system_prompt=system_msg,
        )

        assert agent is not None

    def test_agent_with_system_message_containing_metadata(self) -> None:
        """Test agent with SystemMessage that has additional metadata."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        system_msg = SystemMessage(
            content="You are a helpful assistant",
            additional_kwargs={"role": "system_admin", "priority": "high"},
            response_metadata={"model": "gpt-4", "temperature": 0.7},
        )

        agent = create_agent(
            model=model,
            system_prompt=system_msg,
        )

        assert agent is not None


class TestSystemMessageUpdateViaMiddleware:
    """Test updating system messages through middleware."""

    def test_middleware_can_set_initial_system_message(self) -> None:
        """Test middleware setting system message when none exists."""

        def set_system_message_middleware(
            request: ModelRequest,
            handler,
        ) -> ModelResponse:
            """Middleware that sets initial system message."""
            new_request = request.override(
                system_message=SystemMessage(content="Set by middleware")
            )
            return handler(new_request)

        # Create request with no system message
        model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
        request = ModelRequest(
            model=model,
            system_message=None,
            messages=[HumanMessage(content="Hello")],
            tool_choice=None,
            tools=[],
            response_format=None,
            state=cast("AgentState", {"messages": []}),  # type: ignore[name-defined]
            runtime=_fake_runtime(),
        )

        captured_request = None

        def mock_handler(req: ModelRequest) -> ModelResponse:
            nonlocal captured_request
            captured_request = req
            return ModelResponse(result=[AIMessage(content="response")])

        set_system_message_middleware(request, mock_handler)

        assert captured_request is not None
        assert captured_request.system_message is not None
        assert len(captured_request.system_message.content_blocks) == 1
        assert captured_request.system_message.content_blocks[0]["text"] == "Set by middleware"

    def test_middleware_can_update_via_system_message_object(self) -> None:
        """Test middleware updating system message using SystemMessage objects."""

        def append_with_metadata_middleware(
            request: ModelRequest,
            handler,
        ) -> ModelResponse:
            """Append using SystemMessage to preserve metadata."""
            base_content = request.system_message.text if request.system_message else ""
            base_kwargs = request.system_message.additional_kwargs if request.system_message else {}

            new_message = SystemMessage(
                content=base_content + " Additional instructions.",
                additional_kwargs={**base_kwargs, "middleware": "applied"},
            )
            new_request = request.override(system_message=new_message)
            return handler(new_request)

        model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
        request = ModelRequest(
            model=model,
            system_message=SystemMessage(
                content="Base prompt", additional_kwargs={"base": "value"}
            ),
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state=cast("AgentState", {"messages": []}),  # type: ignore[name-defined]
            runtime=_fake_runtime(),
        )

        captured_request = None

        def mock_handler(req: ModelRequest) -> ModelResponse:
            nonlocal captured_request
            captured_request = req
            return ModelResponse(result=[AIMessage(content="response")])

        append_with_metadata_middleware(request, mock_handler)

        assert captured_request is not None
        assert "Base prompt Additional instructions." == captured_request.system_message.text
        assert captured_request.system_message.additional_kwargs["base"] == "value"
        assert captured_request.system_message.additional_kwargs["middleware"] == "applied"


class TestMultipleMiddlewareChaining:
    """Test multiple middleware modifying system message in sequence."""

    def test_multiple_middleware_can_chain_modifications(self) -> None:
        """Test that multiple middleware can modify system message sequentially."""

        def first_middleware(request: ModelRequest, handler) -> ModelResponse:
            """First middleware sets base system message."""
            new_request = request.override(
                system_message=SystemMessage(
                    content="Base prompt",
                    additional_kwargs={"middleware_1": "applied"},
                )
            )
            return handler(new_request)

        def second_middleware(request: ModelRequest, handler) -> ModelResponse:
            """Second middleware appends to system message."""
            current_content = request.system_message.text
            current_kwargs = request.system_message.additional_kwargs

            new_request = request.override(
                system_message=SystemMessage(
                    content=current_content + " + middleware 2",
                    additional_kwargs={**current_kwargs, "middleware_2": "applied"},
                )
            )
            return handler(new_request)

        def third_middleware(request: ModelRequest, handler) -> ModelResponse:
            """Third middleware appends to system message."""
            current_content = request.system_message.text
            current_kwargs = request.system_message.additional_kwargs

            new_request = request.override(
                system_message=SystemMessage(
                    content=current_content + " + middleware 3",
                    additional_kwargs={**current_kwargs, "middleware_3": "applied"},
                )
            )
            return handler(new_request)

        model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
        request = ModelRequest(
            model=model,
            system_message=None,
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state=cast("AgentState", {"messages": []}),  # type: ignore[name-defined]
            runtime=_fake_runtime(),
        )

        def final_handler(req: ModelRequest) -> ModelResponse:
            # Verify all middleware applied
            assert req.system_message.text == "Base prompt + middleware 2 + middleware 3"
            assert req.system_message.additional_kwargs["middleware_1"] == "applied"
            assert req.system_message.additional_kwargs["middleware_2"] == "applied"
            assert req.system_message.additional_kwargs["middleware_3"] == "applied"
            return ModelResponse(result=[AIMessage(content="response")])

        # Chain middleware calls
        first_middleware(
            request,
            lambda req: second_middleware(req, lambda req2: third_middleware(req2, final_handler)),
        )

    def test_middleware_can_mix_string_and_system_message_updates(self) -> None:
        """Test mixing string and SystemMessage updates across middleware."""

        def string_middleware(request: ModelRequest, handler) -> ModelResponse:
            """Use string-based update."""
            new_request = request.override(system_prompt="String prompt")
            return handler(new_request)

        def system_message_middleware(request: ModelRequest, handler) -> ModelResponse:
            """Use SystemMessage-based update."""
            current_content = request.system_message.text if request.system_message else ""
            new_request = request.override(
                system_message=SystemMessage(
                    content=current_content + " + SystemMessage",
                    additional_kwargs={"metadata": "added"},
                )
            )
            return handler(new_request)

        model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
        request = ModelRequest(
            model=model,
            system_message=None,
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state=cast("AgentState", {"messages": []}),  # type: ignore[name-defined]
            runtime=_fake_runtime(),
        )

        def final_handler(req: ModelRequest) -> ModelResponse:
            assert "String prompt + SystemMessage" == req.system_message.text
            assert req.system_message.additional_kwargs["metadata"] == "added"
            return ModelResponse(result=[AIMessage(content="response")])

        string_middleware(request, lambda req: system_message_middleware(req, final_handler))


class TestCacheControlPreservation:
    """Test cache control metadata preservation in system messages."""

    def test_middleware_can_add_cache_control(self) -> None:
        """Test middleware adding cache control to system message."""

        def cache_control_middleware(request: ModelRequest, handler) -> ModelResponse:
            """Add cache control to system message."""
            new_message = SystemMessage(
                content=[
                    {"type": "text", "text": "Base instructions"},
                    {
                        "type": "text",
                        "text": "Cached instructions",
                        "cache_control": {"type": "ephemeral"},
                    },
                ]
            )
            new_request = request.override(system_message=new_message)
            return handler(new_request)

        model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
        request = ModelRequest(
            model=model,
            system_message=None,
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state=cast("AgentState", {"messages": []}),  # type: ignore[name-defined]
            runtime=_fake_runtime(),
        )

        captured_request = None

        def mock_handler(req: ModelRequest) -> ModelResponse:
            nonlocal captured_request
            captured_request = req
            return ModelResponse(result=[AIMessage(content="response")])

        cache_control_middleware(request, mock_handler)

        assert captured_request is not None
        assert isinstance(captured_request.system_message.content_blocks, list)
        assert captured_request.system_message.content_blocks[1]["cache_control"] == {
            "type": "ephemeral"
        }

    def test_cache_control_preserved_across_middleware(self) -> None:
        """Test that cache control is preserved when middleware modifies message."""

        def first_middleware_with_cache(request: ModelRequest, handler) -> ModelResponse:
            """Set system message with cache control."""
            new_message = SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Cached content",
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            )
            new_request = request.override(system_message=new_message)
            return handler(new_request)

        def second_middleware_appends(request: ModelRequest, handler) -> ModelResponse:
            """Append to system message while preserving cache control."""
            existing_content = request.system_message.content_blocks
            new_content = existing_content + [{"type": "text", "text": "Additional text"}]

            new_message = SystemMessage(content=new_content)
            new_request = request.override(system_message=new_message)
            return handler(new_request)

        model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
        request = ModelRequest(
            model=model,
            system_message=None,
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state=cast("AgentState", {"messages": []}),  # type: ignore[name-defined]
            runtime=_fake_runtime(),
        )

        def final_handler(req: ModelRequest) -> ModelResponse:
            # Verify cache control was preserved
            assert isinstance(req.system_message.content_blocks, list)
            assert len(req.system_message.content_blocks) == 2
            assert req.system_message.content_blocks[0]["cache_control"] == {"type": "ephemeral"}
            return ModelResponse(result=[AIMessage(content="response")])

        first_middleware_with_cache(
            request, lambda req: second_middleware_appends(req, final_handler)
        )


class TestMetadataMerging:
    """Test metadata merging behavior when updating system messages."""

    def test_additional_kwargs_merge_across_updates(self) -> None:
        """Test that additional_kwargs merge when updating system message."""
        base_message = SystemMessage(
            content="Base", additional_kwargs={"key1": "value1", "shared": "original"}
        )

        def update_middleware(request: ModelRequest, handler) -> ModelResponse:
            """Update system message, merging metadata."""
            current_kwargs = request.system_message.additional_kwargs
            new_kwargs = {**current_kwargs, "key2": "value2", "shared": "updated"}

            new_request = request.override(
                system_message=SystemMessage(content="Updated", additional_kwargs=new_kwargs)
            )
            return handler(new_request)

        model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
        request = ModelRequest(
            model=model,
            system_message=base_message,
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state=cast("AgentState", {"messages": []}),  # type: ignore[name-defined]
            runtime=_fake_runtime(),
        )

        captured_request = None

        def mock_handler(req: ModelRequest) -> ModelResponse:
            nonlocal captured_request
            captured_request = req
            return ModelResponse(result=[AIMessage(content="response")])

        update_middleware(request, mock_handler)

        assert captured_request is not None
        assert captured_request.system_message.additional_kwargs == {
            "key1": "value1",
            "key2": "value2",
            "shared": "updated",  # Later value wins
        }

    def test_response_metadata_merge_across_updates(self) -> None:
        """Test that response_metadata merges when updating system message."""
        base_message = SystemMessage(
            content="Base",
            response_metadata={"model": "gpt-4", "region": "us-east"},
        )

        def update_middleware(request: ModelRequest, handler) -> ModelResponse:
            """Update system message, merging response metadata."""
            current_metadata = request.system_message.response_metadata
            new_metadata = {**current_metadata, "tokens": 100, "region": "eu-west"}

            new_request = request.override(
                system_message=SystemMessage(content="Updated", response_metadata=new_metadata)
            )
            return handler(new_request)

        model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
        request = ModelRequest(
            model=model,
            system_message=base_message,
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state=cast("AgentState", {"messages": []}),  # type: ignore[name-defined]
            runtime=_fake_runtime(),
        )

        captured_request = None

        def mock_handler(req: ModelRequest) -> ModelResponse:
            nonlocal captured_request
            captured_request = req
            return ModelResponse(result=[AIMessage(content="response")])

        update_middleware(request, mock_handler)

        assert captured_request is not None
        assert captured_request.system_message.response_metadata == {
            "model": "gpt-4",
            "tokens": 100,
            "region": "eu-west",  # Later value wins
        }


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling for system messages."""

    def test_reset_system_prompt_to_none(self) -> None:
        """Test resetting system prompt to None."""
        base_message = SystemMessage(content="Original prompt")

        model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
        request = ModelRequest(
            model=model,
            system_message=base_message,
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state=cast("AgentState", {"messages": []}),  # type: ignore[name-defined]
            runtime=_fake_runtime(),
        )

        # Reset to None
        new_request = request.override(system_message=None)

        assert new_request.system_message is None
        assert new_request.system_prompt is None

    def test_empty_system_message_content(self) -> None:
        """Test handling empty SystemMessage content."""
        empty_message = SystemMessage(content="")

        model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
        request = ModelRequest(
            model=model,
            system_message=empty_message,
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state=cast("AgentState", {"messages": []}),  # type: ignore[name-defined]
            runtime=_fake_runtime(),
        )

        assert len(request.system_message.content_blocks) == 0
        assert request.system_prompt == ""

    def test_system_message_with_multiple_content_blocks(self) -> None:
        """Test SystemMessage with multiple content blocks."""
        multi_block_message = SystemMessage(
            content=[
                {"type": "text", "text": "Block 1"},
                {"type": "text", "text": "Block 2"},
                {"type": "text", "text": "Block 3"},
            ]
        )

        model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
        request = ModelRequest(
            model=model,
            system_message=multi_block_message,
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state=cast("AgentState", {"messages": []}),  # type: ignore[name-defined]
            runtime=_fake_runtime(),
        )

        assert isinstance(request.system_message.content_blocks, list)
        assert len(request.system_message.content_blocks) == 3

    def test_cannot_set_both_system_prompt_and_system_message(self) -> None:
        """Test that setting both system_prompt and system_message raises error."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))

        with pytest.raises(ValueError, match="Cannot specify both"):
            ModelRequest(
                model=model,
                system_message=SystemMessage(content="Message"),
                system_prompt="Prompt",
                messages=[],
                tool_choice=None,
                tools=[],
                response_format=None,
                state=cast("AgentState", {"messages": []}),  # type: ignore[name-defined]
                runtime=_fake_runtime(),
            )

    def test_override_cannot_set_both_system_prompt_and_system_message(self) -> None:
        """Test that override with both system_prompt and system_message raises error."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
        request = ModelRequest(
            model=model,
            system_message=None,
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state=cast("AgentState", {"messages": []}),  # type: ignore[name-defined]
            runtime=_fake_runtime(),
        )

        with pytest.raises(ValueError, match="Cannot specify both"):
            request.override(
                system_message=SystemMessage(content="Message"),
                system_prompt="Prompt",
            )
