"""Comprehensive unit tests for system message handling in agents.

This module consolidates all system message and dynamic prompt tests:
- Basic system message scenarios (none, string, SystemMessage)
- ModelRequest system_message field support
- System message updates via middleware
- Multiple middleware chaining
- Cache control preservation
- Metadata merging
- Dynamic system prompt middleware
- Edge cases and error handling

These tests replicate functionality from langchainjs PR #9459.
"""

from typing import cast

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import AgentState, ModelRequest, ModelResponse


def _fake_runtime(context: dict | None = None) -> Runtime:
    """Create a fake runtime with optional context."""
    if context:
        # Create a simple object with context
        class FakeRuntime:
            def __init__(self):
                self.context = type("Context", (), context)()

        return cast("Runtime", FakeRuntime())
    return cast("Runtime", object())


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


# =============================================================================
# ModelRequest Tests
# =============================================================================


class TestModelRequestSystemMessage:
    """Test ModelRequest with system_message field."""

    @pytest.mark.parametrize(
        "system_message,system_prompt,expected_msg,expected_prompt",
        [
            # Test with SystemMessage
            (
                SystemMessage(content="You are helpful"),
                None,
                SystemMessage(content="You are helpful"),
                "You are helpful",
            ),
            # Test with None
            (None, None, None, None),
            # Test with string (backward compat)
            (None, "You are helpful", SystemMessage(content="You are helpful"), "You are helpful"),
        ],
    )
    def test_create_with_various_system_inputs(
        self, system_message, system_prompt, expected_msg, expected_prompt
    ) -> None:
        """Test creating ModelRequest with various system message inputs."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))

        request = ModelRequest(
            model=model,
            system_message=system_message,
            system_prompt=system_prompt,
            messages=[HumanMessage("Hi")],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={},
            runtime=None,
        )

        if expected_msg is None:
            assert request.system_message is None
        else:
            assert request.system_message.content == expected_msg.content
        assert request.system_prompt == expected_prompt

    def test_system_prompt_property_with_list_content(self) -> None:
        """Test system_prompt property handles list content."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        system_msg = SystemMessage(content=["Part 1", "Part 2"])

        request = ModelRequest(
            model=model,
            system_message=system_msg,
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={},
            runtime=None,
        )

        assert request.system_prompt is not None
        assert "Part 1" in request.system_prompt

    @pytest.mark.parametrize(
        "override_with,expected_text",
        [
            ("system_message", "New"),
            ("system_prompt", "New prompt"),
        ],
    )
    def test_override_methods(self, override_with, expected_text) -> None:
        """Test override() with system_message and system_prompt parameters."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        original_msg = SystemMessage(content="Original")

        original_request = ModelRequest(
            model=model,
            system_message=original_msg,
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={},
            runtime=None,
        )

        if override_with == "system_message":
            new_request = original_request.override(system_message=SystemMessage(content="New"))
        else:  # system_prompt
            new_request = original_request.override(system_prompt="New prompt")

        assert isinstance(new_request.system_message, SystemMessage)
        assert new_request.system_prompt == expected_text
        assert original_request.system_prompt == "Original"

    def test_override_system_prompt_to_none(self) -> None:
        """Test override() setting system_prompt to None."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))

        original_request = ModelRequest(
            model=model,
            system_message=SystemMessage(content="Original"),
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={},
            runtime=None,
        )

        new_request = original_request.override(system_prompt=None)

        assert new_request.system_message is None
        assert new_request.system_prompt is None

    @pytest.mark.parametrize(
        "use_constructor",
        [True, False],
        ids=["constructor", "override"],
    )
    def test_cannot_set_both_system_prompt_and_system_message(self, use_constructor) -> None:
        """Test that setting both system_prompt and system_message raises error."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))

        with pytest.raises(ValueError, match="Cannot specify both"):
            if use_constructor:
                ModelRequest(
                    model=model,
                    system_prompt="String prompt",
                    system_message=SystemMessage(content="Message prompt"),
                    messages=[],
                    tool_choice=None,
                    tools=[],
                    response_format=None,
                    state={},
                    runtime=None,
                )
            else:
                request = ModelRequest(
                    model=model,
                    system_message=None,
                    messages=[],
                    tool_choice=None,
                    tools=[],
                    response_format=None,
                    state={},
                    runtime=None,
                )
                request.override(
                    system_prompt="String prompt",
                    system_message=SystemMessage(content="Message prompt"),
                )

    @pytest.mark.parametrize(
        "new_value,should_be_none",
        [
            ("New prompt", False),
            (None, True),
        ],
    )
    def test_setattr_system_prompt_deprecated(self, new_value, should_be_none) -> None:
        """Test that setting system_prompt via setattr raises deprecation warning."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))

        request = ModelRequest(
            model=model,
            system_message=SystemMessage(content="Original") if not should_be_none else None,
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={},
            runtime=None,
        )

        with pytest.warns(DeprecationWarning, match="system_prompt is deprecated"):
            request.system_prompt = new_value

        if should_be_none:
            assert request.system_message is None
            assert request.system_prompt is None
        else:
            assert isinstance(request.system_message, SystemMessage)
            assert request.system_message.content_blocks[0]["text"] == new_value

    def test_system_message_with_complex_content(self) -> None:
        """Test SystemMessage with complex content (list of dicts)."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        system_msg = SystemMessage(
            content=[
                {"type": "text", "text": "You are helpful"},
                {"type": "text", "text": "Be concise", "cache_control": {"type": "ephemeral"}},
            ]
        )

        request = ModelRequest(
            model=model,
            system_message=system_msg,
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={},
            runtime=None,
        )

        assert isinstance(request.system_message.content_blocks, list)
        assert len(request.system_message.content_blocks) == 2
        assert request.system_message.content_blocks[1].get("cache_control") == {
            "type": "ephemeral"
        }

    def test_multiple_overrides_with_system_message(self) -> None:
        """Test chaining overrides with system_message."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))

        original_request = ModelRequest(
            model=model,
            system_message=SystemMessage(content="Prompt 1"),
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={},
            runtime=None,
        )

        final_request = (
            original_request.override(system_message=SystemMessage(content="Prompt 2"))
            .override(tool_choice="auto")
            .override(system_message=SystemMessage(content="Prompt 3"))
        )

        assert final_request.system_prompt == "Prompt 3"
        assert final_request.tool_choice == "auto"
        assert original_request.system_prompt == "Prompt 1"


# =============================================================================
# create_agent Tests
# =============================================================================


class TestCreateAgentSystemMessage:
    """Test create_agent with various system message inputs."""

    @pytest.mark.parametrize(
        "system_prompt",
        [
            None,
            "You are a helpful assistant",
            SystemMessage(content="You are a helpful assistant"),
            SystemMessage(
                content="You are a helpful assistant",
                additional_kwargs={"role": "system_admin", "priority": "high"},
                response_metadata={"model": "gpt-4", "temperature": 0.7},
            ),
            SystemMessage(
                content=[
                    {"type": "text", "text": "You are a helpful assistant"},
                    {
                        "type": "text",
                        "text": "Follow these rules carefully",
                        "cache_control": {"type": "ephemeral"},
                    },
                ]
            ),
        ],
        ids=[
            "none",
            "string",
            "system_message",
            "system_message_with_metadata",
            "system_message_with_complex_content",
        ],
    )
    def test_create_agent_with_various_system_prompts(self, system_prompt) -> None:
        """Test create_agent accepts various system_prompt formats."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))

        agent = create_agent(
            model=model,
            system_prompt=system_prompt,
        )

        assert agent is not None


# =============================================================================
# Middleware Tests
# =============================================================================


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
        assert captured_request.system_message.text == "Base prompt Additional instructions."
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
            assert req.system_message.text == "String prompt + SystemMessage"
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
            new_content = [*existing_content, {"type": "text", "text": "Additional text"}]

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

    @pytest.mark.parametrize(
        "metadata_type,initial_metadata,update_metadata,expected_result",
        [
            # additional_kwargs merging
            (
                "additional_kwargs",
                {"key1": "value1", "shared": "original"},
                {"key2": "value2", "shared": "updated"},
                {"key1": "value1", "key2": "value2", "shared": "updated"},
            ),
            # response_metadata merging
            (
                "response_metadata",
                {"model": "gpt-4", "region": "us-east"},
                {"tokens": 100, "region": "eu-west"},
                {"model": "gpt-4", "tokens": 100, "region": "eu-west"},
            ),
        ],
        ids=["additional_kwargs", "response_metadata"],
    )
    def test_metadata_merge_across_updates(
        self, metadata_type, initial_metadata, update_metadata, expected_result
    ) -> None:
        """Test that metadata merges correctly when updating system message."""
        base_message = SystemMessage(
            content="Base",
            **{metadata_type: initial_metadata},
        )

        def update_middleware(request: ModelRequest, handler) -> ModelResponse:
            """Update system message, merging metadata."""
            current_metadata = getattr(request.system_message, metadata_type)
            new_metadata = {**current_metadata, **update_metadata}

            new_request = request.override(
                system_message=SystemMessage(content="Updated", **{metadata_type: new_metadata})
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
        assert getattr(captured_request.system_message, metadata_type) == expected_result


# =============================================================================
# Dynamic System Prompt Middleware Tests
# =============================================================================


class TestDynamicSystemPromptMiddleware:
    """Test middleware that accepts SystemMessage return types."""

    def test_middleware_can_return_system_message(self) -> None:
        """Test that middleware can return a SystemMessage with dynamic content."""

        def dynamic_system_prompt_middleware(request: ModelRequest) -> SystemMessage:
            """Return a SystemMessage with dynamic content."""
            region = getattr(request.runtime.context, "region", "n/a")
            return SystemMessage(content=f"You are a helpful assistant. Region: {region}")

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

        new_system_message = dynamic_system_prompt_middleware(request)

        assert isinstance(new_system_message, SystemMessage)
        assert len(new_system_message.content_blocks) == 1
        assert (
            new_system_message.content_blocks[0]["text"]
            == "You are a helpful assistant. Region: EU"
        )

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
        assert isinstance(new_request.system_message, SystemMessage)

    @pytest.mark.parametrize(
        "initial_value",
        [
            SystemMessage(content="Hello"),
            "Hello",
            None,
        ],
        ids=["system_message", "string", "none"],
    )
    def test_middleware_can_switch_between_formats(self, initial_value) -> None:
        """Test middleware can work with SystemMessage, string, or None."""

        def flexible_middleware(request: ModelRequest) -> ModelRequest:
            """Middleware that works with various formats."""
            if request.system_message:
                new_message = SystemMessage(content=request.system_message.text + " [modified]")
                return request.override(system_message=new_message)
            new_message = SystemMessage(content="[created]")
            return request.override(system_message=new_message)

        if isinstance(initial_value, SystemMessage):
            request = _make_request(system_message=initial_value)
            expected_text = "Hello [modified]"
        elif isinstance(initial_value, str):
            request = _make_request(system_prompt=initial_value)
            expected_text = "Hello [modified]"
        else:  # None
            request = _make_request(system_message=None)
            expected_text = "[created]"

        result = flexible_middleware(request)
        assert len(result.system_message.content_blocks) == 1
        assert result.system_message.content_blocks[0]["text"] == expected_text


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling for system messages."""

    @pytest.mark.parametrize(
        "content,expected_blocks,expected_prompt",
        [
            ("", 0, ""),
            (
                [
                    {"type": "text", "text": "Block 1"},
                    {"type": "text", "text": "Block 2"},
                    {"type": "text", "text": "Block 3"},
                ],
                3,
                None,
            ),
        ],
        ids=["empty_content", "multiple_blocks"],
    )
    def test_system_message_content_variations(
        self, content, expected_blocks, expected_prompt
    ) -> None:
        """Test SystemMessage with various content variations."""
        system_message = SystemMessage(content=content)

        model = GenericFakeChatModel(messages=iter([AIMessage(content="response")]))
        request = ModelRequest(
            model=model,
            system_message=system_message,
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state=cast("AgentState", {"messages": []}),  # type: ignore[name-defined]
            runtime=_fake_runtime(),
        )

        if isinstance(content, list):
            assert isinstance(request.system_message.content_blocks, list)
            assert len(request.system_message.content_blocks) == expected_blocks
        else:
            assert len(request.system_message.content_blocks) == expected_blocks
            assert request.system_prompt == expected_prompt

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

        new_request = request.override(system_message=None)

        assert new_request.system_message is None
        assert new_request.system_prompt is None
