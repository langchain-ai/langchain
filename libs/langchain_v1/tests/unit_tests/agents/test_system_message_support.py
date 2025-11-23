"""Unit tests for SystemMessage support in create_agent and ModelRequest."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain.agents.middleware.types import ModelRequest


class TestModelRequestSystemMessage:
    """Test ModelRequest with system_message field."""

    def test_create_with_system_message(self) -> None:
        """Test creating ModelRequest with SystemMessage."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        system_msg = SystemMessage(content="You are a helpful assistant")

        request = ModelRequest(
            model=model,
            system_message=system_msg,
            messages=[HumanMessage("Hi")],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={},
            runtime=None,
        )

        assert request.system_message == system_msg
        assert request.system_prompt == "You are a helpful assistant"

    def test_create_with_none_system_message(self) -> None:
        """Test creating ModelRequest with None system_message."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))

        request = ModelRequest(
            model=model,
            system_message=None,
            messages=[HumanMessage("Hi")],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={},
            runtime=None,
        )

        assert request.system_message is None
        assert request.system_prompt is None

    def test_system_prompt_property_with_string_content(self) -> None:
        """Test system_prompt property returns content from SystemMessage."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        system_msg = SystemMessage(content="Test prompt")

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

        assert request.system_prompt == "Test prompt"

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

        # Should convert list content to string
        assert request.system_prompt is not None
        assert "Part 1" in request.system_prompt

    def test_override_with_system_message(self) -> None:
        """Test override() with system_message parameter."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        original_msg = SystemMessage(content="Original")
        new_msg = SystemMessage(content="New")

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

        new_request = original_request.override(system_message=new_msg)

        assert new_request.system_message == new_msg
        assert new_request.system_prompt == "New"
        assert original_request.system_message == original_msg
        assert original_request.system_prompt == "Original"

    def test_override_with_system_prompt_backward_compat(self) -> None:
        """Test override() with system_prompt parameter (backward compatibility)."""
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

        # Override using system_prompt (backward compat)
        new_request = original_request.override(system_prompt="New prompt")

        assert new_request.system_prompt == "New prompt"
        assert isinstance(new_request.system_message, SystemMessage)
        assert len(new_request.system_message.content_blocks) == 1
        assert new_request.system_message.content_blocks[0]["text"] == "New prompt"

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

    def test_create_with_system_prompt_backward_compat(self) -> None:
        """Test creating ModelRequest with system_prompt parameter (backward compat)."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))

        request = ModelRequest(
            model=model,
            system_prompt="You are a helpful assistant",
            messages=[HumanMessage("Hi")],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={},
            runtime=None,
        )

        # system_prompt should be converted to SystemMessage
        assert isinstance(request.system_message, SystemMessage)
        assert len(request.system_message.content_blocks) == 1
        assert request.system_message.content_blocks[0]["text"] == "You are a helpful assistant"
        assert request.system_prompt == "You are a helpful assistant"

    def test_cannot_set_both_system_prompt_and_system_message_in_constructor(self) -> None:
        """Test that constructor rejects both system_prompt and system_message."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))

        with pytest.raises(ValueError, match="Cannot specify both"):
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

    def test_cannot_set_both_system_prompt_and_system_message_in_override(self) -> None:
        """Test that override rejects both system_prompt and system_message."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))

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

        with pytest.raises(ValueError, match="Cannot specify both"):
            request.override(
                system_prompt="String prompt",
                system_message=SystemMessage(content="Message prompt"),
            )

    def test_setattr_system_prompt_deprecated(self) -> None:
        """Test that setting system_prompt via setattr raises deprecation warning."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))

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

        with pytest.warns(DeprecationWarning, match="system_prompt is deprecated"):
            request.system_prompt = "New prompt"

        # Should still work but convert to SystemMessage
        assert isinstance(request.system_message, SystemMessage)
        assert len(request.system_message.content_blocks) == 1
        assert request.system_message.content_blocks[0]["text"] == "New prompt"

    def test_setattr_system_prompt_to_none(self) -> None:
        """Test that setting system_prompt to None via setattr works."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))

        request = ModelRequest(
            model=model,
            system_message=SystemMessage(content="Original"),
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={},
            runtime=None,
        )

        with pytest.warns(DeprecationWarning, match="system_prompt is deprecated"):
            request.system_prompt = None

        # Should clear the system_message
        assert request.system_message is None
        assert request.system_prompt is None

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


class TestCreateAgentSystemMessage:
    """Test create_agent with SystemMessage support."""

    def test_create_agent_with_string_system_prompt(self) -> None:
        """Test create_agent accepts string system_prompt (backward compat)."""
        from langchain.agents import create_agent

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))

        # Should not raise - backward compatibility
        agent = create_agent(
            model=model,
            system_prompt="You are a helpful assistant",
        )

        assert agent is not None

    def test_create_agent_with_system_message(self) -> None:
        """Test create_agent accepts SystemMessage for system_prompt."""
        from langchain.agents import create_agent

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        system_msg = SystemMessage(content="You are a helpful assistant")

        # Should not raise
        agent = create_agent(
            model=model,
            system_prompt=system_msg,
        )

        assert agent is not None

    def test_create_agent_with_none_system_prompt(self) -> None:
        """Test create_agent with None system_prompt."""
        from langchain.agents import create_agent

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))

        # Should not raise
        agent = create_agent(
            model=model,
            system_prompt=None,
        )

        assert agent is not None

    def test_create_agent_system_message_with_metadata(self) -> None:
        """Test create_agent with SystemMessage containing metadata."""
        from langchain.agents import create_agent

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        system_msg = SystemMessage(
            content="You are a helpful assistant", additional_kwargs={"role": "system_admin"}
        )

        # Should not raise and preserve metadata
        agent = create_agent(
            model=model,
            system_prompt=system_msg,
        )

        assert agent is not None

    def test_create_agent_system_message_with_complex_content(self) -> None:
        """Test create_agent with SystemMessage containing complex content blocks."""
        from langchain.agents import create_agent

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        system_msg = SystemMessage(
            content=[
                {"type": "text", "text": "You are a helpful assistant"},
                {
                    "type": "text",
                    "text": "Follow these rules carefully",
                    "cache_control": {"type": "ephemeral"},
                },
            ]
        )

        # Should not raise and preserve complex content
        agent = create_agent(
            model=model,
            system_prompt=system_msg,
        )

        assert agent is not None

    def test_create_agent_type_annotations(self) -> None:
        """Test that create_agent properly handles the str | SystemMessage type."""
        from langchain.agents import create_agent

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))

        # Test with string
        agent1 = create_agent(model=model, system_prompt="String prompt")
        assert agent1 is not None

        # Test with SystemMessage
        agent2 = create_agent(model=model, system_prompt=SystemMessage(content="Message prompt"))
        assert agent2 is not None

        # Test with None
        agent3 = create_agent(model=model, system_prompt=None)
        assert agent3 is not None
