"""Test that reasoning content is stripped from agent message history."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from langchain.agents.factory import create_agent
from tests.unit_tests.agents.model import FakeToolCallingModel


class FakeReasoningModel(FakeToolCallingModel):
    """Fake model that includes reasoning content in responses."""

    reasoning_style: str = "content_blocks"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response with reasoning content."""
        result = super()._generate(messages, stop, run_manager, **kwargs)
        ai_msg = result.generations[0].message
        assert isinstance(ai_msg, AIMessage)

        if self.reasoning_style == "content_blocks":
            # Add reasoning content blocks to the message content
            text_content = ai_msg.content if isinstance(ai_msg.content, str) else ""
            new_content: list[dict[str, Any]] = [
                {"type": "reasoning", "reasoning": "Let me think step by step..."},
                {"type": "text", "text": text_content},
            ]
            ai_msg = ai_msg.model_copy(update={"content": new_content})
        elif self.reasoning_style == "thinking_blocks":
            text_content = ai_msg.content if isinstance(ai_msg.content, str) else ""
            new_content = [
                {"type": "thinking", "thinking": "Internal monologue..."},
                {"type": "text", "text": text_content},
            ]
            ai_msg = ai_msg.model_copy(update={"content": new_content})
        elif self.reasoning_style == "additional_kwargs":
            ai_msg = ai_msg.model_copy(
                update={
                    "additional_kwargs": {
                        **ai_msg.additional_kwargs,
                        "reasoning_content": "Step by step reasoning...",
                    }
                }
            )
        elif self.reasoning_style == "responses_api":
            ai_msg = ai_msg.model_copy(
                update={
                    "additional_kwargs": {
                        **ai_msg.additional_kwargs,
                        "reasoning": {
                            "id": "rs_123",
                            "type": "reasoning",
                            "summary": [{"text": "Thought process"}],
                        },
                    }
                }
            )

        return ChatResult(generations=[ChatGeneration(message=ai_msg)])


def test_reasoning_blocks_stripped_from_agent_output() -> None:
    """Test that reasoning content blocks are stripped from output messages."""
    model = FakeReasoningModel(
        tool_calls=[[]],
        reasoning_style="content_blocks",
    )
    agent = create_agent(model=model, tools=[])

    result = agent.invoke({"messages": [HumanMessage("hello")]})
    messages = result["messages"]

    for msg in messages:
        if isinstance(msg, AIMessage) and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict):
                    assert block.get("type") != "reasoning", "Reasoning blocks should be stripped"


def test_thinking_blocks_stripped_from_agent_output() -> None:
    """Test that Anthropic-style thinking blocks are stripped."""
    model = FakeReasoningModel(
        tool_calls=[[]],
        reasoning_style="thinking_blocks",
    )
    agent = create_agent(model=model, tools=[])

    result = agent.invoke({"messages": [HumanMessage("hello")]})
    messages = result["messages"]

    for msg in messages:
        if isinstance(msg, AIMessage) and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict):
                    assert block.get("type") != "thinking", "Thinking blocks should be stripped"


def test_additional_kwargs_reasoning_stripped() -> None:
    """Test that reasoning_content in additional_kwargs is stripped."""
    model = FakeReasoningModel(
        tool_calls=[[]],
        reasoning_style="additional_kwargs",
    )
    agent = create_agent(model=model, tools=[])

    result = agent.invoke({"messages": [HumanMessage("hello")]})
    messages = result["messages"]

    for msg in messages:
        if isinstance(msg, AIMessage):
            assert "reasoning_content" not in msg.additional_kwargs, (
                "reasoning_content should be stripped from additional_kwargs"
            )


def test_responses_api_reasoning_stripped() -> None:
    """Test that OpenAI Responses API reasoning dict is stripped."""
    model = FakeReasoningModel(
        tool_calls=[[]],
        reasoning_style="responses_api",
    )
    agent = create_agent(model=model, tools=[])

    result = agent.invoke({"messages": [HumanMessage("hello")]})
    messages = result["messages"]

    for msg in messages:
        if isinstance(msg, AIMessage):
            assert "reasoning" not in msg.additional_kwargs, (
                "reasoning dict should be stripped from additional_kwargs"
            )


def test_reasoning_stripped_with_tool_calls() -> None:
    """Test that reasoning is stripped but tool calls are preserved."""

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results: {query}"

    model = FakeReasoningModel(
        tool_calls=[
            [ToolCall(name="search", args={"query": "test"}, id="tc1")],
            [],
        ],
        reasoning_style="content_blocks",
    )
    agent = create_agent(model=model, tools=[search])

    result = agent.invoke({"messages": [HumanMessage("search for test")]})
    messages = result["messages"]

    # Verify tool was called (search results in messages)
    tool_msgs = [m for m in messages if hasattr(m, "tool_call_id")]
    assert len(tool_msgs) > 0, "Tool should have been called"

    # Verify no reasoning in any AI message
    for msg in messages:
        if isinstance(msg, AIMessage) and isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict):
                    assert block.get("type") not in {"reasoning", "thinking"}


def test_no_reasoning_passthrough() -> None:
    """Test that messages without reasoning pass through unchanged."""
    model = FakeToolCallingModel(tool_calls=[[]])
    agent = create_agent(model=model, tools=[])

    result = agent.invoke({"messages": [HumanMessage("hello")]})
    messages = result["messages"]

    # Should have at least the human message and an AI response
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    # Content should be a string (no list processing needed)
    assert isinstance(ai_messages[-1].content, str)
