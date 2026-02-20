"""Tests for reasoning content stripping in create_agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForLLMRun

from langchain.agents import create_agent
from tests.unit_tests.agents.model import FakeToolCallingModel


class FakeReasoningModel(FakeToolCallingModel):
    """Fake model that includes reasoning content in responses."""

    reasoning_blocks: list[dict[str, Any]] | None = None
    reasoning_additional_kwargs: dict[str, Any] | None = None

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        content: str | list[dict[str, Any]]
        if self.reasoning_blocks:
            content = [
                *self.reasoning_blocks,
                {"type": "text", "text": "The answer is 42."},
            ]
        else:
            content = "The answer is 42."

        message = AIMessage(
            content=content,
            id=str(self.index),
            additional_kwargs=dict(self.reasoning_additional_kwargs or {}),
        )
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_reasoning_blocks_stripped_from_agent_output() -> None:
    """Reasoning content blocks should not appear in agent output messages."""
    model = FakeReasoningModel(
        reasoning_blocks=[
            {"type": "reasoning", "reasoning": "Let me think step by step..."},
        ],
    )
    agent = create_agent(model, [])
    result = agent.invoke({"messages": [HumanMessage("What is the answer?")]})

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 1

    ai_msg = ai_messages[0]
    # Reasoning blocks should be stripped, only text remains
    assert ai_msg.content == [{"type": "text", "text": "The answer is 42."}]


def test_thinking_blocks_stripped_from_agent_output() -> None:
    """Thinking blocks (e.g. Anthropic) should not appear in agent output messages."""
    model = FakeReasoningModel(
        reasoning_blocks=[
            {"type": "thinking", "thinking": "I need to consider..."},
        ],
    )
    agent = create_agent(model, [])
    result = agent.invoke({"messages": [HumanMessage("What is the answer?")]})

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    ai_msg = ai_messages[0]
    assert ai_msg.content == [{"type": "text", "text": "The answer is 42."}]


def test_reasoning_additional_kwargs_stripped_from_agent_output() -> None:
    """reasoning_content in additional_kwargs should be stripped."""
    model = FakeReasoningModel(
        reasoning_additional_kwargs={
            "reasoning_content": "Deep reasoning here...",
        },
    )
    agent = create_agent(model, [])
    result = agent.invoke({"messages": [HumanMessage("What is the answer?")]})

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    ai_msg = ai_messages[0]
    assert ai_msg.content == "The answer is 42."
    assert "reasoning_content" not in ai_msg.additional_kwargs


def test_reasoning_key_stripped_from_additional_kwargs() -> None:
    """Reasoning key (OpenAI Responses API) in additional_kwargs should be stripped."""
    model = FakeReasoningModel(
        reasoning_additional_kwargs={
            "reasoning": {"id": "rs_abc", "summary": [{"text": "thinking..."}]},
        },
    )
    agent = create_agent(model, [])
    result = agent.invoke({"messages": [HumanMessage("What is the answer?")]})

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    ai_msg = ai_messages[0]
    assert "reasoning" not in ai_msg.additional_kwargs


def test_no_reasoning_messages_unchanged() -> None:
    """Messages without reasoning content should pass through unchanged."""
    model = FakeReasoningModel()
    agent = create_agent(model, [])
    result = agent.invoke({"messages": [HumanMessage("What is the answer?")]})

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    ai_msg = ai_messages[0]
    assert ai_msg.content == "The answer is 42."
