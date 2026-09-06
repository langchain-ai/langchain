from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


executed: list[tuple[str, object]] = []


@tool
def tool_a(value: int) -> str:
    """Run tool A."""
    executed.append(("tool_a", value))
    return "A executed"


@tool
def tool_b(value: str) -> str:
    """Run tool B."""
    executed.append(("tool_b", value))
    return "B executed"


class DeterministicModel(BaseChatModel):
    """Emit one call to tool A, followed by a final response."""

    response_index: int = 0

    @property
    def _llm_type(self) -> str:
        return "deterministic-test-model"

    def bind_tools(
        self,
        tools: Any,
        **kwargs: Any,
    ) -> "DeterministicModel":
        return self

    def _generate(
        self,
        messages: Any,
        **kwargs: Any,
    ) -> ChatResult:
        if self.response_index == 0:
            message = AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="tool_a",
                        args={"value": 1},
                        id="call-1",
                    )
                ],
            )
        else:
            message = AIMessage(content="done")

        self.response_index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture
def agent() -> Any:
    return create_agent(
        model=DeterministicModel(),
        tools=[tool_a, tool_b],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "tool_a": {
                        "allowed_decisions": ["approve", "edit"],
                    },
                    "tool_b": {
                        "allowed_decisions": ["approve", "reject"],
                    },
                }
            )
        ],
        checkpointer=InMemorySaver(),
    )


def test_cross_tool_edit_does_not_bypass_target_tool_policy(agent: Any) -> None:
    config = {
        "configurable": {
            "thread_id": "hitl-cross-tool-edit",
        }
    }

    first = agent.invoke(
        {"messages": [HumanMessage(content="Run tool A")]},
        config,
    )

    assert "__interrupt__" in first

    final = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": "tool_b",
                            "args": {"value": "edited"},
                        },
                    }
                ]
            }
        ),
        config,
    )

    # New Assertion
    assert "__interrupt__" in final  # Should be True now!
    assert executed == []  # tool_b should NOT have executed yet!
