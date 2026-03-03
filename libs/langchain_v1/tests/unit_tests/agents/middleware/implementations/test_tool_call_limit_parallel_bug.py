"""Test that middleware correctly handles parallel tool calls with limits.

Verifies that when middleware blocks some parallel tool calls, only the
permitted calls execute and interrupts propagate correctly.
"""

from langchain_core.messages import HumanMessage, ToolCall
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from langchain.agents.factory import create_agent
from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from tests.unit_tests.agents.model import FakeToolCallingModel


def test_instrument_middleware_and_routing() -> None:
    """Trace the middleware return value and what model_to_tools sees."""
    call_log: list[str] = []

    @tool
    def ask_fruit_expert(question: str) -> str:
        """Ask the fruit expert."""
        call_log.append(f"fruit:{question}")
        interrupt("continue?")
        return f"Fruit answer: {question}"

    @tool
    def ask_veggie_expert(question: str) -> str:
        """Ask the veggie expert."""
        call_log.append(f"veggie:{question}")
        return f"Veggie answer: {question}"

    model = FakeToolCallingModel(
        tool_calls=[
            [
                ToolCall(name="ask_fruit_expert", args={"question": "apples"}, id="c1"),
                ToolCall(name="ask_fruit_expert", args={"question": "bananas"}, id="c2"),
            ],
            [],
        ]
    )

    fruit_mw = ToolCallLimitMiddleware(tool_name="ask_fruit_expert", run_limit=1)
    veggie_mw = ToolCallLimitMiddleware(tool_name="ask_veggie_expert", run_limit=1)

    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[ask_fruit_expert, ask_veggie_expert],
        middleware=[fruit_mw, veggie_mw],
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "debug1"}}
    result = agent.invoke(
        {"messages": [HumanMessage("Tell me about apples and bananas")]},
        config,
    )

    assert len(call_log) == 1, f"Expected 1 execution, got {len(call_log)}: {call_log}"
    assert len(result.interrupts) == 1, (
        f"Expected 1 interrupt, got {len(result.interrupts)}"
    )
