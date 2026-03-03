"""Debug: trace exactly what happens when middleware processes parallel calls.

Instruments after_model and model_to_tools to see if the ToolMessage injected
by middleware is visible to the routing edge.
"""

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import GraphOutput, interrupt

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

    # Wrap after_model to trace its return value
    original_after_model = ToolCallLimitMiddleware.after_model

    def traced_after_model(self, state, runtime):
        result = original_after_model(self, state, runtime)
        msgs = state.get("messages", [])
        ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        print(f"\n  [{self.name}].after_model:")
        print(f"    state has {len(msgs)} messages ({len(ai_msgs)} AI, {len(tool_msgs)} Tool)")
        if ai_msgs:
            last_ai = ai_msgs[-1]
            print(f"    last AI tool_calls: {[tc['id'] for tc in last_ai.tool_calls]}")
        print(f"    returning: {result}")
        if result and "messages" in result:
            for m in result["messages"]:
                if isinstance(m, ToolMessage):
                    print(f"      -> injecting ToolMessage(call_id={m.tool_call_id}, status={m.status})")
        return result

    checkpointer = MemorySaver()

    with patch.object(ToolCallLimitMiddleware, "after_model", traced_after_model):
        agent = create_agent(
            model=model,
            tools=[ask_fruit_expert, ask_veggie_expert],
            middleware=[fruit_mw, veggie_mw],
            checkpointer=checkpointer,
        )

        # Also trace the model_to_tools edge
        # Get the compiled graph and print its structure
        print("\n=== Graph nodes ===")
        for name in agent.nodes:
            print(f"  {name}")

        config = {"configurable": {"thread_id": "debug1"}}
        result = agent.invoke(
            {"messages": [HumanMessage("Tell me about apples and bananas")]},
            config,
        )

    # v2 stream format: unwrap GraphOutput
    assert isinstance(result, GraphOutput)
    result_dict = result.value
    interrupts = list(result.interrupts)
    print(f"\n=== Results ===")
    print(f"call_log:   {call_log}")
    print(f"interrupts: {len(interrupts)}")

    tool_msgs = [m for m in result_dict["messages"] if isinstance(m, ToolMessage)]
    error_msgs = [m for m in tool_msgs if m.status == "error"]
    print(f"error tool messages: {len(error_msgs)}")
    for m in error_msgs:
        print(f"  call_id={m.tool_call_id} content={m.content!r}")

    assert len(call_log) == 1, f"Expected 1 execution, got {len(call_log)}: {call_log}"
    assert len(interrupts) == 1, f"Expected 1 interrupt, got {len(interrupts)}"
