from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, cast

from langchain_core.messages import AIMessage
from typing_extensions import Annotated

from langgraph.agents.types import AgentJump, AgentMiddleware, AgentState, AgentUpdate

class State(AgentState):
    tool_call_count: dict[str, int] = field(default_factory=dict)


class ToolCallLimitMiddleware(AgentMiddleware):
    """Terminates after a specific tool is called N times"""

    state_schema = State

    def __init__(self, tool_limits: dict[str, int]):
        self.tool_limits = tool_limits

    def after_model(self, state: State) -> AgentUpdate | AgentJump | None:
        ai_msg: AIMessage = cast(AIMessage, state["messages"][-1])

        tool_calls = {}
        for call in ai_msg.tool_calls or []:
            tool_calls[call["name"]] = tool_calls.get(call["name"], 0) + 1

        aggregate_calls = state["tool_call_count"].copy()
        for tool_name in tool_calls.keys():
            aggregate_calls[tool_name] = aggregate_calls.get(tool_name, 0) + 1

        for tool_name, max_calls in self.tool_limits.items():
            count = aggregate_calls.get(tool_name, 0)
            if count == max_calls:
                return {"tool_call_count": aggregate_calls, "jump_to": "__end__"}

        return {"tool_call_count": aggregate_calls}
