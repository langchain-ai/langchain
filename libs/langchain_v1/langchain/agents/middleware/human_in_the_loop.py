"""Human in the loop middleware."""

from typing import Any

from langgraph.prebuilt.interrupt import (
    ActionRequest,
    HumanInterrupt,
    HumanInterruptConfig,
    HumanResponse,
)
from langgraph.types import interrupt

from langchain.agents.middleware._utils import _generate_correction_tool_messages
from langchain.agents.middleware.types import AgentMiddleware, AgentState

ToolInterruptConfig = dict[str, HumanInterruptConfig]


class HumanInTheLoopMiddleware(AgentMiddleware):
    """Human in the loop middleware."""

    def __init__(
        self,
        tool_configs: ToolInterruptConfig,
        message_prefix: str = "Tool execution requires approval",
    ) -> None:
        """Initialize the human in the loop middleware.

        Args:
            tool_configs: The tool interrupt configs to use for the middleware.
            message_prefix: The message prefix to use when constructing interrupt content.
        """
        super().__init__()
        self.tool_configs = tool_configs
        self.message_prefix = message_prefix

    def after_model(self, state: AgentState) -> dict[str, Any] | None:
        """Trigger HITL flows for relevant tool calls after an AIMessage."""
        messages = state["messages"]
        if not messages:
            return None

        last_message = messages[-1]

        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return None

        # Separate tool calls that need interrupts from those that don't
        interrupt_tool_calls = []
        auto_approved_tool_calls = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            if tool_name in self.tool_configs:
                interrupt_tool_calls.append(tool_call)
            else:
                auto_approved_tool_calls.append(tool_call)

        # If no interrupts needed, return early
        if not interrupt_tool_calls:
            return None

        approved_tool_calls = auto_approved_tool_calls.copy()

        # Right now, we do not support multiple tool calls with interrupts
        if len(interrupt_tool_calls) > 1:
            tool_names = [t["name"] for t in interrupt_tool_calls]
            msg = f"Called the following tools which require interrupts: {tool_names}\n\nYou may only call ONE tool that requires an interrupt at a time"
            return {
                "messages": _generate_correction_tool_messages(msg, last_message.tool_calls),
                "jump_to": "model",
            }

        # Right now, we do not support interrupting a tool call if other tool calls exist
        if auto_approved_tool_calls:
            tool_names = [t["name"] for t in interrupt_tool_calls]
            msg = f"Called the following tools which require interrupts: {tool_names}. You also called other tools that do not require interrupts. If you call a tool that requires and interrupt, you may ONLY call that tool."
            return {
                "messages": _generate_correction_tool_messages(msg, last_message.tool_calls),
                "jump_to": "model",
            }

        # Only one tool call will need interrupts
        tool_call = interrupt_tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        description = f"{self.message_prefix}\n\nTool: {tool_name}\nArgs: {tool_args}"
        tool_config = self.tool_configs[tool_name]

        request: HumanInterrupt = {
            "action_request": ActionRequest(
                action=tool_name,
                args=tool_args,
            ),
            "config": tool_config,
            "description": description,
        }

        responses: list[HumanResponse] = interrupt([request])
        response = responses[0]

        if response["type"] == "accept":
            approved_tool_calls.append(tool_call)
        elif response["type"] == "edit":
            edited: ActionRequest = response["args"]  # type: ignore[assignment]
            new_tool_call = {
                "type": "tool_call",
                "name": tool_call["name"],
                "args": edited["args"],
                "id": tool_call["id"],
            }
            approved_tool_calls.append(new_tool_call)
        elif response["type"] == "ignore":
            return {"jump_to": "__end__"}
        elif response["type"] == "response":
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": response["args"],
            }
            return {"messages": [tool_message], "jump_to": "model"}
        else:
            msg = f"Unknown response type: {response['type']}"
            raise ValueError(msg)

        last_message.tool_calls = approved_tool_calls

        return {"messages": [last_message]}
