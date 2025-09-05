from langgraph.prebuilt.interrupt import (
    ActionRequest,
    HumanInterrupt,
    HumanInterruptConfig,
    HumanResponse,
)
from langgraph.types import interrupt

from langchain.agents.types import AgentJump, AgentMiddleware, AgentState, AgentUpdate

ToolInterruptConfig = dict[str, HumanInterruptConfig]


class HumanInTheLoopMiddleware(AgentMiddleware):
    def __init__(
        self,
        tool_configs: ToolInterruptConfig,
        message_prefix: str = "Tool execution requires approval",
    ):
        super().__init__()
        self.tool_configs = tool_configs
        self.message_prefix = message_prefix

    def after_model(self, state: AgentState) -> AgentUpdate | AgentJump | None:
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

        # Process all tool calls that need interrupts in parallel
        requests = []

        for tool_call in interrupt_tool_calls:
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
            requests.append(request)

        responses: list[HumanResponse] = interrupt(requests)

        for i, response in enumerate(responses):
            tool_call = interrupt_tool_calls[i]

            if response["type"] == "accept":
                approved_tool_calls.append(tool_call)
            elif response["type"] == "edit":
                edited: ActionRequest = response["args"]
                new_tool_call = {
                    "name": tool_call["name"],
                    "args": edited["args"],
                    "id": tool_call["id"],
                }
                approved_tool_calls.append(new_tool_call)
            elif response["type"] == "ignore":
                # NOTE: does not work with multiple interrupts
                return {"goto": "__end__"}
            elif response["type"] == "response":
                # NOTE: does not work with multiple interrupts
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": response["args"],
                }
                return {"messages": [tool_message], "goto": "model"}
            else:
                raise ValueError(f"Unknown response type: {response['type']}")

        last_message.tool_calls = approved_tool_calls

        return {"messages": [last_message]}
