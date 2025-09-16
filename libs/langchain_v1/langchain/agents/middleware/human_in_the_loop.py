"""Human in the loop middleware."""

from typing import Any, Literal, cast

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.types import interrupt
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware.types import AgentMiddleware, AgentState


class HumanInterruptConfig(TypedDict):
    """Configuration that defines what actions are allowed for a human interrupt.

    This controls the available interaction options when the graph is paused for human input.

    Attributes:
        allow_approve: Whether the human can approve a given tool call
        allow_ignore: Whether the human can choose to ignore a given tool call.
            Results in an artificial tool message added to the `messages` list.
        allow_response: Whether the human can provide a response to a given tool call.
            Results in an artificial tool message added to the `messages` list.
        allow_edit: Whether the human can edit a tool call (name and args)
    """

    allow_approve: bool
    allow_ignore: bool
    allow_response: bool
    allow_edit: bool


class HumanInterrupt(TypedDict):
    """Represents an interrupt triggered by the graph that requires human intervention.

    For each tool call requiring human input, a `HumanInterrupt` is created and
    is accessible in the agent state when execution is paused for human input.

    Attributes:
        action: The action being requested (a tool name)
        args: Arguments for the action (tool kwargs)
        config: Configuration defining what actions are allowed
        description: Optional detailed description of what input is needed
        tool_call_id: Identifier for the associated tool call

    Example:
        # Send the interrupt request and get the response
        request = HumanInterrupt(
            action="run_command",  # The action being requested
            args={"command": "ls", "args": ["-l"]},  # Arguments for the action
            config=HumanInterruptConfig(
                allow_ignore=True,  # Allow skipping this step
                allow_response=True,  # Allow text feedback
                allow_edit=False,  # Don't allow editing
                allow_approve=True,  # Allow direct acceptance
            ),
            description="Please review the command before execution",
            tool_call_id="call_123",
        )
        # Send the interrupt request and get the response
        response = interrupt([request])[0]
        ```
    """

    action: str
    args: dict
    config: HumanInterruptConfig
    description: str | None
    tool_call_id: str


class ApprovePayload(TypedDict):
    """Human chose to approve the current state without changes."""

    type: Literal["approve"]
    tool_call_id: str


class IgnorePayload(TypedDict):
    """Human chose to ignore/skip the current step with optional tool message customization."""

    type: Literal["ignore"]
    tool_call_id: str
    tool_message: NotRequired[str | ToolMessage]


class ResponsePayload(TypedDict):
    """Human provided text feedback or instructions."""

    type: Literal["response"]
    tool_call_id: str
    tool_message: str | ToolMessage


class EditPayload(TypedDict):
    """Human chose to edit/modify the current state/content."""

    type: Literal["edit"]
    tool_call_id: str
    action: str
    args: dict


HumanResponse = ApprovePayload | IgnorePayload | ResponsePayload | EditPayload


class HumanInTheLoopMiddleware(AgentMiddleware):
    """Human in the loop middleware."""

    def __init__(
        self,
        tool_configs: dict[str, bool | HumanInterruptConfig],
        *,
        message_prefix: str = "Tool execution requires approval",
    ) -> None:
        """Initialize the human in the loop middleware.

        Args:
            tool_configs: Mapping of tool name to interrupt config (bool | HumanInterruptConfig).
                If a tool doesn't have an entry, it's auto-approved by default.
                * `True` indicates all interrupt config options are allowed.
                * `False` indicates that the tool is auto-approved.
                * `HumanInterruptConfig` indicates the specific interrupt config options to use.
            message_prefix: The prefix to use when constructing interrupts (requests for input).
                This is used to provide context about the tool call and the action being requested.
        """
        super().__init__()
        resolved_tool_configs = {}
        for tool_name, tool_config in tool_configs.items():
            if isinstance(tool_config, bool):
                if tool_config is True:
                    resolved_tool_configs[tool_name] = HumanInterruptConfig(
                        allow_approve=True,
                        allow_ignore=True,
                        allow_response=True,
                        allow_edit=True,
                    )
            else:
                resolved_tool_configs[tool_name] = tool_config
        self.tool_configs = resolved_tool_configs
        self.message_prefix = message_prefix

    def after_model(self, state: AgentState) -> dict[str, Any] | None:  # noqa: PLR0915
        """Trigger HITL flows for relevant tool calls after an AIMessage."""
        messages = state["messages"]
        if not messages:
            return None

        last_message = messages[-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return None

        # Separate tool calls that need interrupts from those that don't
        interrupt_tool_calls: dict[str, ToolCall] = {}
        auto_approved_tool_calls = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            if tool_name in self.tool_configs:
                # fix: id should not be typed as Optional on `langchain_core.messages.tool.ToolCall`
                interrupt_tool_calls[tool_call["id"]] = tool_call  # type: ignore[index]
            else:
                auto_approved_tool_calls.append(tool_call)

        # If no interrupts needed, return early
        if not interrupt_tool_calls:
            return None

        # Process all tool calls that require interrupts
        approved_tool_calls: list[ToolCall] = auto_approved_tool_calls.copy()
        rejected_tool_calls: list[ToolCall] = []
        artificial_tool_messages: list[ToolMessage] = []

        # Create interrupt requests for all tools that need approval
        interrupt_requests: list[HumanInterrupt] = []
        for tool_call in interrupt_tool_calls.values():
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            description = f"{self.message_prefix}\n\nTool: {tool_name}\nArgs: {tool_args}"
            tool_config = self.tool_configs[tool_name]

            request: HumanInterrupt = {
                "action": tool_name,
                "args": tool_args,
                "config": tool_config,
                "description": description,
                # ids should always be present on tool calls
                "tool_call_id": cast("str", tool_call["id"]),
            }
            interrupt_requests.append(request)

        responses: list[HumanResponse] = interrupt(interrupt_requests)

        for response in responses:
            try:
                tool_call = interrupt_tool_calls[response["tool_call_id"]]
            except KeyError:
                msg = (
                    f"Unexpected human response: {response}. "
                    f"Expected one with `'tool_call_id'` in {list(interrupt_tool_calls.keys())}."
                )
                raise ValueError(msg)

            tool_config = self.tool_configs[tool_call["name"]]

            if response["type"] == "approve" and tool_config["allow_approve"]:
                approved_tool_calls.append(tool_call)
            elif response["type"] == "edit" and tool_config["allow_edit"]:
                new_tool_call = ToolCall(
                    type="tool_call",
                    name=response["action"],
                    args=response["args"],
                    id=tool_call["id"],
                )
                approved_tool_calls.append(new_tool_call)
            elif response["type"] == "ignore" and tool_config["allow_ignore"]:
                rejected_tool_calls.append(tool_call)
                if isinstance(human_tool_message := response.get("tool_message"), ToolMessage):
                    tool_message = human_tool_message
                else:
                    if isinstance(human_tool_message, str):
                        content = human_tool_message
                    else:
                        content = (
                            f"User ignored the tool call for `{tool_call['name']}` "
                            f"with id {tool_call['id']}"
                        )

                    tool_message = ToolMessage(
                        content=content,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                        status="success",
                    )

                artificial_tool_messages.append(tool_message)
            elif response["type"] == "response" and tool_config["allow_response"]:
                rejected_tool_calls.append(tool_call)
                if isinstance(human_tool_message := response["tool_message"], ToolMessage):
                    tool_message = human_tool_message
                else:
                    tool_message = ToolMessage(
                        content=human_tool_message,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                        status="error",
                    )
                artificial_tool_messages.append(tool_message)
            else:
                allowed_types = [
                    type_name
                    for type_name, is_allowed in {
                        "accept": tool_config["allow_approve"],
                        "edit": tool_config["allow_edit"],
                        "response": tool_config["allow_response"],
                        "ignore": tool_config["allow_ignore"],
                    }.items()
                    if is_allowed
                ]

                msg = (
                    f"Unexpected human response: {response}. Response type '{response['type']}' "
                    f"is not allowed for tool '{tool_call['name']}'. "
                    f"Expected one with `'type'` in {allowed_types} based on "
                    f"the tool's interrupt configuration."
                )
                raise ValueError(msg)

        last_message.tool_calls = [*approved_tool_calls, *rejected_tool_calls]

        if len(approved_tool_calls) > 0:
            return {"messages": [last_message, *artificial_tool_messages]}

        return {"jump_to": "model", "messages": artificial_tool_messages}
