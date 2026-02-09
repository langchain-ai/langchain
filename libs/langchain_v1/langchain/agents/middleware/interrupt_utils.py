"""Shared utilities for interrupt request/response handling and state processing."""

from typing import Any, Literal, Protocol

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware.types import AgentState


class Action(TypedDict):
    """Represents an action with a name and args."""

    name: str
    """The type or name of action being requested (e.g., `'add_numbers'`)."""

    args: dict[str, Any]
    """Key-value pairs of args needed for the action (e.g., `{"a": 1, "b": 2}`)."""


class ActionRequest(TypedDict):
    """Represents an action request with a name, args, and description."""

    name: str
    """The name of the action being requested."""

    args: dict[str, Any]
    """Key-value pairs of args needed for the action (e.g., `{"a": 1, "b": 2}`)."""

    description: NotRequired[str]
    """The description of the action to be reviewed."""


class _DescriptionFactory(Protocol):
    """Callable that generates a description for a tool call."""

    def __call__(
        self, tool_call: ToolCall, state: AgentState[Any], runtime: Any
    ) -> str:
        """Generate a description for a tool call."""
        ...


class InterruptOnConfig(TypedDict):
    """Configuration for an action requiring human in the loop.

    This is the configuration format used in the `HumanInTheLoopMiddleware.__init__`
    method.
    """

    allowed_decisions: list[str]
    """The decisions that are allowed for this action."""

    description: NotRequired[str | _DescriptionFactory]
    """The description attached to the request for human input.

    Can be either:

    - A static string describing the approval request
    - A callable that dynamically generates the description based on agent state,
        runtime, and tool call information

    Example:
        ```python
        # Static string description
        config = ToolConfig(
            allowed_decisions=["approve", "reject"],
            description="Please review this tool execution"
        )

        # Dynamic callable description
        def format_tool_description(
            tool_call: ToolCall,
            state: AgentState,
            runtime: Runtime[ContextT]
        ) -> str:
            import json
            return (
                f"Tool: {tool_call['name']}\\n"
                f"Arguments:\\n{json.dumps(tool_call['args'], indent=2)}"
            )

        config = InterruptOnConfig(
            allowed_decisions=["approve", "edit", "reject"],
            description=format_tool_description
        )
        ```
    """
    args_schema: NotRequired[dict[str, Any]]
    """JSON schema for the args associated with the action, if edits are allowed."""


DecisionType = Literal["approve", "edit", "reject"]


class ReviewConfig(TypedDict):
    """Policy for reviewing a HITL request."""

    action_name: str
    """Name of the action associated with this review configuration."""

    allowed_decisions: list[DecisionType]
    """The decisions that are allowed for this request."""

    args_schema: NotRequired[dict[str, Any]]
    """JSON schema for the args associated with the action, if edits are allowed."""


class HITLRequest(TypedDict):
    """Request for human feedback on a sequence of actions requested by a model."""

    action_requests: list[ActionRequest]
    """A list of agent actions for human review."""

    review_configs: list[ReviewConfig]
    """Review configuration for all possible actions."""


class ApproveDecision(TypedDict):
    """Response when a human approves the action."""

    type: Literal["approve"]
    """The type of response when a human approves the action."""


class EditDecision(TypedDict):
    """Response when a human edits the action."""

    type: Literal["edit"]
    """The type of response when a human edits the action."""

    edited_action: Action
    """Edited action for the agent to perform.

    Ex: for a tool call, a human reviewer can edit the tool name and args.
    """


class RejectDecision(TypedDict):
    """Response when a human rejects the action."""

    type: Literal["reject"]
    """The type of response when a human rejects the action."""

    message: NotRequired[str]
    """The message sent to the model explaining why the action was rejected."""


Decision = ApproveDecision | EditDecision | RejectDecision


class HITLResponse(TypedDict):
    """Response payload for a HITLRequest."""

    decisions: list[Decision]
    """The decisions made by the human."""


def build_hitl_request(
    tool_calls: list[ToolCall],
    interrupt_configs: dict[str, InterruptOnConfig],
    state: AgentState[Any],
    runtime: Any,
    description_prefix: str = "Tool execution requires approval",
) -> HITLRequest:
    """Build a HITLRequest from tool calls that require interruption.

    Args:
        tool_calls: List of tool calls to potentially interrupt.
        interrupt_configs: Mapping of tool names to their interrupt configurations.
        state: Current agent state.
        runtime: Runtime context.
        description_prefix: Prefix for default descriptions when tool config doesn't specify one.

    Returns:
        A HITLRequest containing action requests and review configs for tools that need approval.
    """
    action_requests: list[ActionRequest] = []
    review_configs: list[ReviewConfig] = []

    for tool_call in tool_calls:
        if (config := interrupt_configs.get(tool_call["name"])) is not None:
            action_request, review_config = _create_action_and_config(
                tool_call, config, state, runtime, description_prefix
            )
            action_requests.append(action_request)
            review_configs.append(review_config)

    return HITLRequest(
        action_requests=action_requests,
        review_configs=review_configs,
    )


def _create_action_and_config(
    tool_call: ToolCall,
    config: InterruptOnConfig,
    state: AgentState[Any],
    runtime: Any,
    description_prefix: str,
) -> tuple[ActionRequest, ReviewConfig]:
    """Create an ActionRequest and ReviewConfig for a tool call."""
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    # Generate description using the description field (str or callable)
    description_value = config.get("description")
    if callable(description_value):
        description = description_value(tool_call, state, runtime)
    elif description_value is not None:
        description = description_value
    else:
        # Use default description format with provided prefix
        description = f"{description_prefix}\n\nTool: {tool_name}\nArgs: {tool_args}"

    # Create ActionRequest with description
    action_request = ActionRequest(
        name=tool_name,
        args=tool_args,
        description=description,
    )

    # Create ReviewConfig
    review_config = ReviewConfig(
        action_name=tool_name,
        allowed_decisions=config["allowed_decisions"],
    )

    return action_request, review_config


def process_interrupt_response(
    decisions: list[Decision],
    original_tool_calls: list[ToolCall],
    interrupt_configs: dict[str, InterruptOnConfig],
    interrupt_indices: list[int],
) -> tuple[list[ToolCall], list[ToolMessage]]:
    """Process interrupt decisions and update tool calls accordingly.

    Args:
        decisions: List of decisions from the interrupt response.
        original_tool_calls: Original tool calls from the AI message.
        interrupt_configs: Mapping of tool names to their interrupt configurations.
        interrupt_indices: Indices of tool calls that were interrupted.

    Returns:
        A tuple of (updated_tool_calls, artificial_messages).
    """
    revised_tool_calls: list[ToolCall] = []
    artificial_messages: list[ToolMessage] = []
    decision_idx = 0

    for idx, tool_call in enumerate(original_tool_calls):
        if idx in interrupt_indices:
            # This was an interrupt tool call - process the decision
            config = interrupt_configs[tool_call["name"]]
            decision = decisions[decision_idx]
            decision_idx += 1

            revised_tool_call, tool_message = _process_decision(
                decision, tool_call, config
            )
            if revised_tool_call is not None:
                revised_tool_calls.append(revised_tool_call)
            if tool_message:
                artificial_messages.append(tool_message)
        else:
            # This was auto-approved - keep original
            revised_tool_calls.append(tool_call)

    return revised_tool_calls, artificial_messages


def _process_decision(
    decision: Decision,
    tool_call: ToolCall,
    config: InterruptOnConfig,
) -> tuple[ToolCall | None, ToolMessage | None]:
    """Process a single decision and return the revised tool call and optional tool message."""
    allowed_decisions = config["allowed_decisions"]

    if decision["type"] == "approve" and "approve" in allowed_decisions:
        return tool_call, None
    if decision["type"] == "edit" and "edit" in allowed_decisions:
        edited_action = decision["edited_action"]
        return (
            ToolCall(
                type="tool_call",
                name=edited_action["name"],
                args=edited_action["args"],
                id=tool_call["id"],
            ),
            None,
        )
    if decision["type"] == "reject" and "reject" in allowed_decisions:
        # Create a tool message with the human's text response
        content = decision.get("message") or (
            f"User rejected the tool call for `{tool_call['name']}` with id {tool_call['id']}"
        )
        tool_message = ToolMessage(
            content=content,
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
            status="error",
        )
        return tool_call, tool_message
    msg = (
        f"Unexpected human decision: {decision}. "
        f"Decision type '{decision.get('type')}' "
        f"is not allowed for tool '{tool_call['name']}'. "
        f"Expected one of {allowed_decisions} based on the tool's configuration."
    )
    raise ValueError(msg)


def validate_decision_count(decisions: list[Decision], expected_count: int) -> None:
    """Validate that the number of decisions matches the expected count.

    Args:
        decisions: List of decisions from interrupt response.
        expected_count: Expected number of decisions.

    Raises:
        ValueError: If the decision count doesn't match.
    """
    if (decisions_len := len(decisions)) != expected_count:
        msg = (
            f"Number of human decisions ({decisions_len}) does not match "
            f"number of hanging tool calls ({expected_count})."
        )
        raise ValueError(msg)
