"""Shared utilities for interrupt request/response handling and state processing."""

from typing import Any

from langchain_core.messages import AIMessage, ToolCall, ToolMessage

from langchain.agents.middleware.human_in_the_loop import (
    Action,
    ActionRequest,
    Decision,
    HITLRequest,
    InterruptOnConfig,
    ReviewConfig,
)
from langchain.agents.middleware.types import AgentState


def build_hitl_request(
    tool_calls: list[ToolCall],
    interrupt_configs: dict[str, InterruptOnConfig],
    state: AgentState[Any],
    runtime: Any,
) -> HITLRequest:
    """Build a HITLRequest from tool calls that require interruption.

    Args:
        tool_calls: List of tool calls to potentially interrupt.
        interrupt_configs: Mapping of tool names to their interrupt configurations.
        state: Current agent state.
        runtime: Runtime context.

    Returns:
        A HITLRequest containing action requests and review configs for tools that need approval.
    """
    action_requests: list[ActionRequest] = []
    review_configs: list[ReviewConfig] = []

    for tool_call in tool_calls:
        if (config := interrupt_configs.get(tool_call["name"])) is not None:
            action_request, review_config = _create_action_and_config(
                tool_call, config, state, runtime
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
        # Use default description format from middleware
        description_prefix = "Tool execution requires approval"
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
