"""Human in the loop middleware."""

from typing import Any, Literal, Protocol

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.runtime import Runtime
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware.async_interrupt import execute_interrupt_async
from langchain.agents.middleware.interrupt_utils import (
    Action,
    ActionRequest,
    Decision,
    DecisionType,
    HITLRequest,
    InterruptOnConfig,
    ReviewConfig,
    build_hitl_request,
    process_interrupt_response,
    validate_decision_count,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ResponseT,
    StateT,
)



class HumanInTheLoopMiddleware(AgentMiddleware[StateT, ContextT, ResponseT]):
    """Human in the loop middleware."""

    def __init__(
        self,
        interrupt_on: dict[str, bool | InterruptOnConfig],
        *,
        description_prefix: str = "Tool execution requires approval",
    ) -> None:
        """Initialize the human in the loop middleware.

        Args:
            interrupt_on: Mapping of tool name to allowed actions.

                If a tool doesn't have an entry, it's auto-approved by default.

                * `True` indicates all decisions are allowed: approve, edit, and reject.
                * `False` indicates that the tool is auto-approved.
                * `InterruptOnConfig` indicates the specific decisions allowed for this
                    tool.

                    The `InterruptOnConfig` can include a `description` field (`str` or
                    `Callable`) for custom formatting of the interrupt description.
            description_prefix: The prefix to use when constructing action requests.

                This is used to provide context about the tool call and the action being
                requested.

                Not used if a tool has a `description` in its `InterruptOnConfig`.
        """
        super().__init__()
        resolved_configs: dict[str, InterruptOnConfig] = {}
        for tool_name, tool_config in interrupt_on.items():
            if isinstance(tool_config, bool):
                if tool_config is True:
                    resolved_configs[tool_name] = InterruptOnConfig(
                        allowed_decisions=["approve", "edit", "reject"]
                    )
            elif tool_config.get("allowed_decisions"):
                resolved_configs[tool_name] = tool_config
        self.interrupt_on = resolved_configs
        self.description_prefix = description_prefix

    def _create_action_and_config(
        self,
        tool_call: ToolCall,
        config: InterruptOnConfig,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
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
            description = f"{self.description_prefix}\n\nTool: {tool_name}\nArgs: {tool_args}"

        # Create ActionRequest with description
        action_request = ActionRequest(
            name=tool_name,
            args=tool_args,
            description=description,
        )

        # Create ReviewConfig
        # eventually can get tool information and populate args_schema from there
        review_config = ReviewConfig(
            action_name=tool_name,
            allowed_decisions=config["allowed_decisions"],
        )

        return action_request, review_config

    @staticmethod
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

    def after_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Trigger interrupt flows for relevant tool calls after an `AIMessage`.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Updated message with the revised tool calls.

        Raises:
            ValueError: If the number of human decisions does not match the number of
                interrupted tool calls.
        """
        messages = state["messages"]
        if not messages:
            return None

        last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        # Find tool calls that require interruption
        interrupt_indices: list[int] = []
        for idx, tool_call in enumerate(last_ai_msg.tool_calls):
            if tool_call["name"] in self.interrupt_on:
                interrupt_indices.append(idx)

        # If no interrupts needed, return early
        if not interrupt_indices:
            return None

        # Build interrupt request for tools that need approval
        hitl_request = build_hitl_request(
            [last_ai_msg.tool_calls[idx] for idx in interrupt_indices],
            self.interrupt_on,
            state,
            runtime,
        )

        # Execute interrupt synchronously
        from langgraph.types import interrupt
        decisions = interrupt(hitl_request)["decisions"]

        # Validate decision count
        validate_decision_count(decisions, len(interrupt_indices))

        # Process decisions and update tool calls
        revised_tool_calls, artificial_messages = process_interrupt_response(
            decisions, last_ai_msg.tool_calls, self.interrupt_on, interrupt_indices
        )

        # Update the AI message with revised tool calls
        last_ai_msg.tool_calls = revised_tool_calls

        return {"messages": [last_ai_msg, *artificial_messages]}

    async def aafter_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async trigger interrupt flows for relevant tool calls after an `AIMessage`.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Updated message with the revised tool calls.

        Raises:
            ValueError: If the number of human decisions does not match the number of
                interrupted tool calls.
        """
        messages = state["messages"]
        if not messages:
            return None

        last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        # Find tool calls that require interruption
        interrupt_indices: list[int] = []
        for idx, tool_call in enumerate(last_ai_msg.tool_calls):
            if tool_call["name"] in self.interrupt_on:
                interrupt_indices.append(idx)

        # If no interrupts needed, return early
        if not interrupt_indices:
            return None

        # Build interrupt request for tools that need approval
        hitl_request = build_hitl_request(
            [last_ai_msg.tool_calls[idx] for idx in interrupt_indices],
            self.interrupt_on,
            state,
            runtime,
        )

        # Execute interrupt asynchronously (preserves runnable context)
        decisions = (await execute_interrupt_async(hitl_request))["decisions"]

        # Validate decision count
        validate_decision_count(decisions, len(interrupt_indices))

        # Process decisions and update tool calls
        revised_tool_calls, artificial_messages = process_interrupt_response(
            decisions, last_ai_msg.tool_calls, self.interrupt_on, interrupt_indices
        )

        # Update the AI message with revised tool calls
        last_ai_msg.tool_calls = revised_tool_calls

        return {"messages": [last_ai_msg, *artificial_messages]}
