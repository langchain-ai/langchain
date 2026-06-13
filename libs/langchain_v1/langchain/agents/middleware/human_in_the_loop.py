"""Human in the loop middleware."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.config import get_config
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import interrupt
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ResponseT,
    StateT,
    ToolCallRequest,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.runtime import Runtime
    from langgraph.types import Command


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


DecisionType = Literal["approve", "edit", "reject", "respond", "accept", "replace"]

_BEFORE_DECISIONS: frozenset[DecisionType] = frozenset({"approve", "edit", "reject", "respond"})
"""Decisions valid when interrupting before a tool executes."""

_AFTER_DECISIONS: frozenset[DecisionType] = frozenset({"accept", "replace"})
"""Decisions valid when interrupting after a tool executes."""


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
    """The message sent to the model explaining why the action was rejected.

    If omitted, the model is told that the tool was not executed and should not
    retry the same tool call unless the user asks for it.
    """


class RespondDecision(TypedDict):
    """Response when a human answers on behalf of the tool, skipping execution.

    Used for "ask user" style tools whose real implementation is the human's
    response. The tool is not executed; instead, a synthetic `ToolMessage` with
    `status="success"` and the provided `message` is returned to the model.
    """

    type: Literal["respond"]
    """The type of response when a human responds on behalf of the tool."""

    message: str
    """Content of the synthetic `ToolMessage` returned to the model."""


class AcceptDecision(TypedDict):
    """Response when a human accepts a tool result produced after execution.

    Used with `interrupt_after` tools: the tool has already executed and the
    human keeps its result unchanged.
    """

    type: Literal["accept"]
    """The type of response when a human accepts the executed tool result."""


class ReplaceDecision(TypedDict):
    """Response when a human replaces a tool result produced after execution.

    Used with `interrupt_after` tools: the tool has already executed (for example,
    a request was posted to an event bus) and the human supplies the final result
    that should be returned to the model in place of the executed result.
    """

    type: Literal["replace"]
    """The type of response when a human replaces the executed tool result."""

    message: str
    """Content of the `ToolMessage` returned to the model in place of the result."""


Decision = (
    ApproveDecision
    | EditDecision
    | RejectDecision
    | RespondDecision
    | AcceptDecision
    | ReplaceDecision
)


class HITLResponse(TypedDict):
    """Response payload for a HITLRequest."""

    decisions: list[Decision]
    """The decisions made by the human."""


class _DescriptionFactory(Protocol):
    """Callable that generates a description for a tool call."""

    def __call__(
        self, tool_call: ToolCall, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> str:
        """Generate a description for a tool call."""
        ...


class InterruptOnConfig(TypedDict):
    """Configuration for an action requiring human in the loop.

    This is the configuration format used in the `HumanInTheLoopMiddleware.__init__`
    method.
    """

    allowed_decisions: list[DecisionType]
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

    when: NotRequired[Callable[[ToolCallRequest], bool]]
    """Optional predicate controlling whether to interrupt for a given tool call.

    Receives a `ToolCallRequest` and returns `True` to interrupt or `False` to
    auto-approve. Works in both `"batch"` and `"per_call"` modes.

    In `"batch"` mode the request is constructed with `tool=None` and
    `runtime` set to the node-level `Runtime` (not a `ToolRuntime`), so
    `request.runtime.tool_call_id` and `request.runtime.tools` are not available.
    In `"per_call"` mode the full `ToolCallRequest` from `wrap_tool_call` is passed.

    Example:
        ```python
        # Only interrupt delete_file calls targeting /etc
        config = InterruptOnConfig(
            allowed_decisions=["approve", "reject"],
            when=lambda req: req.tool_call["args"].get("path", "").startswith("/etc"),
        )
        ```
    """

    interrupt_after: NotRequired[bool]
    """Whether to interrupt *after* the tool executes instead of before.

    Defaults to `False`, which interrupts before execution (the tool call is
    reviewed and approved/edited/rejected before it runs).

    When `True`, the tool runs first and the agent halts immediately afterwards,
    surfacing the executed result for review. This is useful for long-running work
    that is delegated elsewhere (for example, posting a request to an event bus and
    resuming once an external worker produces the result).

    `interrupt_after` tools must restrict `allowed_decisions` to `"accept"` and/or
    `"replace"`. To avoid re-running the tool on resume, configure a `store` on the
    agent (or make the tool idempotent); the middleware caches the executed result
    in the store across the interrupt.
    """


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

                * `True` indicates all decisions are allowed: approve, edit, reject,
                    and respond.
                * `False` indicates that the tool is auto-approved.
                * `InterruptOnConfig` indicates the specific decisions allowed for this
                    tool.

                    The `InterruptOnConfig` can include a `description` field (`str` or
                    `Callable`) for custom formatting of the interrupt description.

                    A `when` predicate can also be provided to dynamically control
                    whether a tool call triggers an interrupt.
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
                        allowed_decisions=["approve", "edit", "reject", "respond"]
                    )
            elif tool_config.get("allowed_decisions"):
                self._validate_allowed_decisions(tool_name, tool_config)
                resolved_configs[tool_name] = tool_config
        self.interrupt_on = resolved_configs
        self.description_prefix = description_prefix

    _cache_namespace = ("__hitl_after_tool_cache__",)
    """Store namespace used to cache executed results for `interrupt_after` tools."""

    @staticmethod
    def _validate_allowed_decisions(tool_name: str, config: InterruptOnConfig) -> None:
        """Ensure `allowed_decisions` matches the interrupt timing for a tool.

        Args:
            tool_name: Name of the tool being configured.
            config: The tool's interrupt configuration.

        Raises:
            ValueError: If the decisions are incompatible with the interrupt timing.
        """
        allowed = set(config["allowed_decisions"])
        if config.get("interrupt_after"):
            invalid = allowed - _AFTER_DECISIONS
            if invalid:
                msg = (
                    f"Tool '{tool_name}' sets `interrupt_after=True`, so "
                    f"`allowed_decisions` may only contain {sorted(_AFTER_DECISIONS)}. "
                    f"Got invalid decisions: {sorted(invalid)}."
                )
                raise ValueError(msg)
        else:
            invalid = allowed & _AFTER_DECISIONS
            if invalid:
                msg = (
                    f"Tool '{tool_name}' interrupts before execution, so "
                    f"`allowed_decisions` may only contain {sorted(_BEFORE_DECISIONS)}. "
                    f"Decisions {sorted(invalid)} require `interrupt_after=True`."
                )
                raise ValueError(msg)

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
            content = decision.get("message") or (
                f"User rejected the tool call for `{tool_call['name']}` with id {tool_call['id']}. "
                "The tool was not executed. Do not retry this tool call unless the user "
                "explicitly requests it."
            )
            tool_message = ToolMessage(
                content=content,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                status="error",
            )
            return tool_call, tool_message
        if decision["type"] == "respond" and "respond" in allowed_decisions:
            # Skip tool execution; the human answers on behalf of the tool.
            tool_message = ToolMessage(
                content=decision["message"],
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                status="success",
            )
            return tool_call, tool_message
        msg = (
            f"Unexpected human decision: {decision}. "
            f"Decision type '{decision.get('type')}' "
            f"is not allowed for tool '{tool_call['name']}'. "
            f"Expected one of {allowed_decisions} based on the tool's configuration."
        )
        raise ValueError(msg)

    def _should_interrupt(
        self,
        tool_call: ToolCall,
        config: InterruptOnConfig,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> bool:
        """Return False if the `when` predicate rejects this tool call, True otherwise."""
        when = config.get("when")
        if when is None:
            return True
        try:
            runnable_config = get_config()
        except RuntimeError:
            runnable_config = {}
        tool_runtime = ToolRuntime(
            state=state,
            context=runtime.context,
            config=runnable_config,
            stream_writer=runtime.stream_writer,
            tool_call_id=tool_call["id"],
            store=runtime.store,
            execution_info=runtime.execution_info,
            server_info=runtime.server_info,
        )
        req = ToolCallRequest(
            tool_call=tool_call,
            tool=None,
            state=state,
            runtime=tool_runtime,  # type: ignore[arg-type]
        )
        return when(req)

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

        # Create action requests and review configs for tools that need approval
        action_requests: list[ActionRequest] = []
        review_configs: list[ReviewConfig] = []
        interrupt_indices: list[int] = []

        for idx, tool_call in enumerate(last_ai_msg.tool_calls):
            if (config := self.interrupt_on.get(tool_call["name"])) is not None:
                # `interrupt_after` tools are handled post-execution in `wrap_tool_call`.
                if config.get("interrupt_after"):
                    continue
                if not self._should_interrupt(tool_call, config, state, runtime):
                    continue
                action_request, review_config = self._create_action_and_config(
                    tool_call, config, state, runtime
                )
                action_requests.append(action_request)
                review_configs.append(review_config)
                interrupt_indices.append(idx)

        # If no interrupts needed, return early
        if not action_requests:
            return None

        # Create single HITLRequest with all actions and configs
        hitl_request = HITLRequest(
            action_requests=action_requests,
            review_configs=review_configs,
        )

        # Send interrupt and get response
        decisions = interrupt(hitl_request)["decisions"]

        # Validate that the number of decisions matches the number of interrupt tool calls
        if (decisions_len := len(decisions)) != (interrupt_count := len(interrupt_indices)):
            msg = (
                f"Number of human decisions ({decisions_len}) does not match "
                f"number of hanging tool calls ({interrupt_count})."
            )
            raise ValueError(msg)

        # Process decisions and rebuild tool calls in original order
        revised_tool_calls: list[ToolCall] = []
        artificial_tool_messages: list[ToolMessage] = []
        decision_idx = 0

        for idx, tool_call in enumerate(last_ai_msg.tool_calls):
            if idx in interrupt_indices:
                # This was an interrupt tool call - process the decision
                config = self.interrupt_on[tool_call["name"]]
                decision = decisions[decision_idx]
                decision_idx += 1

                revised_tool_call, tool_message = self._process_decision(
                    decision, tool_call, config
                )
                if revised_tool_call is not None:
                    revised_tool_calls.append(revised_tool_call)
                if tool_message:
                    artificial_tool_messages.append(tool_message)
            else:
                # This was auto-approved - keep original
                revised_tool_calls.append(tool_call)

        # Update the AI message to only include approved tool calls
        last_ai_msg.tool_calls = revised_tool_calls

        return {"messages": [last_ai_msg, *artificial_tool_messages]}

    async def aafter_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async trigger interrupt flows for relevant tool calls after an `AIMessage`.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Updated message with the revised tool calls.
        """
        return self.after_model(state, runtime)

    def _build_after_hitl_request(
        self,
        request: ToolCallRequest,
        config: InterruptOnConfig,
        result: ToolMessage | Command[Any],
    ) -> HITLRequest:
        """Build the interrupt request surfaced after a tool executes.

        Args:
            request: The tool call request that was executed.
            config: The tool's interrupt configuration.
            result: The result produced by executing the tool.

        Returns:
            A `HITLRequest` describing the executed action for human review.
        """
        tool_call = request.tool_call
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        result_content = result.content if isinstance(result, ToolMessage) else str(result)

        description_value = config.get("description")
        if callable(description_value):
            description = description_value(
                tool_call, request.state, cast("Runtime[ContextT]", request.runtime)
            )
        elif description_value is not None:
            description = description_value
        else:
            description = (
                f"{self.description_prefix}\n\nTool: {tool_name}\n"
                f"Args: {tool_args}\nResult: {result_content}"
            )

        action_request = ActionRequest(
            name=tool_name,
            args=tool_args,
            description=description,
        )
        review_config = ReviewConfig(
            action_name=tool_name,
            allowed_decisions=config["allowed_decisions"],
        )
        return HITLRequest(action_requests=[action_request], review_configs=[review_config])

    @staticmethod
    def _process_after_decision(
        decision: Decision,
        tool_call: ToolCall,
        config: InterruptOnConfig,
        result: ToolMessage | Command[Any],
    ) -> ToolMessage | Command[Any]:
        """Apply a post-execution decision to an executed tool result.

        Args:
            decision: The human decision for the executed tool.
            tool_call: The tool call that was executed.
            config: The tool's interrupt configuration.
            result: The result produced by executing the tool.

        Returns:
            The result to return to the model: the executed result for `accept`, or a
            replacement `ToolMessage` for `replace`.

        Raises:
            ValueError: If the decision type is not allowed for this tool.
        """
        allowed_decisions = config["allowed_decisions"]

        if decision["type"] == "accept" and "accept" in allowed_decisions:
            return result
        if decision["type"] == "replace" and "replace" in allowed_decisions:
            return ToolMessage(
                content=decision["message"],
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                status="success",
            )
        msg = (
            f"Unexpected human decision: {decision}. "
            f"Decision type '{decision.get('type')}' "
            f"is not allowed for tool '{tool_call['name']}'. "
            f"Expected one of {allowed_decisions} based on the tool's configuration."
        )
        raise ValueError(msg)

    def _execute_with_cache(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Execute the tool, caching `ToolMessage` results to survive the interrupt.

        When a `store` is available, the executed result is persisted before the
        interrupt and reused on resume so the tool is not invoked twice. Without a
        store, the tool is executed directly and will re-run on resume.
        """
        store = request.runtime.store
        if store is None:
            return handler(request)

        key = cast("str", request.tool_call["id"])
        if (item := store.get(self._cache_namespace, key)) is not None:
            cached = item.value.get("tool_message")
            if cached is not None:
                return ToolMessage.model_validate(cached)

        result = handler(request)
        if isinstance(result, ToolMessage):
            store.put(self._cache_namespace, key, {"tool_message": result.model_dump()})
        return result

    async def _aexecute_with_cache(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async variant of `_execute_with_cache`."""
        store = request.runtime.store
        if store is None:
            return await handler(request)

        key = cast("str", request.tool_call["id"])
        if (item := await store.aget(self._cache_namespace, key)) is not None:
            cached = item.value.get("tool_message")
            if cached is not None:
                return ToolMessage.model_validate(cached)

        result = await handler(request)
        if isinstance(result, ToolMessage):
            await store.aput(self._cache_namespace, key, {"tool_message": result.model_dump()})
        return result

    def _clear_cache(self, request: ToolCallRequest) -> None:
        """Remove a cached result once the post-execution decision is applied."""
        store = request.runtime.store
        if store is not None:
            store.delete(self._cache_namespace, cast("str", request.tool_call["id"]))

    async def _aclear_cache(self, request: ToolCallRequest) -> None:
        """Async variant of `_clear_cache`."""
        store = request.runtime.store
        if store is not None:
            await store.adelete(self._cache_namespace, cast("str", request.tool_call["id"]))

    @staticmethod
    def _validate_decisions_count(decisions: list[Decision]) -> Decision:
        """Validate that exactly one decision was returned for a single tool call."""
        if (decisions_len := len(decisions)) != 1:
            msg = (
                f"Number of human decisions ({decisions_len}) does not match "
                f"number of executed tool calls (1)."
            )
            raise ValueError(msg)
        return decisions[0]

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Interrupt after a tool executes for tools configured with `interrupt_after`.

        Tools without `interrupt_after` pass straight through to the handler. For
        `interrupt_after` tools, the tool runs first and the agent then halts via
        `interrupt`, surfacing the executed result so a human (or external system) can
        accept it or replace its content before the result reaches the model.

        Args:
            request: The tool call request to execute.
            handler: Callable that executes the tool.

        Returns:
            The (possibly replaced) `ToolMessage` or `Command` to return to the model.
        """
        config = self.interrupt_on.get(request.tool_call["name"])
        if (
            config is None
            or not config.get("interrupt_after")
            or not self._should_interrupt(
                request.tool_call,
                config,
                request.state,
                cast("Runtime[ContextT]", request.runtime),
            )
        ):
            return handler(request)

        result = self._execute_with_cache(request, handler)
        hitl_request = self._build_after_hitl_request(request, config, result)
        decision = self._validate_decisions_count(interrupt(hitl_request)["decisions"])
        final = self._process_after_decision(decision, request.tool_call, config, result)
        self._clear_cache(request)
        return final

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async trigger for interrupting after a tool executes.

        See `wrap_tool_call` for behavior details.

        Args:
            request: The tool call request to execute.
            handler: Async callable that executes the tool.

        Returns:
            The (possibly replaced) `ToolMessage` or `Command` to return to the model.
        """
        config = self.interrupt_on.get(request.tool_call["name"])
        if (
            config is None
            or not config.get("interrupt_after")
            or not self._should_interrupt(
                request.tool_call,
                config,
                request.state,
                cast("Runtime[ContextT]", request.runtime),
            )
        ):
            return await handler(request)

        result = await self._aexecute_with_cache(request, handler)
        hitl_request = self._build_after_hitl_request(request, config, result)
        decision = self._validate_decisions_count(interrupt(hitl_request)["decisions"])
        final = self._process_after_decision(decision, request.tool_call, config, result)
        await self._aclear_cache(request)
        return final
