"""Middleware agent implementation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Generic, cast, get_args, get_origin, get_type_hints

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable, run_in_executor
from langchain_core.tools import BaseTool
from langgraph._internal._runnable import RunnableCallable
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from langgraph.typing import ContextT
from typing_extensions import NotRequired, Required, TypedDict, TypeVar

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    JumpTo,
    MiddlewareHookInfo,
    ModelRequest,
    OmitFromSchema,
    PublicAgentState,
)
from langchain.agents.structured_output import (
    AutoStrategy,
    MultipleStructuredOutputsError,
    OutputToolBinding,
    ProviderStrategy,
    ProviderStrategyBinding,
    ResponseFormat,
    StructuredOutputValidationError,
    ToolStrategy,
)
from langchain.chat_models import init_chat_model
from langchain.tools import ToolNode

STRUCTURED_OUTPUT_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."

ResponseT = TypeVar("ResponseT")


# ============================================================================
# Data Structures for Agent Graph Construction
# ============================================================================


@dataclass
class MiddlewareHooks:
    """Middleware hooks categorized by type for graph construction."""

    before_agent: list[MiddlewareHookInfo]
    """Hooks that run once before the agent starts."""

    before_model: list[MiddlewareHookInfo]
    """Hooks that run before each model call in the agent loop."""

    modify_model_request: list[MiddlewareHookInfo]
    """Hooks that modify the model request before calling the model."""

    after_model: list[MiddlewareHookInfo]
    """Hooks that run after each model call in the agent loop."""

    after_agent: list[MiddlewareHookInfo]
    """Hooks that run once after the agent completes."""

    retry: list[MiddlewareHookInfo]
    """Hooks that handle model invocation errors and optionally retry."""

    @classmethod
    def from_middleware_list(
        cls,
        middleware: Sequence[AgentMiddleware[AgentState[ResponseT], ContextT]],
    ) -> "MiddlewareHooks":
        """Extract and categorize all hooks from middleware instances.

        Args:
            middleware: Sequence of middleware instances to analyze.

        Returns:
            MiddlewareHooks with all hooks organized by type.
        """
        hooks_by_type: dict[str, list[MiddlewareHookInfo]] = {
            "before_agent": [],
            "before_model": [],
            "modify_model_request": [],
            "after_model": [],
            "after_agent": [],
            "retry": [],
        }

        # Map hook names to their category
        hook_name_mapping = {
            "before_agent": "before_agent",
            "before_model": "before_model",
            "modify_model_request": "modify_model_request",
            "after_model": "after_model",
            "after_agent": "after_agent",
            "retry_model_request": "retry",
        }

        for m in middleware:
            for hook_name, category in hook_name_mapping.items():
                if hook_info := m.hook_info(hook_name):
                    hooks_by_type[category].append(hook_info)

        return cls(
            before_agent=hooks_by_type["before_agent"],
            before_model=hooks_by_type["before_model"],
            modify_model_request=hooks_by_type["modify_model_request"],
            after_model=hooks_by_type["after_model"],
            after_agent=hooks_by_type["after_agent"],
            retry=hooks_by_type["retry"],
        )


@dataclass
class AgentComponents:
    """Core components and configuration for agent construction."""

    model: BaseChatModel
    """The language model to use for the agent."""

    tool_node: ToolNode | None
    """The tool execution node, or None if no tools are available."""

    middleware_hooks: MiddlewareHooks
    """Middleware hooks organized by type."""

    structured_output_tools: dict[str, OutputToolBinding]
    """Tools used for structured output parsing."""

    default_tools: list[BaseTool | dict]
    """Default tools available to the agent (regular tools + middleware tools + built-ins)."""

    initial_response_format: ResponseFormat | None
    """The initial response format configuration."""

    system_prompt: str | None
    """The system prompt for the agent."""


@dataclass
class GraphTopology:
    """Key nodes in the graph topology defining the execution flow.

    The agent graph has the following structure:
    START -> entry_node -> [loop: loop_entry_node -> model -> loop_exit_node -> tools]
    -> exit_node -> END

    - entry_node: Runs once at the start (before_agent hooks)
    - loop_entry_node: Beginning of agent loop (before_model hooks)
    - loop_exit_node: End of each loop iteration (after_model hooks)
    - exit_node: Runs once at the end (after_agent hooks) or END
    """

    entry_node: str
    """The first node executed (START -> entry_node)."""

    loop_entry_node: str
    """Where the agent loop begins (where tools loop back to)."""

    loop_exit_node: str
    """The last node in each loop iteration."""

    exit_node: str
    """The final node before END (or END itself)."""

    @classmethod
    def compute(cls, hooks: MiddlewareHooks) -> "GraphTopology":
        """Compute graph topology from middleware hook configuration.

        Args:
            hooks: The categorized middleware hooks.

        Returns:
            GraphTopology describing the flow through the graph.
        """
        # Entry node (runs once at start): before_agent -> before_model -> model_request
        if hooks.before_agent:
            entry_node = hooks.before_agent[0].node_name
        elif hooks.before_model:
            entry_node = hooks.before_model[0].node_name
        else:
            entry_node = "model_request"

        # Loop entry node (beginning of agent loop, excludes before_agent)
        loop_entry_node = hooks.before_model[0].node_name if hooks.before_model else "model_request"

        # Loop exit node (end of each iteration, excludes after_agent)
        loop_exit_node = hooks.after_model[0].node_name if hooks.after_model else "model_request"

        # Exit node (runs once at end): after_agent or END
        exit_node = hooks.after_agent[-1].node_name if hooks.after_agent else END

        return cls(
            entry_node=entry_node,
            loop_entry_node=loop_entry_node,
            loop_exit_node=loop_exit_node,
            exit_node=exit_node,
        )


def _resolve_schema(schemas: set[type], schema_name: str, omit_flag: str | None = None) -> type:
    """Resolve schema by merging schemas and optionally respecting OmitFromSchema annotations.

    Args:
        schemas: List of schema types to merge
        schema_name: Name for the generated TypedDict
        omit_flag: If specified, omit fields with this flag set ('input' or 'output')
    """
    all_annotations = {}

    for schema in schemas:
        hints = get_type_hints(schema, include_extras=True)

        for field_name, field_type in hints.items():
            should_omit = False

            if omit_flag:
                # Check for omission in the annotation metadata
                metadata = _extract_metadata(field_type)
                for meta in metadata:
                    if isinstance(meta, OmitFromSchema) and getattr(meta, omit_flag) is True:
                        should_omit = True
                        break

            if not should_omit:
                all_annotations[field_name] = field_type

    return TypedDict(schema_name, all_annotations)  # type: ignore[operator]


def _extract_metadata(type_: type) -> list:
    """Extract metadata from a field type, handling Required/NotRequired and Annotated wrappers."""
    # Handle Required[Annotated[...]] or NotRequired[Annotated[...]]
    if get_origin(type_) in (Required, NotRequired):
        inner_type = get_args(type_)[0]
        if get_origin(inner_type) is Annotated:
            return list(get_args(inner_type)[1:])

    # Handle direct Annotated[...]
    elif get_origin(type_) is Annotated:
        return list(get_args(type_)[1:])

    return []


# ============================================================================
# Setup and Initialization Functions
# ============================================================================


def _setup_components(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | ToolNode | None,
    middleware: Sequence[AgentMiddleware[AgentState[ResponseT], ContextT]],
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | None,
    system_prompt: str | None,
) -> AgentComponents:
    """Setup and validate agent components.

    Args:
        model: Model name or instance.
        tools: Tools for the agent.
        middleware: Middleware instances.
        response_format: Response format configuration.
        system_prompt: System prompt for the agent.

    Returns:
        AgentComponents with all components configured and validated.
    """
    # Initialize chat model
    if isinstance(model, str):
        model = init_chat_model(model)

    # Handle tools being None or empty
    if tools is None:
        tools = []

    # Convert response format and setup structured output tools
    initial_response_format: ToolStrategy | ProviderStrategy | AutoStrategy | None
    if response_format is None:
        initial_response_format = None
    elif isinstance(response_format, (ToolStrategy, ProviderStrategy, AutoStrategy)):
        initial_response_format = response_format
    else:
        # Raw schema - wrap in AutoStrategy to enable auto-detection
        initial_response_format = AutoStrategy(schema=response_format)

    # For AutoStrategy, convert to ToolStrategy to setup tools upfront
    tool_strategy_for_setup: ToolStrategy | None = None
    if isinstance(initial_response_format, AutoStrategy):
        tool_strategy_for_setup = ToolStrategy(schema=initial_response_format.schema)
    elif isinstance(initial_response_format, ToolStrategy):
        tool_strategy_for_setup = initial_response_format

    structured_output_tools: dict[str, OutputToolBinding] = {}
    if tool_strategy_for_setup:
        for response_schema in tool_strategy_for_setup.schema_specs:
            structured_tool_info = OutputToolBinding.from_schema_spec(response_schema)
            structured_output_tools[structured_tool_info.tool.name] = structured_tool_info

    middleware_tools = [t for m in middleware for t in getattr(m, "tools", [])]

    # Setup tools
    tool_node: ToolNode | None = None
    default_tools: list[BaseTool | dict[str, Any]]

    if isinstance(tools, list):
        # Extract built-in provider tools (dict format) and regular tools (BaseTool)
        built_in_tools = [t for t in tools if isinstance(t, dict)]
        regular_tools = [t for t in tools if not isinstance(t, dict)]

        # Tools that require client-side execution
        available_tools = middleware_tools + regular_tools

        # Only create ToolNode if we have client-side tools
        tool_node = ToolNode(tools=available_tools) if available_tools else None

        # Default tools for ModelRequest initialization
        default_tools = regular_tools + middleware_tools + built_in_tools
    elif isinstance(tools, ToolNode):
        tool_node = tools
        if tool_node:
            # Add middleware tools to existing ToolNode
            available_tools = list(tool_node.tools_by_name.values()) + middleware_tools
            tool_node = ToolNode(available_tools)

            # default_tools includes all client-side tools
            default_tools = available_tools
        else:
            default_tools = middleware_tools
    else:
        # No tools provided, only middleware_tools available
        default_tools = middleware_tools

    # Validate middleware
    assert len({m.name for m in middleware}) == len(middleware), (  # noqa: S101
        "Please remove duplicate middleware instances."
    )

    # Categorize middleware by hooks
    middleware_hooks = MiddlewareHooks.from_middleware_list(middleware)

    return AgentComponents(
        model=model,
        tool_node=tool_node,
        middleware_hooks=middleware_hooks,
        structured_output_tools=structured_output_tools,
        default_tools=default_tools,
        initial_response_format=initial_response_format,
        system_prompt=system_prompt,
    )


def _supports_provider_strategy(model: str | BaseChatModel) -> bool:
    """Check if a model supports provider-specific structured output.

    Args:
        model: Model name string or BaseChatModel instance.

    Returns:
        ``True`` if the model supports provider-specific structured output, ``False`` otherwise.
    """
    model_name: str | None = None
    if isinstance(model, str):
        model_name = model
    elif isinstance(model, BaseChatModel):
        model_name = getattr(model, "model_name", None)

    return (
        "grok" in model_name.lower()
        or any(part in model_name for part in ["gpt-5", "gpt-4.1", "gpt-oss", "o3-pro", "o3-mini"])
        if model_name
        else False
    )


# ============================================================================
# Node Building Functions
# ============================================================================


def _create_hook_node(hook_info: MiddlewareHookInfo) -> RunnableCallable:
    """Create a graph node for a middleware hook.

    Args:
        hook_info: Information about the hook to create a node for.

    Returns:
        RunnableCallable that supports both sync and async execution.
    """
    return RunnableCallable(hook_info.sync_fn, hook_info.async_fn, trace=False)


def _add_middleware_nodes(
    graph: StateGraph[AgentState, ContextT, PublicAgentState, PublicAgentState],
    components: AgentComponents,
    state_schema: type,
) -> None:
    """Add all middleware hook nodes to the graph.

    Args:
        graph: The state graph to add nodes to.
        components: Agent components with middleware hooks.
        state_schema: The state schema for input validation.
    """
    hooks = components.middleware_hooks

    # Add before_agent nodes
    for hook_info in hooks.before_agent:
        node = _create_hook_node(hook_info)
        graph.add_node(hook_info.node_name, node, input_schema=state_schema)

    # Add before_model nodes
    for hook_info in hooks.before_model:
        node = _create_hook_node(hook_info)
        graph.add_node(hook_info.node_name, node, input_schema=state_schema)

    # Add after_model nodes
    for hook_info in hooks.after_model:
        node = _create_hook_node(hook_info)
        graph.add_node(hook_info.node_name, node, input_schema=state_schema)

    # Add after_agent nodes
    for hook_info in hooks.after_agent:
        node = _create_hook_node(hook_info)
        graph.add_node(hook_info.node_name, node, input_schema=state_schema)


def _handle_structured_output_error(
    exception: Exception,
    response_format: ResponseFormat,
) -> tuple[bool, str]:
    """Handle structured output error. Returns (should_retry, retry_tool_message)."""
    if not isinstance(response_format, ToolStrategy):
        return False, ""

    handle_errors = response_format.handle_errors

    if handle_errors is False:
        return False, ""
    if handle_errors is True:
        return True, STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(exception))
    if isinstance(handle_errors, str):
        return True, handle_errors
    if isinstance(handle_errors, type) and issubclass(handle_errors, Exception):
        if isinstance(exception, handle_errors):
            return True, STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(exception))
        return False, ""
    if isinstance(handle_errors, tuple):
        if any(isinstance(exception, exc_type) for exc_type in handle_errors):
            return True, STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(exception))
        return False, ""
    if callable(handle_errors):
        # type narrowing not working appropriately w/ callable check, can fix later
        return True, handle_errors(exception)  # type: ignore[return-value,call-arg]
    return False, ""


# ============================================================================
# Edge Building Functions
# ============================================================================


def _connect_entry_edges(
    graph: StateGraph[AgentState, ContextT, PublicAgentState, PublicAgentState],
    topology: GraphTopology,
) -> None:
    """Connect the entry edge from START to the entry node.

    Args:
        graph: The state graph to add edges to.
        topology: Graph topology configuration.
    """
    graph.add_edge(START, topology.entry_node)


def _connect_loop_edges(
    graph: StateGraph[AgentState, ContextT, PublicAgentState, PublicAgentState],
    topology: GraphTopology,
    components: AgentComponents,
) -> None:
    """Connect conditional edges for the agent loop (tools <-> model).

    Args:
        graph: The state graph to add edges to.
        topology: Graph topology configuration.
        components: Agent components with tool configuration.
    """
    tool_node = components.tool_node
    structured_output_tools = components.structured_output_tools

    if tool_node is None:
        # No tools - connect loop_exit directly to exit_node
        if topology.loop_exit_node == "model_request":
            graph.add_edge(topology.loop_exit_node, topology.exit_node)
        else:
            # We have after_model but no tools
            _add_middleware_edge(
                graph,
                topology.loop_exit_node,
                topology.exit_node,
                topology.loop_entry_node,
                can_jump_to=components.middleware_hooks.after_model[0].can_jump_to,
            )
        return

    # Add conditional edge from tools back to model or exit
    graph.add_conditional_edges(
        "tools",
        _make_tools_to_model_edge(
            tool_node, topology.loop_entry_node, structured_output_tools, topology.exit_node
        ),
        [topology.loop_entry_node, topology.exit_node],
    )

    # Add conditional edge from model to tools or exit
    graph.add_conditional_edges(
        topology.loop_exit_node,
        _make_model_to_tools_edge(
            topology.loop_entry_node, structured_output_tools, tool_node, topology.exit_node
        ),
        [topology.loop_entry_node, "tools", topology.exit_node],
    )


def _connect_middleware_chains(
    graph: StateGraph[AgentState, ContextT, PublicAgentState, PublicAgentState],
    components: AgentComponents,
    topology: GraphTopology,
) -> None:
    """Connect middleware hooks in chains.

    Args:
        graph: The state graph to add edges to.
        components: Agent components with middleware hooks.
        topology: Graph topology configuration.
    """
    hooks = components.middleware_hooks

    # Connect before_agent chain
    if hooks.before_agent:
        for i in range(len(hooks.before_agent) - 1):
            _add_middleware_edge(
                graph,
                hooks.before_agent[i].node_name,
                hooks.before_agent[i + 1].node_name,
                topology.loop_entry_node,
                can_jump_to=hooks.before_agent[i].can_jump_to,
            )
        # Connect last before_agent to loop_entry_node
        _add_middleware_edge(
            graph,
            hooks.before_agent[-1].node_name,
            topology.loop_entry_node,
            topology.loop_entry_node,
            can_jump_to=hooks.before_agent[-1].can_jump_to,
        )

    # Connect before_model chain
    if hooks.before_model:
        for i in range(len(hooks.before_model) - 1):
            _add_middleware_edge(
                graph,
                hooks.before_model[i].node_name,
                hooks.before_model[i + 1].node_name,
                topology.loop_entry_node,
                can_jump_to=hooks.before_model[i].can_jump_to,
            )
        # Connect last before_model to model_request
        _add_middleware_edge(
            graph,
            hooks.before_model[-1].node_name,
            "model_request",
            topology.loop_entry_node,
            can_jump_to=hooks.before_model[-1].can_jump_to,
        )

    # Connect after_model chain (reverse order)
    if hooks.after_model:
        graph.add_edge("model_request", hooks.after_model[-1].node_name)
        for i in range(len(hooks.after_model) - 1, 0, -1):
            _add_middleware_edge(
                graph,
                hooks.after_model[i].node_name,
                hooks.after_model[i - 1].node_name,
                topology.loop_entry_node,
                can_jump_to=hooks.after_model[i].can_jump_to,
            )

    # Connect after_agent chain (reverse order)
    if hooks.after_agent:
        for i in range(len(hooks.after_agent) - 1, 0, -1):
            _add_middleware_edge(
                graph,
                hooks.after_agent[i].node_name,
                hooks.after_agent[i - 1].node_name,
                topology.loop_entry_node,
                can_jump_to=hooks.after_agent[i].can_jump_to,
            )
        # Connect first after_agent to END
        _add_middleware_edge(
            graph,
            hooks.after_agent[0].node_name,
            END,
            topology.loop_entry_node,
            can_jump_to=hooks.after_agent[0].can_jump_to,
        )


def create_agent(  # noqa: PLR0915
    *,
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | ToolNode | None = None,
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware[AgentState[ResponseT], ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | None = None,
    context_schema: type[ContextT] | None = None,
) -> StateGraph[
    AgentState[ResponseT], ContextT, PublicAgentState[ResponseT], PublicAgentState[ResponseT]
]:
    """Create a middleware agent graph.

    Args:
        model: Model name or BaseChatModel instance.
        tools: Tools for the agent to use.
        system_prompt: System prompt for the agent.
        middleware: Middleware instances to customize agent behavior.
        response_format: Response format configuration for structured outputs.
        context_schema: Context schema for the graph runtime.

    Returns:
        StateGraph configured with all nodes and edges.
    """
    # Phase 1: Setup and validate components
    components = _setup_components(model, tools, middleware, response_format, system_prompt)

    # Phase 2: Create schemas
    state_schemas = {m.state_schema for m in middleware}
    state_schemas.add(AgentState)

    state_schema = _resolve_schema(state_schemas, "StateSchema", None)
    input_schema = _resolve_schema(state_schemas, "InputSchema", "input")
    output_schema = _resolve_schema(state_schemas, "OutputSchema", "output")

    # Phase 3: Create graph
    graph: StateGraph[
        AgentState[ResponseT], ContextT, PublicAgentState[ResponseT], PublicAgentState[ResponseT]
    ] = StateGraph(
        state_schema=state_schema,
        input_schema=input_schema,
        output_schema=output_schema,
        context_schema=context_schema,
    )

    # Phase 4: Define model request handlers (need access to components via closure)
    # These are inner functions because they need access to components
    structured_output_tools = components.structured_output_tools
    default_tools = components.default_tools
    initial_response_format = components.initial_response_format
    model_instance = components.model
    middleware_w_modify_model_request = components.middleware_hooks.modify_model_request
    middleware_w_retry = components.middleware_hooks.retry

    def _handle_model_output(
        output: AIMessage, effective_response_format: ResponseFormat | None
    ) -> dict[str, Any]:
        """Handle model output including structured responses.

        Args:
            output: The AI message output from the model.
            effective_response_format: The actual strategy used
                (may differ from initial if auto-detected).
        """
        # Handle structured output with provider strategy
        if isinstance(effective_response_format, ProviderStrategy):
            if not output.tool_calls:
                provider_strategy_binding = ProviderStrategyBinding.from_schema_spec(
                    effective_response_format.schema_spec
                )
                structured_response = provider_strategy_binding.parse(output)
                return {"messages": [output], "structured_response": structured_response}
            return {"messages": [output]}

        # Handle structured output with tool strategy
        if (
            isinstance(effective_response_format, ToolStrategy)
            and isinstance(output, AIMessage)
            and output.tool_calls
        ):
            structured_tool_calls = [
                tc for tc in output.tool_calls if tc["name"] in structured_output_tools
            ]

            if structured_tool_calls:
                exception: Exception | None = None
                if len(structured_tool_calls) > 1:
                    # Handle multiple structured outputs error
                    tool_names = [tc["name"] for tc in structured_tool_calls]
                    exception = MultipleStructuredOutputsError(tool_names)
                    should_retry, error_message = _handle_structured_output_error(
                        exception, effective_response_format
                    )
                    if not should_retry:
                        raise exception

                    # Add error messages and retry
                    tool_messages = [
                        ToolMessage(
                            content=error_message,
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        )
                        for tc in structured_tool_calls
                    ]
                    return {"messages": [output, *tool_messages]}

                # Handle single structured output
                tool_call = structured_tool_calls[0]
                try:
                    structured_tool_binding = structured_output_tools[tool_call["name"]]
                    structured_response = structured_tool_binding.parse(tool_call["args"])

                    tool_message_content = (
                        effective_response_format.tool_message_content
                        if effective_response_format.tool_message_content
                        else f"Returning structured response: {structured_response}"
                    )

                    return {
                        "messages": [
                            output,
                            ToolMessage(
                                content=tool_message_content,
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                            ),
                        ],
                        "structured_response": structured_response,
                    }
                except Exception as exc:  # noqa: BLE001
                    exception = StructuredOutputValidationError(tool_call["name"], exc)
                    should_retry, error_message = _handle_structured_output_error(
                        exception, effective_response_format
                    )
                    if not should_retry:
                        raise exception

                    return {
                        "messages": [
                            output,
                            ToolMessage(
                                content=error_message,
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                            ),
                        ],
                    }

        return {"messages": [output]}

    def _get_bound_model(request: ModelRequest) -> tuple[Runnable, ResponseFormat | None]:
        """Get the model with appropriate tool bindings.

        Performs auto-detection of strategy if needed based on model capabilities.

        Args:
            request: The model request containing model, tools, and response format.

        Returns:
            Tuple of (bound_model, effective_response_format) where ``effective_response_format``
            is the actual strategy used (may differ from initial if auto-detected).
        """
        # Validate ONLY client-side tools that need to exist in tool_node
        # Build map of available client-side tools (regular_tools + middleware_tools)
        available_tools_by_name = {t.name: t for t in default_tools if isinstance(t, BaseTool)}

        # Check if any requested tools are unknown CLIENT-SIDE tools
        unknown_tool_names = []
        for t in request.tools:
            # Only validate BaseTool instances (skip built-in dict tools)
            if isinstance(t, dict):
                continue
            if t.name not in available_tools_by_name:
                unknown_tool_names.append(t.name)

        if unknown_tool_names:
            available_tool_names = sorted(available_tools_by_name.keys())
            msg = (
                f"Middleware returned unknown tool names: {unknown_tool_names}\n\n"
                f"Available client-side tools: {available_tool_names}\n\n"
                "To fix this issue:\n"
                "1. Ensure the tools are passed to create_agent() via "
                "the 'tools' parameter\n"
                "2. If using custom middleware with tools, ensure "
                "they're registered via middleware.tools attribute\n"
                "3. Verify that tool names in ModelRequest.tools match "
                "the actual tool.name values\n"
                "Note: Built-in provider tools (dict format) can be added dynamically."
            )
            raise ValueError(msg)

        # Determine effective response format (auto-detect if needed)
        effective_response_format: ResponseFormat | None
        if isinstance(request.response_format, AutoStrategy):
            # User provided raw schema via AutoStrategy - auto-detect best strategy based on model
            if _supports_provider_strategy(request.model):
                # Model supports provider strategy - use it
                effective_response_format = ProviderStrategy(schema=request.response_format.schema)
            else:
                # Model doesn't support provider strategy - use ToolStrategy
                effective_response_format = ToolStrategy(schema=request.response_format.schema)
        else:
            # User explicitly specified a strategy - preserve it
            effective_response_format = request.response_format

        # Build final tools list including structured output tools
        # request.tools already contains both BaseTool and dict (built-in) tools
        final_tools = list(request.tools)
        if isinstance(effective_response_format, ToolStrategy):
            # Add structured output tools to final tools list
            structured_tools = [info.tool for info in structured_output_tools.values()]
            final_tools.extend(structured_tools)

        # Bind model based on effective response format
        if isinstance(effective_response_format, ProviderStrategy):
            # Use provider-specific structured output
            kwargs = effective_response_format.to_model_kwargs()
            return (
                request.model.bind_tools(
                    final_tools, strict=True, **kwargs, **request.model_settings
                ),
                effective_response_format,
            )

        if isinstance(effective_response_format, ToolStrategy):
            # Current implementation requires that tools used for structured output
            # have to be declared upfront when creating the agent as part of the
            # response format. Middleware is allowed to change the response format
            # to a subset of the original structured tools when using ToolStrategy,
            # but not to add new structured tools that weren't declared upfront.
            # Compute output binding
            for tc in effective_response_format.schema_specs:
                if tc.name not in structured_output_tools:
                    msg = (
                        f"ToolStrategy specifies tool '{tc.name}' "
                        "which wasn't declared in the original "
                        "response format when creating the agent."
                    )
                    raise ValueError(msg)

            # Force tool use if we have structured output tools
            tool_choice = "any" if structured_output_tools else request.tool_choice
            return (
                request.model.bind_tools(
                    final_tools, tool_choice=tool_choice, **request.model_settings
                ),
                effective_response_format,
            )

        # No structured output - standard model binding
        if final_tools:
            return (
                request.model.bind_tools(
                    final_tools, tool_choice=request.tool_choice, **request.model_settings
                ),
                None,
            )
        return request.model.bind(**request.model_settings), None

    def model_request(state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any]:
        """Sync model request handler with sequential middleware processing."""
        request = ModelRequest(
            model=model_instance,
            tools=default_tools,
            system_prompt=components.system_prompt,
            response_format=initial_response_format,
            messages=state["messages"],
            tool_choice=None,
        )

        # Apply modify_model_request middleware in sequence
        for hook_info in middleware_w_modify_model_request:
            if hook_info.sync_fn:
                hook_info.sync_fn(request, state, runtime)
            else:
                msg = (
                    f"No synchronous function provided for "
                    f"{hook_info.middleware_name}.amodify_model_request"
                    "\nEither initialize with a synchronous function or invoke"
                    " via the async API (ainvoke, astream, etc.)"
                )
                raise TypeError(msg)

        # Retry loop for model invocation with error handling
        max_attempts = 100
        for attempt in range(1, max_attempts + 1):
            try:
                # Get the bound model (with auto-detection if needed)
                model_, effective_response_format = _get_bound_model(request)
                messages = request.messages
                if request.system_prompt:
                    messages = [SystemMessage(request.system_prompt), *messages]

                output = model_.invoke(messages)
                return {
                    "thread_model_call_count": state.get("thread_model_call_count", 0) + 1,
                    "run_model_call_count": state.get("run_model_call_count", 0) + 1,
                    **_handle_model_output(output, effective_response_format),
                }
            except Exception as error:
                # Try retry_model_request on each middleware
                for hook_info in middleware_w_retry:
                    if hook_info.sync_fn:
                        if retry_request := hook_info.sync_fn(
                            error, request, state, runtime, attempt
                        ):
                            request = retry_request
                            break
                    else:
                        msg = (
                            f"No synchronous function provided for "
                            f"{hook_info.middleware_name}.aretry_model_request"
                            "\nEither initialize with a synchronous function or invoke"
                            " via the async API (ainvoke, astream, etc.)"
                        )
                        raise TypeError(msg)
                else:
                    raise

        # If we exit the loop, max attempts exceeded
        msg = f"Maximum retry attempts ({max_attempts}) exceeded"
        raise RuntimeError(msg)

    async def amodel_request(state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any]:
        """Async model request handler with sequential middleware processing."""
        request = ModelRequest(
            model=model_instance,
            tools=default_tools,
            system_prompt=components.system_prompt,
            response_format=initial_response_format,
            messages=state["messages"],
            tool_choice=None,
        )

        # Apply modify_model_request middleware in sequence
        for hook_info in middleware_w_modify_model_request:
            if hook_info.async_fn:
                await hook_info.async_fn(request, state, runtime)
            elif hook_info.sync_fn:
                # Fallback to sync if only sync is implemented
                await run_in_executor(None, hook_info.sync_fn, request, state, runtime)
            else:
                msg = f"No function provided for {hook_info.middleware_name}.modify_model_request"
                raise RuntimeError(msg)

        # Retry loop for model invocation with error handling
        max_attempts = 100
        for attempt in range(1, max_attempts + 1):
            try:
                # Get the bound model (with auto-detection if needed)
                model_, effective_response_format = _get_bound_model(request)
                messages = request.messages
                if request.system_prompt:
                    messages = [SystemMessage(request.system_prompt), *messages]

                output = await model_.ainvoke(messages)
                return {
                    "thread_model_call_count": state.get("thread_model_call_count", 0) + 1,
                    "run_model_call_count": state.get("run_model_call_count", 0) + 1,
                    **_handle_model_output(output, effective_response_format),
                }
            except Exception as error:
                # Try retry_model_request on each middleware
                for hook_info in middleware_w_retry:
                    retry_request = None
                    if hook_info.async_fn:
                        retry_request = await hook_info.async_fn(
                            error, request, state, runtime, attempt
                        )
                    elif hook_info.sync_fn:
                        # Fallback to sync if only sync is implemented
                        retry_request = await run_in_executor(
                            None, hook_info.sync_fn, error, request, state, runtime, attempt
                        )

                    if retry_request:
                        request = retry_request
                        break
                else:
                    raise

        # If we exit the loop, max attempts exceeded
        msg = f"Maximum retry attempts ({max_attempts}) exceeded"
        raise RuntimeError(msg)

    # Phase 5: Add nodes to graph
    graph.add_node("model_request", RunnableCallable(model_request, amodel_request, trace=False))
    if components.tool_node is not None:
        graph.add_node("tools", components.tool_node)
    _add_middleware_nodes(graph, components, state_schema)

    # Phase 6: Compute graph topology
    topology = GraphTopology.compute(components.middleware_hooks)

    # Phase 7: Connect edges
    _connect_entry_edges(graph, topology)
    _connect_loop_edges(graph, topology, components)
    _connect_middleware_chains(graph, components, topology)

    return graph


def _resolve_jump(jump_to: JumpTo | None, first_node: str) -> str | None:
    if jump_to == "model":
        return first_node
    if jump_to == "end":
        return "__end__"
    if jump_to == "tools":
        return "tools"
    return None


def _fetch_last_ai_and_tool_messages(
    messages: list[AnyMessage],
) -> tuple[AIMessage, list[ToolMessage]]:
    last_ai_index: int
    last_ai_message: AIMessage

    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            last_ai_index = i
            last_ai_message = cast("AIMessage", messages[i])
            break

    tool_messages = [m for m in messages[last_ai_index + 1 :] if isinstance(m, ToolMessage)]
    return last_ai_message, tool_messages


def _make_model_to_tools_edge(
    first_node: str,
    structured_output_tools: dict[str, OutputToolBinding],
    tool_node: ToolNode,
    exit_node: str,
) -> Callable[[dict[str, Any]], str | list[Send] | None]:
    def model_to_tools(state: dict[str, Any]) -> str | list[Send] | None:
        # 1. if there's an explicit jump_to in the state, use it
        if jump_to := state.get("jump_to"):
            return _resolve_jump(jump_to, first_node)

        last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])
        tool_message_ids = [m.tool_call_id for m in tool_messages]

        # 2. if the model hasn't called any tools, exit the loop
        # this is the classic exit condition for an agent loop
        if len(last_ai_message.tool_calls) == 0:
            return exit_node

        pending_tool_calls = [
            c
            for c in last_ai_message.tool_calls
            if c["id"] not in tool_message_ids and c["name"] not in structured_output_tools
        ]

        # 3. if there are pending tool calls, jump to the tool node
        if pending_tool_calls:
            pending_tool_calls = [
                tool_node.inject_tool_args(call, state, None) for call in pending_tool_calls
            ]
            return [Send("tools", [tool_call]) for tool_call in pending_tool_calls]

        # 4. AIMessage has tool calls, but there are no pending tool calls
        # which suggests the injection of artificial tool messages. jump to the first node
        return first_node

    return model_to_tools


def _make_tools_to_model_edge(
    tool_node: ToolNode,
    next_node: str,
    structured_output_tools: dict[str, OutputToolBinding],
    exit_node: str,
) -> Callable[[dict[str, Any]], str | None]:
    def tools_to_model(state: dict[str, Any]) -> str | None:
        last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])

        if all(
            tool_node.tools_by_name[c["name"]].return_direct
            for c in last_ai_message.tool_calls
            if c["name"] in tool_node.tools_by_name
        ):
            return exit_node

        if any(t.name in structured_output_tools for t in tool_messages):
            return exit_node

        return next_node

    return tools_to_model


def _add_middleware_edge(
    graph: StateGraph[AgentState, ContextT, PublicAgentState, PublicAgentState],
    name: str,
    default_destination: str,
    model_destination: str,
    can_jump_to: list[JumpTo] | None,
) -> None:
    """Add an edge to the graph for a middleware node.

    Args:
        graph: The graph to add the edge to.
        method: The method to call for the middleware node.
        name: The name of the middleware node.
        default_destination: The default destination for the edge.
        model_destination: The destination for the edge to the model.
        can_jump_to: The conditionally jumpable destinations for the edge.
    """
    if can_jump_to:

        def jump_edge(state: dict[str, Any]) -> str:
            return _resolve_jump(state.get("jump_to"), model_destination) or default_destination

        destinations = [default_destination]

        if "end" in can_jump_to:
            destinations.append(END)
        if "tools" in can_jump_to:
            destinations.append("tools")
        if "model" in can_jump_to and name != model_destination:
            destinations.append(model_destination)

        graph.add_conditional_edges(name, jump_edge, destinations)

    else:
        graph.add_edge(name, default_destination)
