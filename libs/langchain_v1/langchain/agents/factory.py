"""Agent factory for creating agents with middleware support."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Annotated, Any, cast, get_args, get_origin, get_type_hints

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph._internal._runnable import RunnableCallable
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.runtime import Runtime  # noqa: TC002
from langgraph.types import Command, Send
from langgraph.typing import ContextT  # noqa: TC002
from typing_extensions import NotRequired, Required, TypedDict, TypeVar

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    JumpTo,
    ModelRequest,
    ModelResponse,
    OmitFromSchema,
    _InputAgentState,
    _OutputAgentState,
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
from langchain.tools.tool_node import ToolCallWithContext, _ToolNode

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langchain_core.runnables import Runnable
    from langgraph.cache.base import BaseCache
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.store.base import BaseStore
    from langgraph.types import Checkpointer

    from langchain.tools.tool_node import ToolCallRequest, ToolCallWrapper

STRUCTURED_OUTPUT_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."

ResponseT = TypeVar("ResponseT")


def _normalize_to_model_response(result: ModelResponse | AIMessage) -> ModelResponse:
    """Normalize middleware return value to ModelResponse."""
    if isinstance(result, AIMessage):
        return ModelResponse(result=[result], structured_response=None)
    return result


def _chain_model_call_handlers(
    handlers: Sequence[
        Callable[
            [ModelRequest, Callable[[ModelRequest], ModelResponse]],
            ModelResponse | AIMessage,
        ]
    ],
) -> (
    Callable[
        [ModelRequest, Callable[[ModelRequest], ModelResponse]],
        ModelResponse,
    ]
    | None
):
    """Compose multiple wrap_model_call handlers into single middleware stack.

    Composes handlers so first in list becomes outermost layer. Each handler
    receives a handler callback to execute inner layers.

    Args:
        handlers: List of handlers. First handler wraps all others.

    Returns:
        Composed handler, or `None` if handlers empty.

    Example:
        ```python
        # handlers=[auth, retry] means: auth wraps retry
        # Flow: auth calls retry, retry calls base handler
        def auth(req, state, runtime, handler):
            try:
                return handler(req)
            except UnauthorizedError:
                refresh_token()
                return handler(req)


        def retry(req, state, runtime, handler):
            for attempt in range(3):
                try:
                    return handler(req)
                except Exception:
                    if attempt == 2:
                        raise


        handler = _chain_model_call_handlers([auth, retry])
        ```
    """
    if not handlers:
        return None

    if len(handlers) == 1:
        # Single handler - wrap to normalize output
        single_handler = handlers[0]

        def normalized_single(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            result = single_handler(request, handler)
            return _normalize_to_model_response(result)

        return normalized_single

    def compose_two(
        outer: Callable[
            [ModelRequest, Callable[[ModelRequest], ModelResponse]],
            ModelResponse | AIMessage,
        ],
        inner: Callable[
            [ModelRequest, Callable[[ModelRequest], ModelResponse]],
            ModelResponse | AIMessage,
        ],
    ) -> Callable[
        [ModelRequest, Callable[[ModelRequest], ModelResponse]],
        ModelResponse,
    ]:
        """Compose two handlers where outer wraps inner."""

        def composed(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            # Create a wrapper that calls inner with the base handler and normalizes
            def inner_handler(req: ModelRequest) -> ModelResponse:
                inner_result = inner(req, handler)
                return _normalize_to_model_response(inner_result)

            # Call outer with the wrapped inner as its handler and normalize
            outer_result = outer(request, inner_handler)
            return _normalize_to_model_response(outer_result)

        return composed

    # Compose right-to-left: outer(inner(innermost(handler)))
    result = handlers[-1]
    for handler in reversed(handlers[:-1]):
        result = compose_two(handler, result)

    # Wrap to ensure final return type is exactly ModelResponse
    def final_normalized(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # result here is typed as returning ModelResponse | AIMessage but compose_two normalizes
        final_result = result(request, handler)
        return _normalize_to_model_response(final_result)

    return final_normalized


def _chain_async_model_call_handlers(
    handlers: Sequence[
        Callable[
            [ModelRequest, Callable[[ModelRequest], Awaitable[ModelResponse]]],
            Awaitable[ModelResponse | AIMessage],
        ]
    ],
) -> (
    Callable[
        [ModelRequest, Callable[[ModelRequest], Awaitable[ModelResponse]]],
        Awaitable[ModelResponse],
    ]
    | None
):
    """Compose multiple async `wrap_model_call` handlers into single middleware stack.

    Args:
        handlers: List of async handlers. First handler wraps all others.

    Returns:
        Composed async handler, or `None` if handlers empty.
    """
    if not handlers:
        return None

    if len(handlers) == 1:
        # Single handler - wrap to normalize output
        single_handler = handlers[0]

        async def normalized_single(
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            result = await single_handler(request, handler)
            return _normalize_to_model_response(result)

        return normalized_single

    def compose_two(
        outer: Callable[
            [ModelRequest, Callable[[ModelRequest], Awaitable[ModelResponse]]],
            Awaitable[ModelResponse | AIMessage],
        ],
        inner: Callable[
            [ModelRequest, Callable[[ModelRequest], Awaitable[ModelResponse]]],
            Awaitable[ModelResponse | AIMessage],
        ],
    ) -> Callable[
        [ModelRequest, Callable[[ModelRequest], Awaitable[ModelResponse]]],
        Awaitable[ModelResponse],
    ]:
        """Compose two async handlers where outer wraps inner."""

        async def composed(
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            # Create a wrapper that calls inner with the base handler and normalizes
            async def inner_handler(req: ModelRequest) -> ModelResponse:
                inner_result = await inner(req, handler)
                return _normalize_to_model_response(inner_result)

            # Call outer with the wrapped inner as its handler and normalize
            outer_result = await outer(request, inner_handler)
            return _normalize_to_model_response(outer_result)

        return composed

    # Compose right-to-left: outer(inner(innermost(handler)))
    result = handlers[-1]
    for handler in reversed(handlers[:-1]):
        result = compose_two(handler, result)

    # Wrap to ensure final return type is exactly ModelResponse
    async def final_normalized(
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        # result here is typed as returning ModelResponse | AIMessage but compose_two normalizes
        final_result = await result(request, handler)
        return _normalize_to_model_response(final_result)

    return final_normalized


def _resolve_schema(schemas: set[type], schema_name: str, omit_flag: str | None = None) -> type:
    """Resolve schema by merging schemas and optionally respecting `OmitFromSchema` annotations.

    Args:
        schemas: List of schema types to merge
        schema_name: Name for the generated `TypedDict`
        omit_flag: If specified, omit fields with this flag set (`'input'` or
            `'output'`)
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


def _get_can_jump_to(middleware: AgentMiddleware[Any, Any], hook_name: str) -> list[JumpTo]:
    """Get the `can_jump_to` list from either sync or async hook methods.

    Args:
        middleware: The middleware instance to inspect.
        hook_name: The name of the hook (`'before_model'` or `'after_model'`).

    Returns:
        List of jump destinations, or empty list if not configured.
    """
    # Get the base class method for comparison
    base_sync_method = getattr(AgentMiddleware, hook_name, None)
    base_async_method = getattr(AgentMiddleware, f"a{hook_name}", None)

    # Try sync method first - only if it's overridden from base class
    sync_method = getattr(middleware.__class__, hook_name, None)
    if (
        sync_method
        and sync_method is not base_sync_method
        and hasattr(sync_method, "__can_jump_to__")
    ):
        return sync_method.__can_jump_to__

    # Try async method - only if it's overridden from base class
    async_method = getattr(middleware.__class__, f"a{hook_name}", None)
    if (
        async_method
        and async_method is not base_async_method
        and hasattr(async_method, "__can_jump_to__")
    ):
        return async_method.__can_jump_to__

    return []


def _supports_provider_strategy(model: str | BaseChatModel) -> bool:
    """Check if a model supports provider-specific structured output.

    Args:
        model: Model name string or `BaseChatModel` instance.

    Returns:
        `True` if the model supports provider-specific structured output, `False` otherwise.
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


def _handle_structured_output_error(
    exception: Exception,
    response_format: ResponseFormat,
) -> tuple[bool, str]:
    """Handle structured output error. Returns `(should_retry, retry_tool_message)`."""
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


def _chain_tool_call_wrappers(
    wrappers: Sequence[ToolCallWrapper],
) -> ToolCallWrapper | None:
    """Compose wrappers into middleware stack (first = outermost).

    Args:
        wrappers: Wrappers in middleware order.

    Returns:
        Composed wrapper, or `None` if empty.

    Example:
        wrapper = _chain_tool_call_wrappers([auth, cache, retry])
        # Request flows: auth -> cache -> retry -> tool
        # Response flows: tool -> retry -> cache -> auth
    """
    if not wrappers:
        return None

    if len(wrappers) == 1:
        return wrappers[0]

    def compose_two(outer: ToolCallWrapper, inner: ToolCallWrapper) -> ToolCallWrapper:
        """Compose two wrappers where outer wraps inner."""

        def composed(
            request: ToolCallRequest,
            execute: Callable[[ToolCallRequest], ToolMessage | Command],
        ) -> ToolMessage | Command:
            # Create a callable that invokes inner with the original execute
            def call_inner(req: ToolCallRequest) -> ToolMessage | Command:
                return inner(req, execute)

            # Outer can call call_inner multiple times
            return outer(request, call_inner)

        return composed

    # Chain all wrappers: first -> second -> ... -> last
    result = wrappers[-1]
    for wrapper in reversed(wrappers[:-1]):
        result = compose_two(wrapper, result)

    return result


def _chain_async_tool_call_wrappers(
    wrappers: Sequence[
        Callable[
            [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]],
            Awaitable[ToolMessage | Command],
        ]
    ],
) -> (
    Callable[
        [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]],
        Awaitable[ToolMessage | Command],
    ]
    | None
):
    """Compose async wrappers into middleware stack (first = outermost).

    Args:
        wrappers: Async wrappers in middleware order.

    Returns:
        Composed async wrapper, or `None` if empty.
    """
    if not wrappers:
        return None

    if len(wrappers) == 1:
        return wrappers[0]

    def compose_two(
        outer: Callable[
            [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]],
            Awaitable[ToolMessage | Command],
        ],
        inner: Callable[
            [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]],
            Awaitable[ToolMessage | Command],
        ],
    ) -> Callable[
        [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]],
        Awaitable[ToolMessage | Command],
    ]:
        """Compose two async wrappers where outer wraps inner."""

        async def composed(
            request: ToolCallRequest,
            execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
        ) -> ToolMessage | Command:
            # Create an async callable that invokes inner with the original execute
            async def call_inner(req: ToolCallRequest) -> ToolMessage | Command:
                return await inner(req, execute)

            # Outer can call call_inner multiple times
            return await outer(request, call_inner)

        return composed

    # Chain all wrappers: first -> second -> ... -> last
    result = wrappers[-1]
    for wrapper in reversed(wrappers[:-1]):
        result = compose_two(wrapper, result)

    return result


def create_agent(  # noqa: PLR0915
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware[AgentState[ResponseT], ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | None = None,
    state_schema: type[AgentState[ResponseT]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph[
    AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]
]:
    """Creates an agent graph that calls tools in a loop until a stopping condition is met.

    For more details on using `create_agent`,
    visit the [Agents](https://docs.langchain.com/oss/python/langchain/agents) docs.

    Args:
        model: The language model for the agent. Can be a string identifier
            (e.g., `"openai:gpt-4"`) or a chat model instance (e.g., `ChatOpenAI()`).
            For a full list of supported model strings, see
            [`init_chat_model`][langchain.chat_models.init_chat_model(model_provider)].
        tools: A list of tools, `dicts`, or `Callable`. If `None` or an empty list,
            the agent will consist of a model node without a tool calling loop.
        system_prompt: An optional system prompt for the LLM. Prompts are converted to a
            `SystemMessage` and added to the beginning of the message list.
        middleware: A sequence of middleware instances to apply to the agent.
            Middleware can intercept and modify agent behavior at various stages.
        response_format: An optional configuration for structured responses.
            Can be a `ToolStrategy`, `ProviderStrategy`, or a Pydantic model class.
            If provided, the agent will handle structured output during the
            conversation flow. Raw schemas will be wrapped in an appropriate strategy
            based on model capabilities.
        state_schema: An optional `TypedDict` schema that extends `AgentState`.
            When provided, this schema is used instead of `AgentState` as the base
            schema for merging with middleware state schemas. This allows users to
            add custom state fields without needing to create custom middleware.
            Generally, it's recommended to use state_schema extensions via middleware
            to keep relevant extensions scoped to corresponding hooks / tools.
            The schema must be a subclass of `AgentState[ResponseT]`.
        context_schema: An optional schema for runtime context.
        checkpointer: An optional checkpoint saver object. This is used for persisting
            the state of the graph (e.g., as chat memory) for a single thread
            (e.g., a single conversation).
        store: An optional store object. This is used for persisting data
            across multiple threads (e.g., multiple conversations / users).
        interrupt_before: An optional list of node names to interrupt before.
            Useful if you want to add a user confirmation or other interrupt
            before taking an action.
        interrupt_after: An optional list of node names to interrupt after.
            Useful if you want to return directly or run additional processing
            on an output.
        debug: Whether to enable verbose logging for graph execution. When enabled,
            prints detailed information about each node execution, state updates,
            and transitions during agent runtime. Useful for debugging middleware
            behavior and understanding agent execution flow.
        name: An optional name for the `CompiledStateGraph`.
            This name will be automatically used when adding the agent graph to
            another graph as a subgraph node - particularly useful for building
            multi-agent systems.
        cache: An optional `BaseCache` instance to enable caching of graph execution.

    Returns:
        A compiled `StateGraph` that can be used for chat interactions.

    The agent node calls the language model with the messages list (after applying
    the system prompt). If the resulting `AIMessage` contains `tool_calls`, the graph
    will then call the tools. The tools node executes the tools and adds the responses
    to the messages list as `ToolMessage` objects. The agent node then calls the
    language model again. The process repeats until no more `tool_calls` are
    present in the response. The agent then returns the full list of messages.

    Example:
        ```python
        from langchain.agents import create_agent


        def check_weather(location: str) -> str:
            '''Return the weather forecast for the specified location.'''
            return f"It's always sunny in {location}"


        graph = create_agent(
            model="anthropic:claude-sonnet-4-5",
            tools=[check_weather],
            system_prompt="You are a helpful assistant",
        )
        inputs = {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
        for chunk in graph.stream(inputs, stream_mode="updates"):
            print(chunk)
        ```
    """
    # init chat model
    if isinstance(model, str):
        model = init_chat_model(model)

    # Handle tools being None or empty
    if tools is None:
        tools = []

    # Convert response format and setup structured output tools
    # Raw schemas are wrapped in AutoStrategy to preserve auto-detection intent.
    # AutoStrategy is converted to ToolStrategy upfront to calculate tools during agent creation,
    # but may be replaced with ProviderStrategy later based on model capabilities.
    initial_response_format: ToolStrategy | ProviderStrategy | AutoStrategy | None
    if response_format is None:
        initial_response_format = None
    elif isinstance(response_format, (ToolStrategy, ProviderStrategy)):
        # Preserve explicitly requested strategies
        initial_response_format = response_format
    elif isinstance(response_format, AutoStrategy):
        # AutoStrategy provided - preserve it for later auto-detection
        initial_response_format = response_format
    else:
        # Raw schema - wrap in AutoStrategy to enable auto-detection
        initial_response_format = AutoStrategy(schema=response_format)

    # For AutoStrategy, convert to ToolStrategy to setup tools upfront
    # (may be replaced with ProviderStrategy later based on model)
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

    # Collect middleware with wrap_tool_call or awrap_tool_call hooks
    # Include middleware with either implementation to ensure NotImplementedError is raised
    # when middleware doesn't support the execution path
    middleware_w_wrap_tool_call = [
        m
        for m in middleware
        if m.__class__.wrap_tool_call is not AgentMiddleware.wrap_tool_call
        or m.__class__.awrap_tool_call is not AgentMiddleware.awrap_tool_call
    ]

    # Chain all wrap_tool_call handlers into a single composed handler
    wrap_tool_call_wrapper = None
    if middleware_w_wrap_tool_call:
        wrappers = [m.wrap_tool_call for m in middleware_w_wrap_tool_call]
        wrap_tool_call_wrapper = _chain_tool_call_wrappers(wrappers)

    # Collect middleware with awrap_tool_call or wrap_tool_call hooks
    # Include middleware with either implementation to ensure NotImplementedError is raised
    # when middleware doesn't support the execution path
    middleware_w_awrap_tool_call = [
        m
        for m in middleware
        if m.__class__.awrap_tool_call is not AgentMiddleware.awrap_tool_call
        or m.__class__.wrap_tool_call is not AgentMiddleware.wrap_tool_call
    ]

    # Chain all awrap_tool_call handlers into a single composed async handler
    awrap_tool_call_wrapper = None
    if middleware_w_awrap_tool_call:
        async_wrappers = [m.awrap_tool_call for m in middleware_w_awrap_tool_call]
        awrap_tool_call_wrapper = _chain_async_tool_call_wrappers(async_wrappers)

    # Setup tools
    tool_node: _ToolNode | None = None
    # Extract built-in provider tools (dict format) and regular tools (BaseTool/callables)
    built_in_tools = [t for t in tools if isinstance(t, dict)]
    regular_tools = [t for t in tools if not isinstance(t, dict)]

    # Tools that require client-side execution (must be in ToolNode)
    available_tools = middleware_tools + regular_tools

    # Only create ToolNode if we have client-side tools
    tool_node = (
        _ToolNode(
            tools=available_tools,
            wrap_tool_call=wrap_tool_call_wrapper,
            awrap_tool_call=awrap_tool_call_wrapper,
        )
        if available_tools
        else None
    )

    # Default tools for ModelRequest initialization
    # Use converted BaseTool instances from ToolNode (not raw callables)
    # Include built-ins and converted tools (can be changed dynamically by middleware)
    # Structured tools are NOT included - they're added dynamically based on response_format
    if tool_node:
        default_tools = list(tool_node.tools_by_name.values()) + built_in_tools
    else:
        default_tools = list(built_in_tools)

    # validate middleware
    assert len({m.name for m in middleware}) == len(middleware), (  # noqa: S101
        "Please remove duplicate middleware instances."
    )
    middleware_w_before_agent = [
        m
        for m in middleware
        if m.__class__.before_agent is not AgentMiddleware.before_agent
        or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
    ]
    middleware_w_before_model = [
        m
        for m in middleware
        if m.__class__.before_model is not AgentMiddleware.before_model
        or m.__class__.abefore_model is not AgentMiddleware.abefore_model
    ]
    middleware_w_after_model = [
        m
        for m in middleware
        if m.__class__.after_model is not AgentMiddleware.after_model
        or m.__class__.aafter_model is not AgentMiddleware.aafter_model
    ]
    middleware_w_after_agent = [
        m
        for m in middleware
        if m.__class__.after_agent is not AgentMiddleware.after_agent
        or m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
    ]
    # Collect middleware with wrap_model_call or awrap_model_call hooks
    # Include middleware with either implementation to ensure NotImplementedError is raised
    # when middleware doesn't support the execution path
    middleware_w_wrap_model_call = [
        m
        for m in middleware
        if m.__class__.wrap_model_call is not AgentMiddleware.wrap_model_call
        or m.__class__.awrap_model_call is not AgentMiddleware.awrap_model_call
    ]
    # Collect middleware with awrap_model_call or wrap_model_call hooks
    # Include middleware with either implementation to ensure NotImplementedError is raised
    # when middleware doesn't support the execution path
    middleware_w_awrap_model_call = [
        m
        for m in middleware
        if m.__class__.awrap_model_call is not AgentMiddleware.awrap_model_call
        or m.__class__.wrap_model_call is not AgentMiddleware.wrap_model_call
    ]

    # Compose wrap_model_call handlers into a single middleware stack (sync)
    wrap_model_call_handler = None
    if middleware_w_wrap_model_call:
        sync_handlers = [m.wrap_model_call for m in middleware_w_wrap_model_call]
        wrap_model_call_handler = _chain_model_call_handlers(sync_handlers)

    # Compose awrap_model_call handlers into a single middleware stack (async)
    awrap_model_call_handler = None
    if middleware_w_awrap_model_call:
        async_handlers = [m.awrap_model_call for m in middleware_w_awrap_model_call]
        awrap_model_call_handler = _chain_async_model_call_handlers(async_handlers)

    state_schemas = {m.state_schema for m in middleware}
    # Use provided state_schema if available, otherwise use base AgentState
    base_state = state_schema if state_schema is not None else AgentState
    state_schemas.add(base_state)

    resolved_state_schema = _resolve_schema(state_schemas, "StateSchema", None)
    input_schema = _resolve_schema(state_schemas, "InputSchema", "input")
    output_schema = _resolve_schema(state_schemas, "OutputSchema", "output")

    # create graph, add nodes
    graph: StateGraph[
        AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]
    ] = StateGraph(
        state_schema=resolved_state_schema,
        input_schema=input_schema,
        output_schema=output_schema,
        context_schema=context_schema,
    )

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
            Tuple of `(bound_model, effective_response_format)` where
            `effective_response_format` is the actual strategy used (may differ from
            initial if auto-detected).
        """
        # Validate ONLY client-side tools that need to exist in tool_node
        # Build map of available client-side tools from the ToolNode
        # (which has already converted callables)
        available_tools_by_name = {}
        if tool_node:
            available_tools_by_name = tool_node.tools_by_name.copy()

        # Check if any requested tools are unknown CLIENT-SIDE tools
        unknown_tool_names = []
        for t in request.tools:
            # Only validate BaseTool instances (skip built-in dict tools)
            if isinstance(t, dict):
                continue
            if isinstance(t, BaseTool) and t.name not in available_tools_by_name:
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
        # request.tools now only contains BaseTool instances (converted from callables)
        # and dicts (built-ins)
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

    def _execute_model_sync(request: ModelRequest) -> ModelResponse:
        """Execute model and return response.

        This is the core model execution logic wrapped by `wrap_model_call` handlers.
        Raises any exceptions that occur during model invocation.
        """
        # Get the bound model (with auto-detection if needed)
        model_, effective_response_format = _get_bound_model(request)
        messages = request.messages
        if request.system_prompt:
            messages = [SystemMessage(request.system_prompt), *messages]

        output = model_.invoke(messages)

        # Handle model output to get messages and structured_response
        handled_output = _handle_model_output(output, effective_response_format)
        messages_list = handled_output["messages"]
        structured_response = handled_output.get("structured_response")

        return ModelResponse(
            result=messages_list,
            structured_response=structured_response,
        )

    def model_node(state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any]:
        """Sync model request handler with sequential middleware processing."""
        request = ModelRequest(
            model=model,
            tools=default_tools,
            system_prompt=system_prompt,
            response_format=initial_response_format,
            messages=state["messages"],
            tool_choice=None,
            state=state,
            runtime=runtime,
        )

        if wrap_model_call_handler is None:
            # No handlers - execute directly
            response = _execute_model_sync(request)
        else:
            # Call composed handler with base handler
            response = wrap_model_call_handler(request, _execute_model_sync)

        # Extract state updates from ModelResponse
        state_updates = {"messages": response.result}
        if response.structured_response is not None:
            state_updates["structured_response"] = response.structured_response

        return state_updates

    async def _execute_model_async(request: ModelRequest) -> ModelResponse:
        """Execute model asynchronously and return response.

        This is the core async model execution logic wrapped by `wrap_model_call`
        handlers.

        Raises any exceptions that occur during model invocation.
        """
        # Get the bound model (with auto-detection if needed)
        model_, effective_response_format = _get_bound_model(request)
        messages = request.messages
        if request.system_prompt:
            messages = [SystemMessage(request.system_prompt), *messages]

        output = await model_.ainvoke(messages)

        # Handle model output to get messages and structured_response
        handled_output = _handle_model_output(output, effective_response_format)
        messages_list = handled_output["messages"]
        structured_response = handled_output.get("structured_response")

        return ModelResponse(
            result=messages_list,
            structured_response=structured_response,
        )

    async def amodel_node(state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any]:
        """Async model request handler with sequential middleware processing."""
        request = ModelRequest(
            model=model,
            tools=default_tools,
            system_prompt=system_prompt,
            response_format=initial_response_format,
            messages=state["messages"],
            tool_choice=None,
            state=state,
            runtime=runtime,
        )

        if awrap_model_call_handler is None:
            # No async handlers - execute directly
            response = await _execute_model_async(request)
        else:
            # Call composed async handler with base handler
            response = await awrap_model_call_handler(request, _execute_model_async)

        # Extract state updates from ModelResponse
        state_updates = {"messages": response.result}
        if response.structured_response is not None:
            state_updates["structured_response"] = response.structured_response

        return state_updates

    # Use sync or async based on model capabilities
    graph.add_node("model", RunnableCallable(model_node, amodel_node, trace=False))

    # Only add tools node if we have tools
    if tool_node is not None:
        graph.add_node("tools", tool_node)

    # Add middleware nodes
    for m in middleware:
        if (
            m.__class__.before_agent is not AgentMiddleware.before_agent
            or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
        ):
            # Use RunnableCallable to support both sync and async
            # Pass None for sync if not overridden to avoid signature conflicts
            sync_before_agent = (
                m.before_agent
                if m.__class__.before_agent is not AgentMiddleware.before_agent
                else None
            )
            async_before_agent = (
                m.abefore_agent
                if m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
                else None
            )
            before_agent_node = RunnableCallable(sync_before_agent, async_before_agent, trace=False)
            graph.add_node(
                f"{m.name}.before_agent", before_agent_node, input_schema=resolved_state_schema
            )

        if (
            m.__class__.before_model is not AgentMiddleware.before_model
            or m.__class__.abefore_model is not AgentMiddleware.abefore_model
        ):
            # Use RunnableCallable to support both sync and async
            # Pass None for sync if not overridden to avoid signature conflicts
            sync_before = (
                m.before_model
                if m.__class__.before_model is not AgentMiddleware.before_model
                else None
            )
            async_before = (
                m.abefore_model
                if m.__class__.abefore_model is not AgentMiddleware.abefore_model
                else None
            )
            before_node = RunnableCallable(sync_before, async_before, trace=False)
            graph.add_node(
                f"{m.name}.before_model", before_node, input_schema=resolved_state_schema
            )

        if (
            m.__class__.after_model is not AgentMiddleware.after_model
            or m.__class__.aafter_model is not AgentMiddleware.aafter_model
        ):
            # Use RunnableCallable to support both sync and async
            # Pass None for sync if not overridden to avoid signature conflicts
            sync_after = (
                m.after_model
                if m.__class__.after_model is not AgentMiddleware.after_model
                else None
            )
            async_after = (
                m.aafter_model
                if m.__class__.aafter_model is not AgentMiddleware.aafter_model
                else None
            )
            after_node = RunnableCallable(sync_after, async_after, trace=False)
            graph.add_node(f"{m.name}.after_model", after_node, input_schema=resolved_state_schema)

        if (
            m.__class__.after_agent is not AgentMiddleware.after_agent
            or m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
        ):
            # Use RunnableCallable to support both sync and async
            # Pass None for sync if not overridden to avoid signature conflicts
            sync_after_agent = (
                m.after_agent
                if m.__class__.after_agent is not AgentMiddleware.after_agent
                else None
            )
            async_after_agent = (
                m.aafter_agent
                if m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
                else None
            )
            after_agent_node = RunnableCallable(sync_after_agent, async_after_agent, trace=False)
            graph.add_node(
                f"{m.name}.after_agent", after_agent_node, input_schema=resolved_state_schema
            )

    # Determine the entry node (runs once at start): before_agent -> before_model -> model
    if middleware_w_before_agent:
        entry_node = f"{middleware_w_before_agent[0].name}.before_agent"
    elif middleware_w_before_model:
        entry_node = f"{middleware_w_before_model[0].name}.before_model"
    else:
        entry_node = "model"

    # Determine the loop entry node (beginning of agent loop, excludes before_agent)
    # This is where tools will loop back to for the next iteration
    if middleware_w_before_model:
        loop_entry_node = f"{middleware_w_before_model[0].name}.before_model"
    else:
        loop_entry_node = "model"

    # Determine the loop exit node (end of each iteration, can run multiple times)
    # This is after_model or model, but NOT after_agent
    if middleware_w_after_model:
        loop_exit_node = f"{middleware_w_after_model[0].name}.after_model"
    else:
        loop_exit_node = "model"

    # Determine the exit node (runs once at end): after_agent or END
    if middleware_w_after_agent:
        exit_node = f"{middleware_w_after_agent[-1].name}.after_agent"
    else:
        exit_node = END

    graph.add_edge(START, entry_node)
    # add conditional edges only if tools exist
    if tool_node is not None:
        # Only include exit_node in destinations if any tool has return_direct=True
        # or if there are structured output tools
        tools_to_model_destinations = [loop_entry_node]
        if (
            any(tool.return_direct for tool in tool_node.tools_by_name.values())
            or structured_output_tools
        ):
            tools_to_model_destinations.append(exit_node)

        graph.add_conditional_edges(
            "tools",
            _make_tools_to_model_edge(
                tool_node=tool_node,
                model_destination=loop_entry_node,
                structured_output_tools=structured_output_tools,
                end_destination=exit_node,
            ),
            tools_to_model_destinations,
        )

        # base destinations are tools and exit_node
        # we add the loop_entry node to edge destinations if:
        # - there is an after model hook(s) -- allows jump_to to model
        #   potentially artificially injected tool messages, ex HITL
        # - there is a response format -- to allow for jumping to model to handle
        #   regenerating structured output tool calls
        model_to_tools_destinations = ["tools", exit_node]
        if response_format or loop_exit_node != "model":
            model_to_tools_destinations.append(loop_entry_node)

        graph.add_conditional_edges(
            loop_exit_node,
            _make_model_to_tools_edge(
                model_destination=loop_entry_node,
                structured_output_tools=structured_output_tools,
                end_destination=exit_node,
            ),
            model_to_tools_destinations,
        )
    elif len(structured_output_tools) > 0:
        graph.add_conditional_edges(
            loop_exit_node,
            _make_model_to_model_edge(
                model_destination=loop_entry_node,
                end_destination=exit_node,
            ),
            [loop_entry_node, exit_node],
        )
    elif loop_exit_node == "model":
        # If no tools and no after_model, go directly to exit_node
        graph.add_edge(loop_exit_node, exit_node)
    # No tools but we have after_model - connect after_model to exit_node
    else:
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_after_model[0].name}.after_model",
            default_destination=exit_node,
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_after_model[0], "after_model"),
        )

    # Add before_agent middleware edges
    if middleware_w_before_agent:
        for m1, m2 in itertools.pairwise(middleware_w_before_agent):
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.before_agent",
                default_destination=f"{m2.name}.before_agent",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "before_agent"),
            )
        # Connect last before_agent to loop_entry_node (before_model or model)
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_before_agent[-1].name}.before_agent",
            default_destination=loop_entry_node,
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_before_agent[-1], "before_agent"),
        )

    # Add before_model middleware edges
    if middleware_w_before_model:
        for m1, m2 in itertools.pairwise(middleware_w_before_model):
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.before_model",
                default_destination=f"{m2.name}.before_model",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "before_model"),
            )
        # Go directly to model after the last before_model
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_before_model[-1].name}.before_model",
            default_destination="model",
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_before_model[-1], "before_model"),
        )

    # Add after_model middleware edges
    if middleware_w_after_model:
        graph.add_edge("model", f"{middleware_w_after_model[-1].name}.after_model")
        for idx in range(len(middleware_w_after_model) - 1, 0, -1):
            m1 = middleware_w_after_model[idx]
            m2 = middleware_w_after_model[idx - 1]
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.after_model",
                default_destination=f"{m2.name}.after_model",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "after_model"),
            )
        # Note: Connection from after_model to after_agent/END is handled above
        # in the conditional edges section

    # Add after_agent middleware edges
    if middleware_w_after_agent:
        # Chain after_agent middleware (runs once at the very end, before END)
        for idx in range(len(middleware_w_after_agent) - 1, 0, -1):
            m1 = middleware_w_after_agent[idx]
            m2 = middleware_w_after_agent[idx - 1]
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.after_agent",
                default_destination=f"{m2.name}.after_agent",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "after_agent"),
            )

        # Connect the last after_agent to END
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_after_agent[0].name}.after_agent",
            default_destination=END,
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_after_agent[0], "after_agent"),
        )

    return graph.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        name=name,
        cache=cache,
    )


def _resolve_jump(
    jump_to: JumpTo | None,
    *,
    model_destination: str,
    end_destination: str,
) -> str | None:
    if jump_to == "model":
        return model_destination
    if jump_to == "end":
        return end_destination
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
    *,
    model_destination: str,
    structured_output_tools: dict[str, OutputToolBinding],
    end_destination: str,
) -> Callable[[dict[str, Any]], str | list[Send] | None]:
    def model_to_tools(
        state: dict[str, Any],
    ) -> str | list[Send] | None:
        # 1. if there's an explicit jump_to in the state, use it
        if jump_to := state.get("jump_to"):
            return _resolve_jump(
                jump_to,
                model_destination=model_destination,
                end_destination=end_destination,
            )

        last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])
        tool_message_ids = [m.tool_call_id for m in tool_messages]

        # 2. if the model hasn't called any tools, exit the loop
        # this is the classic exit condition for an agent loop
        if len(last_ai_message.tool_calls) == 0:
            return end_destination

        pending_tool_calls = [
            c
            for c in last_ai_message.tool_calls
            if c["id"] not in tool_message_ids and c["name"] not in structured_output_tools
        ]

        # 3. if there are pending tool calls, jump to the tool node
        if pending_tool_calls:
            return [
                Send(
                    "tools",
                    ToolCallWithContext(
                        __type="tool_call_with_context",
                        tool_call=tool_call,
                        state=state,
                    ),
                )
                for tool_call in pending_tool_calls
            ]

        # 4. if there is a structured response, exit the loop
        if "structured_response" in state:
            return end_destination

        # 5. AIMessage has tool calls, but there are no pending tool calls
        # which suggests the injection of artificial tool messages. jump to the model node
        return model_destination

    return model_to_tools


def _make_model_to_model_edge(
    *,
    model_destination: str,
    end_destination: str,
) -> Callable[[dict[str, Any]], str | list[Send] | None]:
    def model_to_model(
        state: dict[str, Any],
    ) -> str | list[Send] | None:
        # 1. Priority: Check for explicit jump_to directive from middleware
        if jump_to := state.get("jump_to"):
            return _resolve_jump(
                jump_to,
                model_destination=model_destination,
                end_destination=end_destination,
            )

        # 2. Exit condition: A structured response was generated
        if "structured_response" in state:
            return end_destination

        # 3. Default: Continue the loop, there may have been an issue
        #     with structured output generation, so we need to retry
        return model_destination

    return model_to_model


def _make_tools_to_model_edge(
    *,
    tool_node: _ToolNode,
    model_destination: str,
    structured_output_tools: dict[str, OutputToolBinding],
    end_destination: str,
) -> Callable[[dict[str, Any]], str | None]:
    def tools_to_model(state: dict[str, Any]) -> str | None:
        last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])

        # 1. Exit condition: All executed tools have return_direct=True
        # Filter to only client-side tools (provider tools are not in tool_node)
        client_side_tool_calls = [
            c for c in last_ai_message.tool_calls if c["name"] in tool_node.tools_by_name
        ]
        if client_side_tool_calls and all(
            tool_node.tools_by_name[c["name"]].return_direct for c in client_side_tool_calls
        ):
            return end_destination

        # 2. Exit condition: A structured output tool was executed
        if any(t.name in structured_output_tools for t in tool_messages):
            return end_destination

        # 3. Default: Continue the loop
        #    Tool execution completed successfully, route back to the model
        #    so it can process the tool results and decide the next action.
        return model_destination

    return tools_to_model


def _add_middleware_edge(
    graph: StateGraph[
        AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]
    ],
    *,
    name: str,
    default_destination: str,
    model_destination: str,
    end_destination: str,
    can_jump_to: list[JumpTo] | None,
) -> None:
    """Add an edge to the graph for a middleware node.

    Args:
        graph: The graph to add the edge to.
        name: The name of the middleware node.
        default_destination: The default destination for the edge.
        model_destination: The destination for the edge to the model.
        end_destination: The destination for the edge to the end.
        can_jump_to: The conditionally jumpable destinations for the edge.
    """
    if can_jump_to:

        def jump_edge(state: dict[str, Any]) -> str:
            return (
                _resolve_jump(
                    state.get("jump_to"),
                    model_destination=model_destination,
                    end_destination=end_destination,
                )
                or default_destination
            )

        destinations = [default_destination]

        if "end" in can_jump_to:
            destinations.append(end_destination)
        if "tools" in can_jump_to:
            destinations.append("tools")
        if "model" in can_jump_to and name != model_destination:
            destinations.append(model_destination)

        graph.add_conditional_edges(name, jump_edge, destinations)

    else:
        graph.add_edge(name, default_destination)


__all__ = [
    "create_agent",
]
