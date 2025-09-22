"""Middleware agent implementation."""

import itertools
from collections.abc import Callable, Sequence
from inspect import signature
from typing import Annotated, Any, cast, get_args, get_origin, get_type_hints

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
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
    ModelRequest,
    OmitFromSchema,
    PublicAgentState,
)
from langchain.agents.structured_output import (
    MultipleStructuredOutputsError,
    OutputToolBinding,
    ProviderStrategy,
    ProviderStrategyBinding,
    ResponseFormat,
    StructuredOutputValidationError,
    ToolStrategy,
)
from langchain.agents.tool_node import ToolNode
from langchain.chat_models import init_chat_model

STRUCTURED_OUTPUT_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."


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


def _supports_native_structured_output(model: str | BaseChatModel) -> bool:
    """Check if a model supports native structured output."""
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


ResponseT = TypeVar("ResponseT")


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
    """Create a middleware agent graph."""
    # init chat model
    if isinstance(model, str):
        model = init_chat_model(model)

    # Handle tools being None or empty
    if tools is None:
        tools = []

    # Setup structured output
    structured_output_tools: dict[str, OutputToolBinding] = {}
    native_output_binding: ProviderStrategyBinding | None = None

    if response_format is not None:
        if not isinstance(response_format, (ToolStrategy, ProviderStrategy)):
            # Auto-detect strategy based on model capabilities
            if _supports_native_structured_output(model):
                response_format = ProviderStrategy(schema=response_format)
            else:
                response_format = ToolStrategy(schema=response_format)

        if isinstance(response_format, ToolStrategy):
            # Setup tools strategy for structured output
            for response_schema in response_format.schema_specs:
                structured_tool_info = OutputToolBinding.from_schema_spec(response_schema)
                structured_output_tools[structured_tool_info.tool.name] = structured_tool_info
        elif isinstance(response_format, ProviderStrategy):
            # Setup native strategy
            native_output_binding = ProviderStrategyBinding.from_schema_spec(
                response_format.schema_spec
            )
    middleware_tools = [t for m in middleware for t in getattr(m, "tools", [])]

    # Setup tools
    tool_node: ToolNode | None = None
    if isinstance(tools, list):
        # Extract builtin provider tools (dict format)
        builtin_tools = [t for t in tools if isinstance(t, dict)]
        regular_tools = [t for t in tools if not isinstance(t, dict)]

        # Add structured output tools to regular tools
        structured_tools = [info.tool for info in structured_output_tools.values()]
        all_tools = middleware_tools + regular_tools + structured_tools

        # Only create ToolNode if we have tools
        tool_node = ToolNode(tools=all_tools) if all_tools else None
        default_tools = regular_tools + builtin_tools + structured_tools + middleware_tools
    elif isinstance(tools, ToolNode):
        # tools is ToolNode or None
        tool_node = tools
        if tool_node:
            default_tools = list(tool_node.tools_by_name.values()) + middleware_tools
            # Update tool node to know about tools provided by middleware
            all_tools = list(tool_node.tools_by_name.values()) + middleware_tools
            tool_node = ToolNode(all_tools)
            # Add structured output tools
            for info in structured_output_tools.values():
                default_tools.append(info.tool)
    else:
        default_tools = (
            list(structured_output_tools.values()) if structured_output_tools else []
        ) + middleware_tools

    # validate middleware
    assert len({m.__class__.__name__ for m in middleware}) == len(middleware), (  # noqa: S101
        "Please remove duplicate middleware instances."
    )
    middleware_w_before = [
        m for m in middleware if m.__class__.before_model is not AgentMiddleware.before_model
    ]
    middleware_w_modify_model_request = [
        m
        for m in middleware
        if m.__class__.modify_model_request is not AgentMiddleware.modify_model_request
    ]
    middleware_w_after = [
        m for m in middleware if m.__class__.after_model is not AgentMiddleware.after_model
    ]

    state_schemas = {m.state_schema for m in middleware}
    state_schemas.add(AgentState)

    state_schema = _resolve_schema(state_schemas, "StateSchema", None)
    input_schema = _resolve_schema(state_schemas, "InputSchema", "input")
    output_schema = _resolve_schema(state_schemas, "OutputSchema", "output")

    # create graph, add nodes
    graph: StateGraph[
        AgentState[ResponseT], ContextT, PublicAgentState[ResponseT], PublicAgentState[ResponseT]
    ] = StateGraph(
        state_schema=state_schema,
        input_schema=input_schema,
        output_schema=output_schema,
        context_schema=context_schema,
    )

    def _handle_model_output(output: AIMessage) -> dict[str, Any]:
        """Handle model output including structured responses."""
        # Handle structured output with native strategy
        if isinstance(response_format, ProviderStrategy):
            if not output.tool_calls and native_output_binding:
                structured_response = native_output_binding.parse(output)
                return {"messages": [output], "response": structured_response}
            return {"messages": [output]}

        # Handle structured output with tools strategy
        if (
            isinstance(response_format, ToolStrategy)
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
                        exception, response_format
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
                        response_format.tool_message_content
                        if response_format.tool_message_content
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
                        "response": structured_response,
                    }
                except Exception as exc:  # noqa: BLE001
                    exception = StructuredOutputValidationError(tool_call["name"], exc)
                    should_retry, error_message = _handle_structured_output_error(
                        exception, response_format
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

    def _get_bound_model(request: ModelRequest) -> Runnable:
        """Get the model with appropriate tool bindings."""
        if isinstance(response_format, ProviderStrategy):
            # Use native structured output
            kwargs = response_format.to_model_kwargs()
            return request.model.bind_tools(
                request.tools, strict=True, **kwargs, **request.model_settings
            )
        if isinstance(response_format, ToolStrategy):
            tool_choice = "any" if structured_output_tools else request.tool_choice
            return request.model.bind_tools(
                request.tools, tool_choice=tool_choice, **request.model_settings
            )
        # Standard model binding
        if request.tools:
            return request.model.bind_tools(
                request.tools, tool_choice=request.tool_choice, **request.model_settings
            )
        return request.model.bind(**request.model_settings)

    model_request_signatures: list[
        tuple[bool, AgentMiddleware[AgentState[ResponseT], ContextT]]
    ] = [
        ("runtime" in signature(m.modify_model_request).parameters, m)
        for m in middleware_w_modify_model_request
    ]

    def model_request(state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any]:
        """Sync model request handler with sequential middleware processing."""
        request = ModelRequest(
            model=model,
            tools=default_tools,
            system_prompt=system_prompt,
            response_format=response_format,
            messages=state["messages"],
            tool_choice=None,
        )

        # Apply modify_model_request middleware in sequence
        for use_runtime, m in model_request_signatures:
            if use_runtime:
                m.modify_model_request(request, state, runtime)
            else:
                m.modify_model_request(request, state)  # type: ignore[call-arg]

        # Get the final model and messages
        model_ = _get_bound_model(request)
        messages = request.messages
        if request.system_prompt:
            messages = [SystemMessage(request.system_prompt), *messages]

        output = model_.invoke(messages)
        return _handle_model_output(output)

    async def amodel_request(state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any]:
        """Async model request handler with sequential middleware processing."""
        # Start with the base model request
        request = ModelRequest(
            model=model,
            tools=default_tools,
            system_prompt=system_prompt,
            response_format=response_format,
            messages=state["messages"],
            tool_choice=None,
        )

        # Apply modify_model_request middleware in sequence
        for use_runtime, m in model_request_signatures:
            if use_runtime:
                m.modify_model_request(request, state, runtime)
            else:
                m.modify_model_request(request, state)  # type: ignore[call-arg]

        # Get the final model and messages
        model_ = _get_bound_model(request)
        messages = request.messages
        if request.system_prompt:
            messages = [SystemMessage(request.system_prompt), *messages]

        output = await model_.ainvoke(messages)
        return _handle_model_output(output)

    # Use sync or async based on model capabilities
    from langgraph._internal._runnable import RunnableCallable

    graph.add_node("model_request", RunnableCallable(model_request, amodel_request))

    # Only add tools node if we have tools
    if tool_node is not None:
        graph.add_node("tools", tool_node)

    # Add middleware nodes
    for m in middleware:
        if m.__class__.before_model is not AgentMiddleware.before_model:
            graph.add_node(
                f"{m.__class__.__name__}.before_model", m.before_model, input_schema=state_schema
            )

        if m.__class__.after_model is not AgentMiddleware.after_model:
            graph.add_node(
                f"{m.__class__.__name__}.after_model", m.after_model, input_schema=state_schema
            )

    # add start edge
    first_node = (
        f"{middleware_w_before[0].__class__.__name__}.before_model"
        if middleware_w_before
        else "model_request"
    )
    last_node = (
        f"{middleware_w_after[0].__class__.__name__}.after_model"
        if middleware_w_after
        else "model_request"
    )
    graph.add_edge(START, first_node)

    # add conditional edges only if tools exist
    if tool_node is not None:
        graph.add_conditional_edges(
            "tools",
            _make_tools_to_model_edge(tool_node, first_node, structured_output_tools),
            [first_node, END],
        )
        graph.add_conditional_edges(
            last_node,
            _make_model_to_tools_edge(first_node, structured_output_tools, tool_node),
            [first_node, "tools", END],
        )
    elif last_node == "model_request":
        # If no tools, just go to END from model
        graph.add_edge(last_node, END)
    else:
        # If after_model, then need to check for jump_to
        _add_middleware_edge(
            graph,
            f"{middleware_w_after[0].__class__.__name__}.after_model",
            END,
            first_node,
            tools_available=tool_node is not None,
        )

    # Add middleware edges (same as before)
    if middleware_w_before:
        for m1, m2 in itertools.pairwise(middleware_w_before):
            _add_middleware_edge(
                graph,
                f"{m1.__class__.__name__}.before_model",
                f"{m2.__class__.__name__}.before_model",
                first_node,
                tools_available=tool_node is not None,
            )
        # Go directly to model_request after the last before_model
        _add_middleware_edge(
            graph,
            f"{middleware_w_before[-1].__class__.__name__}.before_model",
            "model_request",
            first_node,
            tools_available=tool_node is not None,
        )

    if middleware_w_after:
        graph.add_edge("model_request", f"{middleware_w_after[-1].__class__.__name__}.after_model")
        for idx in range(len(middleware_w_after) - 1, 0, -1):
            m1 = middleware_w_after[idx]
            m2 = middleware_w_after[idx - 1]
            _add_middleware_edge(
                graph,
                f"{m1.__class__.__name__}.after_model",
                f"{m2.__class__.__name__}.after_model",
                first_node,
                tools_available=tool_node is not None,
            )

    return graph


def _resolve_jump(jump_to: JumpTo | None, first_node: str) -> str | None:
    if jump_to == "model":
        return first_node
    if jump_to:
        return jump_to
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
    first_node: str, structured_output_tools: dict[str, OutputToolBinding], tool_node: ToolNode
) -> Callable[[AgentState], str | list[Send] | None]:
    def model_to_tools(state: AgentState) -> str | list[Send] | None:
        if jump_to := state.get("jump_to"):
            return _resolve_jump(jump_to, first_node)

        last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])
        tool_message_ids = [m.tool_call_id for m in tool_messages]

        pending_tool_calls = [
            c
            for c in last_ai_message.tool_calls
            if c["id"] not in tool_message_ids and c["name"] not in structured_output_tools
        ]

        if pending_tool_calls:
            # imo we should not be injecting state, store here,
            # this should be done by the tool node itself ideally but this is a consequence
            # of using Send w/ tool calls directly which allows more intuitive interrupt behavior
            # largely internal so can be fixed later
            pending_tool_calls = [
                tool_node.inject_tool_args(call, state, None)  # type: ignore[arg-type]
                for call in pending_tool_calls
            ]
            return [Send("tools", [tool_call]) for tool_call in pending_tool_calls]

        return END

    return model_to_tools


def _make_tools_to_model_edge(
    tool_node: ToolNode, next_node: str, structured_output_tools: dict[str, OutputToolBinding]
) -> Callable[[AgentState], str | None]:
    def tools_to_model(state: AgentState) -> str | None:
        last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])

        if all(
            tool_node.tools_by_name[c["name"]].return_direct
            for c in last_ai_message.tool_calls
            if c["name"] in tool_node.tools_by_name
        ):
            return END

        if any(t.name in structured_output_tools for t in tool_messages):
            return END

        return next_node

    return tools_to_model


def _add_middleware_edge(
    graph: StateGraph[AgentState, ContextT, PublicAgentState, PublicAgentState],
    name: str,
    default_destination: str,
    model_destination: str,
    tools_available: bool,  # noqa: FBT001
) -> None:
    """Add an edge to the graph for a middleware node.

    Args:
        graph: The graph to add the edge to.
        method: The method to call for the middleware node.
        name: The name of the middleware node.
        default_destination: The default destination for the edge.
        model_destination: The destination for the edge to the model.
        tools_available: Whether tools are available for the edge to potentially route to.
    """

    def jump_edge(state: AgentState) -> str:
        return _resolve_jump(state.get("jump_to"), model_destination) or default_destination

    destinations = [default_destination]
    if default_destination != END:
        destinations.append(END)
    if tools_available:
        destinations.append("tools")
    if name != model_destination:
        destinations.append(model_destination)

    graph.add_conditional_edges(name, jump_edge, destinations)
