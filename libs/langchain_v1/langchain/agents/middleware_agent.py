"""Middleware agent implementation."""

import itertools
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.typing import ContextT
from typing_extensions import TypedDict, TypeVar

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    JumpTo,
    ModelRequest,
    PublicAgentState,
)

# Import structured output classes from the old implementation
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


def _merge_state_schemas(schemas: list[type]) -> type:
    """Merge multiple TypedDict schemas into a single schema with all fields."""
    if not schemas:
        return AgentState

    all_annotations = {}

    for schema in schemas:
        all_annotations.update(schema.__annotations__)

    return TypedDict("MergedState", all_annotations)  # type: ignore[operator]


def _filter_state_for_schema(state: dict[str, Any], schema: type) -> dict[str, Any]:
    """Filter state to only include fields defined in the given schema."""
    if not hasattr(schema, "__annotations__"):
        return state

    schema_fields = set(schema.__annotations__.keys())
    return {k: v for k, v in state.items() if k in schema_fields}


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
    middleware: Sequence[AgentMiddleware] = (),
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

    # Collect all middleware state schemas and create merged schema
    merged_state_schema: type[AgentState] = _merge_state_schemas(
        [m.state_schema for m in middleware]
    )

    # create graph, add nodes
    graph = StateGraph(
        merged_state_schema,
        input_schema=PublicAgentState,
        output_schema=PublicAgentState,
        context_schema=context_schema,
    )

    def _prepare_model_request(state: dict[str, Any]) -> tuple[ModelRequest, list[AnyMessage]]:
        """Prepare model request and messages."""
        request = state.get("model_request") or ModelRequest(
            model=model,
            tools=default_tools,
            system_prompt=system_prompt,
            response_format=response_format,
            messages=state["messages"],
            tool_choice=None,
        )

        # prepare messages
        messages = request.messages
        if request.system_prompt:
            messages = [SystemMessage(request.system_prompt), *messages]

        return request, messages

    def _handle_model_output(state: dict[str, Any], output: AIMessage) -> dict[str, Any]:
        """Handle model output including structured responses."""
        # Handle structured output with native strategy
        if isinstance(response_format, ProviderStrategy):
            if not output.tool_calls and native_output_binding:
                structured_response = native_output_binding.parse(output)
                return {"messages": [output], "response": structured_response}
            if state.get("response") is not None:
                return {"messages": [output], "response": None}
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

        # Standard response handling
        if state.get("response") is not None:
            return {"messages": [output], "response": None}
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

    def model_request(state: dict[str, Any]) -> dict[str, Any]:
        """Sync model request handler with sequential middleware processing."""
        # Start with the base model request
        request, messages = _prepare_model_request(state)

        # Apply modify_model_request middleware in sequence
        for m in middleware_w_modify_model_request:
            # Filter state to only include fields defined in this middleware's schema
            filtered_state = _filter_state_for_schema(state, m.state_schema)
            request = m.modify_model_request(request, filtered_state)

        # Get the bound model with the final request
        model_ = _get_bound_model(request)
        output = model_.invoke(messages)
        return _handle_model_output(state, output)

    async def amodel_request(state: dict[str, Any]) -> dict[str, Any]:
        """Async model request handler with sequential middleware processing."""
        # Start with the base model request
        request, messages = _prepare_model_request(state)

        # Apply modify_model_request middleware in sequence
        for m in middleware_w_modify_model_request:
            # Filter state to only include fields defined in this middleware's schema
            filtered_state = _filter_state_for_schema(state, m.state_schema)
            request = m.modify_model_request(request, filtered_state)

        # Get the bound model with the final request
        model_ = _get_bound_model(request)
        output = await model_.ainvoke(messages)
        return _handle_model_output(state, output)

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
                f"{m.__class__.__name__}.before_model",
                m.before_model,
                input_schema=m.state_schema,
            )

        if m.__class__.after_model is not AgentMiddleware.after_model:
            graph.add_node(
                f"{m.__class__.__name__}.after_model",
                m.after_model,
                input_schema=m.state_schema,
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
            _make_tools_to_model_edge(tool_node, first_node),
            [first_node, END],
        )
        graph.add_conditional_edges(
            last_node,
            _make_model_to_tools_edge(first_node, structured_output_tools),
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


def _make_model_to_tools_edge(
    first_node: str, structured_output_tools: dict[str, OutputToolBinding]
) -> Callable[[AgentState], str | None]:
    def model_to_tools(state: AgentState) -> str | None:
        if jump_to := state.get("jump_to"):
            return _resolve_jump(jump_to, first_node)

        message = state["messages"][-1]

        # Check if this is a ToolMessage from structured output - if so, end
        if isinstance(message, ToolMessage) and message.name in structured_output_tools:
            return END

        # Check for tool calls
        if isinstance(message, AIMessage) and message.tool_calls:
            # If all tool calls are for structured output, don't go to tools
            non_structured_calls = [
                tc for tc in message.tool_calls if tc["name"] not in structured_output_tools
            ]
            if non_structured_calls:
                return "tools"

        return END

    return model_to_tools


def _make_tools_to_model_edge(
    tool_node: ToolNode, next_node: str
) -> Callable[[AgentState], str | None]:
    def tools_to_model(state: AgentState) -> str | None:
        ai_message = [m for m in state["messages"] if isinstance(m, AIMessage)][-1]
        if all(
            tool_node.tools_by_name[c["name"]].return_direct
            for c in ai_message.tool_calls
            if c["name"] in tool_node.tools_by_name
        ):
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
