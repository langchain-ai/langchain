from __future__ import annotations

import inspect
from collections.abc import Sequence  # noqa: TC003
from dataclasses import asdict, is_dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Generic,
    Optional,
    Union,
    cast,
    get_type_hints,
)

from langchain_core.language_models import (
    BaseChatModel,
    LanguageModelInput,
    LanguageModelLike,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
)
from langgraph._internal._runnable import RunnableCallable, RunnableLike
from langgraph.errors import ErrorCode, create_error_message
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps  # noqa: TC002
from langgraph.types import Checkpointer, Command, Send
from langgraph.typing import ContextT, StateT
from pydantic import BaseModel
from typing_extensions import NotRequired, TypedDict, TypeVar

from langchain.agents.structured_output import (
    MultipleStructuredOutputsError,
    NativeOutput,
    NativeOutputBinding,
    OutputToolBinding,
    ResponseFormat,
    StructuredOutputParsingError,
    ToolOutput,
)
from langchain.agents.tool_node import ToolNode

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.runtime import Runtime
    from langgraph.store.base import BaseStore

    from langchain.agents._internal._typing import (
        SyncOrAsync,
    )

StructuredResponseT = TypeVar(
    "StructuredResponseT",
    default=None,
)

STRUCTURED_OUTPUT_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."


class AgentState(TypedDict, Generic[StructuredResponseT]):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    remaining_steps: NotRequired[RemainingSteps]

    structured_response: NotRequired[StructuredResponseT]


PROMPT_RUNNABLE_NAME = "Prompt"

Prompt = Union[
    SystemMessage,
    str,
    Callable[[StateT], LanguageModelInput],
    Runnable[StateT, LanguageModelInput],
]


def _get_state_value(state: StateT, key: str, default: Any = None) -> Any:
    return state.get(key, default) if isinstance(state, dict) else getattr(state, key, default)


def _get_prompt_runnable(prompt: Optional[Prompt]) -> Runnable:
    prompt_runnable: Runnable
    if prompt is None:
        prompt_runnable = RunnableCallable(
            lambda state: _get_state_value(state, "messages"), name=PROMPT_RUNNABLE_NAME
        )
    elif isinstance(prompt, str):
        _system_message: BaseMessage = SystemMessage(content=prompt)
        prompt_runnable = RunnableCallable(
            lambda state: [_system_message, *_get_state_value(state, "messages")],
            name=PROMPT_RUNNABLE_NAME,
        )
    elif isinstance(prompt, SystemMessage):
        prompt_runnable = RunnableCallable(
            lambda state: [prompt, *_get_state_value(state, "messages")],
            name=PROMPT_RUNNABLE_NAME,
        )
    elif inspect.iscoroutinefunction(prompt):
        prompt_runnable = RunnableCallable(
            None,
            prompt,
            name=PROMPT_RUNNABLE_NAME,
        )
    elif callable(prompt):
        prompt_runnable = RunnableCallable(
            prompt,
            name=PROMPT_RUNNABLE_NAME,
        )
    elif isinstance(prompt, Runnable):
        prompt_runnable = prompt
    else:
        msg = f"Got unexpected type for `prompt`: {type(prompt)}"
        raise ValueError(msg)

    return prompt_runnable


def _validate_chat_history(
    messages: Sequence[BaseMessage],
) -> None:
    """Validate that all tool calls in AIMessages have a corresponding ToolMessage."""
    all_tool_calls = [
        tool_call
        for message in messages
        if isinstance(message, AIMessage)
        for tool_call in message.tool_calls
    ]
    tool_call_ids_with_results = {
        message.tool_call_id for message in messages if isinstance(message, ToolMessage)
    }
    tool_calls_without_results = [
        tool_call
        for tool_call in all_tool_calls
        if tool_call["id"] not in tool_call_ids_with_results
    ]
    if not tool_calls_without_results:
        return

    error_message = create_error_message(
        message="Found AIMessages with tool_calls that do not have a corresponding ToolMessage. "
        f"Here are the first few of those tool calls: {tool_calls_without_results[:3]}.\n\n"
        "Every tool call (LLM requesting to call a tool) in the message history MUST have a "
        "corresponding ToolMessage (result of a tool invocation to return to the LLM) - "
        "this is required by most LLM providers.",
        error_code=ErrorCode.INVALID_CHAT_HISTORY,
    )
    raise ValueError(error_message)


class _AgentBuilder(Generic[StateT, ContextT, StructuredResponseT]):
    """Internal builder class for constructing and agent."""

    _final_state_schema: type[StateT]

    def __init__(
        self,
        model: Union[
            str,
            BaseChatModel,
            SyncOrAsync[[StateT, Runtime[ContextT]], BaseChatModel],
        ],
        tools: Union[Sequence[Union[BaseTool, Callable, dict[str, Any]]], ToolNode],
        *,
        prompt: Optional[Prompt] = None,
        response_format: Optional[ResponseFormat[StructuredResponseT]] = None,
        pre_model_hook: Optional[RunnableLike] = None,
        post_model_hook: Optional[RunnableLike] = None,
        state_schema: Optional[type[StateT]] = None,
        context_schema: Optional[type[ContextT]] = None,
        name: Optional[str] = None,
        store: Optional[BaseStore] = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.prompt = prompt
        self.response_format = response_format
        self.pre_model_hook = pre_model_hook
        self.post_model_hook = post_model_hook
        self.state_schema = state_schema
        self.context_schema = context_schema
        self.name = name
        self.store = store

        if isinstance(model, Runnable) and not isinstance(model, BaseChatModel):
            msg = (
                "Expected `model` to be a BaseChatModel or a string, got {type(model)}."
                "The `model` parameter should not have pre-bound tools, simply pass the model and tools separately."
            )
            raise TypeError(msg)

        self._setup_tools()
        self._setup_state_schema()
        self._setup_structured_output()
        self._setup_model()

    def _setup_tools(self) -> None:
        """Setup tool-related attributes."""
        if isinstance(self.tools, ToolNode):
            self._tool_classes = list(self.tools.tools_by_name.values())
            self._tool_node = self.tools
            self._llm_builtin_tools = []
        else:
            self._llm_builtin_tools = [t for t in self.tools if isinstance(t, dict)]
            self._tool_node = ToolNode([t for t in self.tools if not isinstance(t, dict)])
            self._tool_classes = list(self._tool_node.tools_by_name.values())

        self._should_return_direct = {t.name for t in self._tool_classes if t.return_direct}
        self._tool_calling_enabled = len(self._tool_classes) > 0

    def _setup_structured_output(self) -> None:
        """Set up structured output tracking for "tools" and "native" strategies.

        "tools" strategy for structured output:
        1. Converting response format schemas to LangChain tools
        2. Creating metadata for proper response reconstruction
        3. Handling both Pydantic models and dict schemas

        "native" strategy for structured output:
        1. Capturing the schema reference for later parsing
        2. Binding provider-native response_format kwargs at model bind time
        3. Parsing provider-enforced structured output directly into the schema
        """
        self.structured_output_tools: dict[str, OutputToolBinding[StructuredResponseT]] = {}
        self.native_output_binding: NativeOutputBinding[StructuredResponseT] | None = None

        if self.response_format is not None:
            response_format = self.response_format

            if isinstance(response_format, ToolOutput):
                # check if response_format.schema is a union
                for response_schema in response_format.schema_specs:
                    structured_tool_info = OutputToolBinding.from_schema_spec(response_schema)
                    self.structured_output_tools[structured_tool_info.tool.name] = (
                        structured_tool_info
                    )
            elif isinstance(response_format, NativeOutput):
                # Use native strategy - create NativeOutputBinding for parsing
                self.native_output_binding = NativeOutputBinding.from_schema_spec(
                    response_format.schema_spec
                )
            else:
                # This shouldn't happen with the new ResponseFormat type, but keeping for safety
                msg = (
                    f"Unsupported response_format type: {type(response_format)}. "
                    f"Expected ToolOutput."
                )
                raise ValueError(msg)

    def _setup_state_schema(self) -> None:
        """Setup state schema with validation."""
        if self.state_schema is not None:
            required_keys = {"messages", "remaining_steps"}
            if self.response_format is not None:
                required_keys.add("structured_response")

            schema_keys = set(get_type_hints(self.state_schema))
            if missing_keys := required_keys - schema_keys:
                msg = f"Missing required key(s) {missing_keys} in state_schema"
                raise ValueError(msg)

            self._final_state_schema = self.state_schema
        else:
            self._final_state_schema = cast("type[StateT]", AgentState)

    def _handle_structured_response_tool_calls(self, response: AIMessage) -> Optional[Command]:
        """Handle tool calls that match structured output tools using the tools strategy.

        Args:
            response: The AI message containing potential tool calls

        Returns:
            Command with structured response update if found, None otherwise

        Raises:
            MultipleStructuredOutputsError: If multiple structured responses are returned and error handling is disabled
            StructuredOutputParsingError: If parsing fails and error handling is disabled
        """
        if not isinstance(self.response_format, ToolOutput) or not response.tool_calls:
            return None

        structured_tool_calls = [
            tool_call
            for tool_call in response.tool_calls
            if tool_call["name"] in self.structured_output_tools
        ]

        if not structured_tool_calls:
            return None

        if len(structured_tool_calls) > 1:
            return self._handle_multiple_structured_outputs(response, structured_tool_calls)

        return self._handle_single_structured_output(response, structured_tool_calls[0])

    def _handle_multiple_structured_outputs(
        self,
        response: AIMessage,
        structured_tool_calls: list[ToolCall],
    ) -> Command:
        """Handle multiple structured output tool calls."""
        tool_names = [tool_call["name"] for tool_call in structured_tool_calls]
        exception = MultipleStructuredOutputsError(tool_names)

        should_retry, error_message = self._handle_structured_output_error(exception)

        if not should_retry:
            raise exception

        tool_messages = [
            ToolMessage(
                content=error_message,
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            )
            for tool_call in structured_tool_calls
        ]

        return Command(
            update={"messages": [response, *tool_messages]},
            goto="model",
        )

    def _handle_single_structured_output(
        self,
        response: AIMessage,
        tool_call: Any,
    ) -> Command:
        """Handle a single structured output tool call."""
        structured_tool_binding = self.structured_output_tools[tool_call["name"]]

        try:
            structured_response = structured_tool_binding.parse(tool_call["args"])

            if isinstance(structured_response, BaseModel):
                structured_response_dict = structured_response.model_dump()
            elif is_dataclass(structured_response):
                structured_response_dict = asdict(structured_response)  # type: ignore[arg-type]
            else:
                structured_response_dict = cast("dict", structured_response)

            tool_message_content = (
                self.response_format.tool_message_content
                if isinstance(self.response_format, ToolOutput)
                and self.response_format.tool_message_content
                else f"Returning structured response: {structured_response_dict}"
            )

            return Command(
                update={
                    "messages": [
                        response,
                        ToolMessage(
                            content=tool_message_content,
                            tool_call_id=tool_call["id"],
                            name=tool_call["name"],
                        ),
                    ],
                    "structured_response": structured_response,
                }
            )
        except Exception as parse_error:
            exception = StructuredOutputParsingError(tool_call["name"], parse_error)

            should_retry, error_message = self._handle_structured_output_error(exception)

            if not should_retry:
                raise exception from parse_error

            return Command(
                update={
                    "messages": [
                        response,
                        ToolMessage(
                            content=error_message,
                            tool_call_id=tool_call["id"],
                            name=tool_call["name"],
                        ),
                    ],
                },
                goto="model",
            )

    def _handle_structured_output_error(  # noqa: PLR0911
        self,
        exception: Exception,
    ) -> tuple[bool, str]:
        """Handle structured output error.

        Returns (should_retry, retry_tool_message).
        """
        handle_errors = cast("ToolOutput", self.response_format).handle_errors

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
            return True, handle_errors(exception)  # type: ignore[call-arg]
        return False, ""

    def _apply_native_output_binding(self, model: LanguageModelLike) -> LanguageModelLike:
        """If native output is configured, bind provider-native kwargs onto the model."""
        if not isinstance(self.response_format, NativeOutput):
            return model
        kwargs = self.response_format.to_model_kwargs()
        return model.bind(**kwargs)

    def _handle_structured_response_native(self, response: AIMessage) -> Optional[Command]:
        """If native output is configured and there are no tool calls, parse using NativeOutputBinding."""
        if self.native_output_binding is None:
            return None
        if response.tool_calls:
            # if the model chooses to call tools, we let the normal flow handle it
            return None

        structured_response = self.native_output_binding.parse(response)

        return Command(update={"messages": [response], "structured_response": structured_response})

    def _setup_model(self) -> None:
        """Setup model-related attributes."""
        self._is_dynamic_model = not isinstance(self.model, (str, Runnable)) and callable(
            self.model
        )
        self._is_async_dynamic_model = self._is_dynamic_model and inspect.iscoroutinefunction(
            self.model
        )

        if not self._is_dynamic_model:
            model = self.model
            if isinstance(model, str):
                try:
                    from langchain.chat_models import (  # type: ignore[import-not-found]
                        init_chat_model,
                    )
                except ImportError:
                    msg = "Please install langchain (`pip install langchain`) to use '<provider>:<model>' string syntax for `model` parameter."
                    raise ImportError(msg) from None
                model = init_chat_model(model)

            # Collect all tools: regular tools + structured output tools
            structured_output_tools = list(self.structured_output_tools.values())
            all_tools = (
                self._tool_classes
                + self._llm_builtin_tools
                + [info.tool for info in structured_output_tools]
            )

            if len(all_tools) > 0:
                # Check if we need to force tool use for structured output
                tool_choice = None
                if self.response_format is not None and isinstance(
                    self.response_format, ToolOutput
                ):
                    tool_choice = "any"

                if tool_choice:
                    model = cast("BaseChatModel", model).bind_tools(  # type: ignore[assignment]
                        all_tools, tool_choice=tool_choice
                    )
                # If native output is configured, bind tools with strict=True. Required for OpenAI.
                elif isinstance(self.response_format, NativeOutput):
                    model = cast("BaseChatModel", model).bind_tools(  # type: ignore[assignment]
                        all_tools, strict=True
                    )
                else:
                    model = cast("BaseChatModel", model).bind_tools(all_tools)  # type: ignore[assignment]

            # bind native structured-output kwargs
            model = self._apply_native_output_binding(model)  # type: ignore[arg-type]

            # Extract just the model part for direct invocation
            self._static_model: Optional[Runnable] = model
        else:
            self._static_model = None

    def _resolve_model(self, state: StateT, runtime: Runtime[ContextT]) -> LanguageModelLike:
        """Resolve the model to use, handling both static and dynamic models."""
        if self._is_dynamic_model:
            dynamic_model = self.model(state, runtime)  # type: ignore[operator, arg-type]
            return self._apply_native_output_binding(dynamic_model)  # type: ignore[arg-type]
        return self._static_model

    async def _aresolve_model(self, state: StateT, runtime: Runtime[ContextT]) -> LanguageModelLike:
        """Async resolve the model to use, handling both static and dynamic models."""
        if self._is_async_dynamic_model:
            dynamic_model = cast(
                "Callable[[StateT, Runtime[ContextT]], Awaitable[BaseChatModel]]",
                self.model,
            )
            return await dynamic_model(state, runtime)
        if self._is_dynamic_model:
            dynamic_model = self.model(state, runtime)  # type: ignore[arg-type, assignment, operator]
            return self._apply_native_output_binding(dynamic_model)  # type: ignore[arg-type]
        return self._static_model

    def create_model_node(self) -> RunnableCallable:  # noqa: C901
        """Create the 'agent' node that calls the LLM."""

        def _get_model_input_state(state: StateT) -> StateT:
            messages = _get_state_value(state, "messages")
            error_msg = f"Expected input to call_model to have 'messages' key, but got {state}"

            if messages is None:
                raise ValueError(error_msg)

            _validate_chat_history(messages)

            if isinstance(self._final_state_schema, type) and issubclass(
                self._final_state_schema, BaseModel
            ):
                # we're passing messages under `messages` key, as this
                # is expected by the prompt
                state.messages = messages
            else:
                state["messages"] = messages
            return state

        def _are_more_steps_needed(state: StateT, response: BaseMessage) -> bool:
            has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
            all_tools_return_direct = (
                all(call["name"] in self._should_return_direct for call in response.tool_calls)
                if isinstance(response, AIMessage)
                else False
            )
            remaining_steps = _get_state_value(state, "remaining_steps", None)
            more_steps_needed = remaining_steps is not None and (
                (remaining_steps < 1 and all_tools_return_direct)
                or (remaining_steps < 2 and has_tool_calls)
            )
            return cast("bool", more_steps_needed)

        def call_model(
            state: StateT, runtime: Runtime[ContextT], config: RunnableConfig
        ) -> dict[str, Any] | Command:
            """Call the model with the current state and return the response."""
            if self._is_async_dynamic_model:
                msg = (
                    "Async model callable provided but agent invoked synchronously. "
                    "Use agent.ainvoke() or agent.astream(), or provide a sync model callable."
                )
                raise RuntimeError(msg)

            model_input = _get_model_input_state(state)
            model = self._resolve_model(state, runtime)

            # Get prompt runnable and invoke it first to prepare messages
            prompt_runnable = _get_prompt_runnable(self.prompt)
            prepared_messages = prompt_runnable.invoke(model_input, config)

            # Then invoke the model with the prepared messages
            response = cast("AIMessage", model.invoke(prepared_messages, config))
            response.name = self.name

            if _are_more_steps_needed(state, response):
                return {
                    "messages": [
                        AIMessage(
                            id=response.id,
                            content="Sorry, need more steps to process this request.",
                        )
                    ]
                }

            # Check if any tool calls match structured output tools
            structured_command = self._handle_structured_response_tool_calls(response)
            if structured_command:
                return structured_command

            # Native structured output
            native_command = self._handle_structured_response_native(response)
            if native_command:
                return native_command

            return {"messages": [response]}

        async def acall_model(
            state: StateT, runtime: Runtime[ContextT], config: RunnableConfig
        ) -> dict[str, Any] | Command:
            """Call the model with the current state and return the response."""
            model_input = _get_model_input_state(state)

            model = await self._aresolve_model(state, runtime)

            # Get prompt runnable and invoke it first to prepare messages
            prompt_runnable = _get_prompt_runnable(self.prompt)
            prepared_messages = await prompt_runnable.ainvoke(model_input, config)

            # Then invoke the model with the prepared messages
            response = cast(
                "AIMessage",
                await model.ainvoke(prepared_messages, config),
            )
            response.name = self.name
            if _are_more_steps_needed(state, response):
                return {
                    "messages": [
                        AIMessage(
                            id=response.id,
                            content="Sorry, need more steps to process this request.",
                        )
                    ]
                }

            # Check if any tool calls match structured output tools
            structured_command = self._handle_structured_response_tool_calls(response)
            if structured_command:
                return structured_command

            # Native structured output
            native_command = self._handle_structured_response_native(response)
            if native_command:
                return native_command

            return {"messages": [response]}

        return RunnableCallable(call_model, acall_model)

    def create_model_router(self) -> Callable[[StateT], Union[str, list[Send]]]:
        """Create routing function for model node conditional edges."""

        def should_continue(state: StateT) -> Union[str, list[Send]]:
            messages = _get_state_value(state, "messages")
            last_message = messages[-1]

            # Check if the last message is a ToolMessage from a structured tool.
            # This condition exists to support structured output via tools.
            # Once a tool has been called for structured output, we skip
            # tool execution and go to END (if there is no post_model_hook).
            if (
                isinstance(last_message, ToolMessage)
                and last_message.name in self.structured_output_tools
            ):
                return END

            if isinstance(last_message, ToolMessage):
                return END

            if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
                if self.post_model_hook is not None:
                    return "post_model_hook"
                return END
            if self.post_model_hook is not None:
                return "post_model_hook"
            tool_calls = [
                self._tool_node.inject_tool_args(call, state, self.store)  # type: ignore[arg-type]
                for call in last_message.tool_calls
            ]
            return [Send("tools", [tool_call]) for tool_call in tool_calls]

        return should_continue

    def create_post_model_hook_router(
        self,
    ) -> Callable[[StateT], Union[str, list[Send]]]:
        """Create a routing function for post_model_hook node conditional edges."""

        def post_model_hook_router(state: StateT) -> Union[str, list[Send]]:
            messages = _get_state_value(state, "messages")

            # Check if the last message is a ToolMessage from a structured tool.
            # This condition exists to support structured output via tools.
            # Once a tool has been called for structured output, we skip
            # tool execution and go to END (if there is no post_model_hook).
            last_message = messages[-1]
            if (
                isinstance(last_message, ToolMessage)
                and last_message.name in self.structured_output_tools
            ):
                return END

            tool_messages = [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]
            last_ai_message = next(m for m in reversed(messages) if isinstance(m, AIMessage))
            pending_tool_calls = [
                c for c in last_ai_message.tool_calls if c["id"] not in tool_messages
            ]

            if pending_tool_calls:
                pending_tool_calls = [
                    self._tool_node.inject_tool_args(call, state, self.store)  # type: ignore[arg-type]
                    for call in pending_tool_calls
                ]
                return [Send("tools", [tool_call]) for tool_call in pending_tool_calls]
            if isinstance(messages[-1], ToolMessage):
                return self._get_entry_point()
            return END

        return post_model_hook_router

    def create_tools_router(self) -> Optional[Callable[[StateT], str]]:
        """Create a routing function for tools node conditional edges."""
        if not self._should_return_direct:
            return None

        def route_tool_responses(state: StateT) -> str:
            messages = _get_state_value(state, "messages")
            for m in reversed(messages):
                if not isinstance(m, ToolMessage):
                    break
                if m.name in self._should_return_direct:
                    return END

            if (
                isinstance(m, AIMessage)
                and m.tool_calls
                and any(call["name"] in self._should_return_direct for call in m.tool_calls)
            ):
                return END

            return self._get_entry_point()

        return route_tool_responses

    def _get_entry_point(self) -> str:
        """Get the workflow entry point."""
        return "pre_model_hook" if self.pre_model_hook else "model"

    def _get_model_paths(self) -> list[str]:
        """Get possible edge destinations from model node."""
        paths = []
        if self._tool_calling_enabled:
            paths.append("tools")
        if self.post_model_hook:
            paths.append("post_model_hook")
        else:
            paths.append(END)

        return paths

    def _get_post_model_hook_paths(self) -> list[str]:
        """Get possible edge destinations from post_model_hook node."""
        paths = []
        if self._tool_calling_enabled:
            paths = [self._get_entry_point(), "tools"]
        paths.append(END)
        return paths

    def build(self) -> StateGraph:  # noqa: PLR0912
        """Build the agent workflow graph (uncompiled)."""
        workflow = StateGraph(
            state_schema=self._final_state_schema,
            context_schema=self.context_schema,
        )

        # Set entry point
        workflow.set_entry_point(self._get_entry_point())

        # Add nodes
        workflow.add_node("model", self.create_model_node())

        if self._tool_calling_enabled:
            workflow.add_node("tools", self._tool_node)

        if self.pre_model_hook:
            workflow.add_node("pre_model_hook", self.pre_model_hook)  # type: ignore[arg-type]

        if self.post_model_hook:
            workflow.add_node("post_model_hook", self.post_model_hook)  # type: ignore[arg-type]

        # Add edges
        if self.pre_model_hook:
            workflow.add_edge("pre_model_hook", "model")

        if self.post_model_hook:
            workflow.add_edge("model", "post_model_hook")
            post_hook_paths = self._get_post_model_hook_paths()
            if len(post_hook_paths) == 1:
                # No need for a conditional edge if there's only one path
                workflow.add_edge("post_model_hook", post_hook_paths[0])
            else:
                workflow.add_conditional_edges(
                    "post_model_hook",
                    self.create_post_model_hook_router(),
                    path_map=post_hook_paths,
                )
        else:
            model_paths = self._get_model_paths()
            if len(model_paths) == 1:
                # No need for a conditional edge if there's only one path
                workflow.add_edge("model", model_paths[0])
            else:
                workflow.add_conditional_edges(
                    "model",
                    self.create_model_router(),
                    path_map=model_paths,
                )

        if self._tool_calling_enabled:
            # In some cases, tools can return directly. In these cases
            # we add a conditional edge from the tools node to the END node
            # instead of going to the entry point.
            tools_router = self.create_tools_router()
            if tools_router:
                workflow.add_conditional_edges(
                    "tools",
                    tools_router,
                    path_map=[self._get_entry_point(), END],
                )
            else:
                workflow.add_edge("tools", self._get_entry_point())

        return workflow


def _supports_native_structured_output(
    model: Union[str, BaseChatModel, SyncOrAsync[[StateT, Runtime[ContextT]], BaseChatModel]],
) -> bool:
    """Check if a model supports native structured output.

    TODO: replace with more robust model profiles.
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


def create_react_agent(
    model: Union[
        str,
        BaseChatModel,
        SyncOrAsync[[StateT, Runtime[ContextT]], BaseChatModel],
    ],
    tools: Union[Sequence[Union[BaseTool, Callable, dict[str, Any]]], ToolNode],
    *,
    prompt: Optional[Prompt] = None,
    response_format: Optional[
        Union[
            ToolOutput[StructuredResponseT],
            NativeOutput[StructuredResponseT],
            type[StructuredResponseT],
        ]
    ] = None,
    pre_model_hook: Optional[RunnableLike] = None,
    post_model_hook: Optional[RunnableLike] = None,
    state_schema: Optional[type[StateT]] = None,
    context_schema: Optional[type[ContextT]] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    debug: bool = False,
    name: Optional[str] = None,
) -> CompiledStateGraph:
    """Creates an agent graph that calls tools in a loop until a stopping condition is met.

    For more details on using `create_react_agent`, visit [Agents](https://langchain-ai.github.io/langgraph/agents/overview/) documentation.

    Args:
        model: The language model for the agent. Supports static and dynamic
            model selection.

            - **Static model**: A chat model instance (e.g., `ChatOpenAI()`) or
              string identifier (e.g., `"openai:gpt-4"`)
            - **Dynamic model**: A callable with signature
              `(state, runtime) -> BaseChatModel` that returns different models
              based on runtime context
              If the model has tools bound via `.bind_tools()` or other configurations,
              the return type should be a Runnable[LanguageModelInput, BaseMessage]
              Coroutines are also supported, allowing for asynchronous model selection.

            Dynamic functions receive graph state and runtime, enabling
            context-dependent model selection. Must return a `BaseChatModel`
            instance. For tool calling, bind tools using `.bind_tools()`.
            Bound tools must be a subset of the `tools` parameter.

            Dynamic model example:
            ```python
            from dataclasses import dataclass

            @dataclass
            class ModelContext:
                model_name: str = "gpt-3.5-turbo"

            # Instantiate models globally
            gpt4_model = ChatOpenAI(model="gpt-4")
            gpt35_model = ChatOpenAI(model="gpt-3.5-turbo")

            def select_model(state: AgentState, runtime: Runtime[ModelContext]) -> ChatOpenAI:
                model_name = runtime.context.model_name
                model = gpt4_model if model_name == "gpt-4" else gpt35_model
                return model.bind_tools(tools)
            ```

            !!! note "Dynamic Model Requirements"
                Ensure returned models have appropriate tools bound via
                `.bind_tools()` and support required functionality. Bound tools
                must be a subset of those specified in the `tools` parameter.

        tools: A list of tools or a ToolNode instance.
            If an empty list is provided, the agent will consist of a single LLM node without tool calling.
        prompt: An optional prompt for the LLM. Can take a few different forms:

            - str: This is converted to a SystemMessage and added to the beginning of the list of messages in state["messages"].
            - SystemMessage: this is added to the beginning of the list of messages in state["messages"].
            - Callable: This function should take in full graph state and the output is then passed to the language model.
            - Runnable: This runnable should take in full graph state and the output is then passed to the language model.

        response_format: An optional UsingToolStrategy configuration for structured responses.

            If provided, the agent will handle structured output via tool calls during the normal conversation flow.
            When the model calls a structured output tool, the response will be captured and returned in the 'structured_response' state key.
            If not provided, `structured_response` will not be present in the output state.

            The UsingToolStrategy should contain:
                - schemas: A sequence of ResponseSchema objects that define the structured output format
                - tool_choice: Either "required" or "auto" to control when structured output is used

            Each ResponseSchema contains:
                - schema: A Pydantic model that defines the structure
                - name: Optional custom name for the tool (defaults to model name)
                - description: Optional custom description (defaults to model docstring)
                - strict: Whether to enforce strict validation

            !!! Important
                `response_format` requires the model to support tool calling

            !!! Note
                Structured responses are handled directly in the model call node via tool calls, eliminating the need for separate structured response nodes.

        pre_model_hook: An optional node to add before the `agent` node (i.e., the node that calls the LLM).
            Useful for managing long message histories (e.g., message trimming, summarization, etc.).
            Pre-model hook must be a callable or a runnable that takes in current graph state and returns a state update in the form of
                ```python
                # Where `messages` MUST be provided
                {
                    # will UPDATE the `messages` in the state
                    "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), ...],
                    # Any other state keys that need to be propagated
                    ...
                }
                ```

            !!! Warning
                you should OVERWRITE the `messages` key by doing the following:

                ```python
                {
                    "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_messages]
                    ...
                }
                ```
        post_model_hook: An optional node to add after the `agent` node (i.e., the node that calls the LLM).
            Useful for implementing human-in-the-loop, guardrails, validation, or other post-processing.
            Post-model hook must be a callable or a runnable that takes in current graph state and returns a state update.
        state_schema: An optional state schema that defines graph state.
            Must have `messages` and `remaining_steps` keys.
            Defaults to `AgentState` that defines those two keys.
        context_schema: An optional schema for runtime context.
        checkpointer: An optional checkpoint saver object. This is used for persisting
            the state of the graph (e.g., as chat memory) for a single thread (e.g., a single conversation).
        store: An optional store object. This is used for persisting data
            across multiple threads (e.g., multiple conversations / users).
        interrupt_before: An optional list of node names to interrupt before.
            Should be one of the following: "model", "tools".
            This is useful if you want to add a user confirmation or other interrupt before taking an action.
        interrupt_after: An optional list of node names to interrupt after.
            Should be one of the following: "model", "tools".
            This is useful if you want to return directly or run additional processing on an output.
        debug: A flag indicating whether to enable debug mode.
        name: An optional name for the CompiledStateGraph.
            This name will be automatically used when adding ReAct agent graph to another graph as a subgraph node -
            particularly useful for building multi-agent systems.

    Returns:
        A CompiledStateGraph that can be used for chat interactions.

    The "model" node calls the language model with the messages list (after applying the prompt).
    If the resulting AIMessage contains `tool_calls`, the graph will then call the ["tools"][langgraph.prebuilt.tool_node.ToolNode].
    The "tools" node executes the tools (1 tool per `tool_call`) and adds the responses to the messages list
    as `ToolMessage` objects. The agent node then calls the language model again.
    The process repeats until no more `tool_calls` are present in the response.
    The agent then returns the full list of messages as a dictionary containing the key "messages".

    ``` mermaid
        sequenceDiagram
            participant U as User
            participant A as LLM
            participant T as Tools
            U->>A: Initial input
            Note over A: Prompt + LLM
            loop while tool_calls present
                A->>T: Execute tools
                T-->>A: ToolMessage for each tool_calls
            end
            A->>U: Return final state
    ```

    Example:
        ```python
        from langchain.agents import create_react_agent

        def check_weather(location: str) -> str:
            '''Return the weather forecast for the specified location.'''
            return f"It's always sunny in {location}"

        graph = create_react_agent(
            "anthropic:claude-3-7-sonnet-latest",
            tools=[check_weather],
            prompt="You are a helpful assistant",
        )
        inputs = {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
        for chunk in graph.stream(inputs, stream_mode="updates"):
            print(chunk)
        ```
    """
    if response_format and not isinstance(response_format, (ToolOutput, NativeOutput)):
        if _supports_native_structured_output(model):
            response_format = NativeOutput(
                schema=response_format,
            )
        else:
            response_format = ToolOutput(
                schema=response_format,
            )
    elif isinstance(response_format, tuple) and len(response_format) == 2:
        msg = "Passing a 2-tuple as response_format is no longer supported. "
        raise ValueError(msg)

    # Create and configure the agent builder
    builder = _AgentBuilder[StateT, ContextT, StructuredResponseT](
        model=model,
        tools=tools,
        prompt=prompt,
        response_format=cast("Union[ResponseFormat[StructuredResponseT], None]", response_format),
        pre_model_hook=pre_model_hook,
        post_model_hook=post_model_hook,
        state_schema=state_schema,
        context_schema=context_schema,
        name=name,
        store=store,
    )

    # Build and compile the workflow
    workflow = builder.build()
    return workflow.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        name=name,
    )


__all__ = [
    "AgentState",
    "create_react_agent",
]
