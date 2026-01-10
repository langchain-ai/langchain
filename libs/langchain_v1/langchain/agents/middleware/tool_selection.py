"""LLM-based tool selector middleware."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import Field, TypeAdapter
from typing_extensions import TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain.chat_models.base import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Your goal is to select the most relevant tools for answering the user's query."
)


@dataclass
class _SelectionRequest:
    """Prepared inputs for tool selection."""

    available_tools: list[BaseTool]
    system_message: str
    last_user_message: HumanMessage
    model: BaseChatModel
    valid_tool_names: list[str]


def _create_tool_selection_response(tools: list[BaseTool]) -> TypeAdapter[Any]:
    """Create a structured output schema for tool selection.

    Args:
        tools: Available tools to include in the schema.

    Returns:
        `TypeAdapter` for a schema where each tool name is a `Literal` with its
            description.

    Raises:
        AssertionError: If `tools` is empty.
    """
    if not tools:
        msg = "Invalid usage: tools must be non-empty"
        raise AssertionError(msg)

    # Create a Union of Annotated Literal types for each tool name with description
    # For instance: Union[Annotated[Literal["tool1"], Field(description="...")], ...]
    literals = [
        Annotated[Literal[tool.name], Field(description=tool.description)] for tool in tools
    ]
    selected_tool_type = Union[tuple(literals)]  # type: ignore[valid-type]  # noqa: UP007

    description = "Tools to use. Place the most relevant tools first."

    class ToolSelectionResponse(TypedDict):
        """Use to select relevant tools."""

        tools: Annotated[list[selected_tool_type], Field(description=description)]  # type: ignore[valid-type]

    return TypeAdapter(ToolSelectionResponse)


def _render_tool_list(tools: list[BaseTool]) -> str:
    """Format tools as markdown list.

    Args:
        tools: Tools to format.

    Returns:
        Markdown string with each tool on a new line.
    """
    return "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)


class LLMToolSelectorMiddleware(AgentMiddleware):
    """Uses an LLM to select relevant tools before calling the main model.

    When an agent has many tools available, this middleware filters them down
    to only the most relevant ones for the user's query. This reduces token usage
    and helps the main model focus on the right tools.

    Examples:
        !!! example "Limit to 3 tools"

            ```python
            from langchain.agents.middleware import LLMToolSelectorMiddleware

            middleware = LLMToolSelectorMiddleware(max_tools=3)

            agent = create_agent(
                model="openai:gpt-4o",
                tools=[tool1, tool2, tool3, tool4, tool5],
                middleware=[middleware],
            )
            ```

        !!! example "Use a smaller model for selection"

            ```python
            middleware = LLMToolSelectorMiddleware(model="openai:gpt-4o-mini", max_tools=2)
            ```
    """

    def __init__(
        self,
        *,
        model: str | BaseChatModel | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tools: int | None = None,
        always_include: list[str] | None = None,
    ) -> None:
        """Initialize the tool selector.

        Args:
            model: Model to use for selection.

                If not provided, uses the agent's main model.

                Can be a model identifier string or `BaseChatModel` instance.
            system_prompt: Instructions for the selection model.
            max_tools: Maximum number of tools to select.

                If the model selects more, only the first `max_tools` will be used.

                If not specified, there is no limit.
            always_include: Tool names to always include regardless of selection.

                These do not count against the `max_tools` limit.
        """
        super().__init__()
        self.system_prompt = system_prompt
        self.max_tools = max_tools
        self.always_include = always_include or []

        if isinstance(model, (BaseChatModel, type(None))):
            self.model: BaseChatModel | None = model
        else:
            self.model = init_chat_model(model)

    def _prepare_selection_request(self, request: ModelRequest) -> _SelectionRequest | None:
        """Prepare inputs for tool selection.

        Args:
            request: the model request.

        Returns:
            `SelectionRequest` with prepared inputs, or `None` if no selection is
            needed.

        Raises:
            ValueError: If tools in `always_include` are not found in the request.
            AssertionError: If no user message is found in the request messages.
        """
        # If no tools available, return None
        if not request.tools or len(request.tools) == 0:
            return None

        # Filter to only BaseTool instances (exclude provider-specific tool dicts)
        base_tools = [tool for tool in request.tools if not isinstance(tool, dict)]

        # Validate that always_include tools exist
        if self.always_include:
            available_tool_names = {tool.name for tool in base_tools}
            missing_tools = [
                name for name in self.always_include if name not in available_tool_names
            ]
            if missing_tools:
                msg = (
                    f"Tools in always_include not found in request: {missing_tools}. "
                    f"Available tools: {sorted(available_tool_names)}"
                )
                raise ValueError(msg)

        # Separate tools that are always included from those available for selection
        available_tools = [tool for tool in base_tools if tool.name not in self.always_include]

        # If no tools available for selection, return None
        if not available_tools:
            return None

        system_message = self.system_prompt
        # If there's a max_tools limit, append instructions to the system prompt
        if self.max_tools is not None:
            system_message += (
                f"\nIMPORTANT: List the tool names in order of relevance, "
                f"with the most relevant first. "
                f"If you exceed the maximum number of tools, "
                f"only the first {self.max_tools} will be used."
            )

        # Get the last user message from the conversation history
        last_user_message: HumanMessage
        for message in reversed(request.messages):
            if isinstance(message, HumanMessage):
                last_user_message = message
                break
        else:
            msg = "No user message found in request messages"
            raise AssertionError(msg)

        model = self.model or request.model
        valid_tool_names = [tool.name for tool in available_tools]

        return _SelectionRequest(
            available_tools=available_tools,
            system_message=system_message,
            last_user_message=last_user_message,
            model=model,
            valid_tool_names=valid_tool_names,
        )

    def _process_selection_response(
        self,
        response: dict[str, Any],
        available_tools: list[BaseTool],
        valid_tool_names: list[str],
        request: ModelRequest,
    ) -> ModelRequest:
        """Process the selection response and return filtered `ModelRequest`."""
        selected_tool_names: list[str] = []
        invalid_tool_selections = []

        for tool_name in response["tools"]:
            if tool_name not in valid_tool_names:
                invalid_tool_selections.append(tool_name)
                continue

            # Only add if not already selected and within max_tools limit
            if tool_name not in selected_tool_names and (
                self.max_tools is None or len(selected_tool_names) < self.max_tools
            ):
                selected_tool_names.append(tool_name)

        if invalid_tool_selections:
            msg = f"Model selected invalid tools: {invalid_tool_selections}"
            raise ValueError(msg)

        # Filter tools based on selection and append always-included tools
        selected_tools: list[BaseTool] = [
            tool for tool in available_tools if tool.name in selected_tool_names
        ]
        always_included_tools: list[BaseTool] = [
            tool
            for tool in request.tools
            if not isinstance(tool, dict) and tool.name in self.always_include
        ]
        selected_tools.extend(always_included_tools)

        # Also preserve any provider-specific tool dicts from the original request
        provider_tools = [tool for tool in request.tools if isinstance(tool, dict)]

        return request.override(tools=[*selected_tools, *provider_tools])

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Filter tools based on LLM selection before invoking the model via handler.

        Args:
            request: Model request to execute (includes state and runtime).
            handler: Async callback that executes the model request and returns
                `ModelResponse`.

        Returns:
            The model call result.

        Raises:
            AssertionError: If the selection model response is not a dict.
        """
        selection_request = self._prepare_selection_request(request)
        if selection_request is None:
            return handler(request)

        # Create dynamic response model with Literal enum of available tool names
        type_adapter = _create_tool_selection_response(selection_request.available_tools)
        schema = type_adapter.json_schema()
        structured_model = selection_request.model.with_structured_output(schema)

        response = structured_model.invoke(
            [
                {"role": "system", "content": selection_request.system_message},
                selection_request.last_user_message,
            ]
        )

        # Response should be a dict since we're passing a schema (not a Pydantic model class)
        if not isinstance(response, dict):
            msg = f"Expected dict response, got {type(response)}"
            raise AssertionError(msg)  # noqa: TRY004
        modified_request = self._process_selection_response(
            response, selection_request.available_tools, selection_request.valid_tool_names, request
        )
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Filter tools based on LLM selection before invoking the model via handler.

        Args:
            request: Model request to execute (includes state and runtime).
            handler: Async callback that executes the model request and returns
                `ModelResponse`.

        Returns:
            The model call result.

        Raises:
            AssertionError: If the selection model response is not a dict.
        """
        selection_request = self._prepare_selection_request(request)
        if selection_request is None:
            return await handler(request)

        # Create dynamic response model with Literal enum of available tool names
        type_adapter = _create_tool_selection_response(selection_request.available_tools)
        schema = type_adapter.json_schema()
        structured_model = selection_request.model.with_structured_output(schema)

        response = await structured_model.ainvoke(
            [
                {"role": "system", "content": selection_request.system_message},
                selection_request.last_user_message,
            ]
        )

        # Response should be a dict since we're passing a schema (not a Pydantic model class)
        if not isinstance(response, dict):
            msg = f"Expected dict response, got {type(response)}"
            raise AssertionError(msg)  # noqa: TRY004
        modified_request = self._process_selection_response(
            response, selection_request.available_tools, selection_request.valid_tool_names, request
        )
        return await handler(modified_request)
