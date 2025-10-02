"""LLM-based tool selection middleware for agents."""

from __future__ import annotations

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict

from langchain.chat_models import init_chat_model

from .types import AgentMiddleware, AgentState, ModelRequest, Runtime


class ToolSelectionSchema(BaseModel):
    """Schema for tool selection structured output."""

    selected_tools: list[str] = Field(description="List of selected tool names")


class LLMToolSelectorConfig(TypedDict):
    """Configuration options for the LLM Tool Selector middleware."""

    model: NotRequired[str | BaseChatModel]
    """The language model to use for tool selection

    default: the provided model from the agent options."""

    system_prompt: NotRequired[str]
    """System prompt for the tool selection model."""

    max_tools: NotRequired[int]
    """Maximum number of tools to select."""

    include_full_history: NotRequired[bool]
    """Whether to include the full conversation history in the tool selection prompt."""

    max_retries: NotRequired[int]
    """Maximum number of retries if the model selects incorrect tools."""


DEFAULT_SYSTEM_PROMPT = (
    "Your goal is to select the most relevant tool for answering the user's query."
)
DEFAULT_INCLUDE_FULL_HISTORY = False
DEFAULT_MAX_RETRIES = 3


class LLMToolSelectorMiddleware(AgentMiddleware):
    """Middleware for selecting tools using an LLM-based strategy.

    This middleware analyzes the user's query and available tools to select
    the most relevant tools for the task, reducing the cognitive load on the
    main model and improving response quality.

    Args:
        model: The language model to use for tool selection
            default: the provided model from the agent options.
        system_prompt: System prompt for the tool selection model.
        max_tools: Maximum number of tools to select.
        include_full_history: Whether to include the full conversation
            history in the tool selection prompt.
        max_retries: Maximum number of retries if the model selects incorrect tools.

    Example:
        ```python
        from langchain.agents.middleware.llm_tool_selector import LLMToolSelectorMiddleware
        from langchain.agents import create_agent

        middleware = LLMToolSelectorMiddleware(
            max_tools=3, system_prompt="Select the most relevant tools for the user's query."
        )

        agent = create_agent(
            model="openai:gpt-4o",
            tools=[tool1, tool2, tool3, tool4, tool5],
            middleware=[middleware],
        )
        ```
    """

    def __init__(
        self,
        *,
        model: str | BaseChatModel | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tools: int | None = None,
        include_full_history: bool = DEFAULT_INCLUDE_FULL_HISTORY,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """Initialize the LLM Tool Selector middleware.

        Args:
            model: The language model to use for tool selection (default: the provided model from the agent options).
            system_prompt: System prompt for the tool selection model.
            max_tools: Maximum number of tools to select.
            include_full_history: Whether to include the full conversation history in the tool selection prompt.
            max_retries: Maximum number of retries if the model selects incorrect tools.
        """
        super().__init__()
        self.model = model
        self.system_prompt = system_prompt
        self.max_tools = max_tools
        self.include_full_history = include_full_history
        self.max_retries = max_retries

    def modify_model_request(
        self,
        request: ModelRequest,
        state: AgentState,  # noqa: ARG002
        runtime: Runtime,
    ) -> ModelRequest:
        """Modify the model request to filter tools based on LLM selection.

        Args:
            request: The original model request.
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            The modified model request with filtered tools.
        """
        # If no tools available, return request unchanged
        if not request.tools or len(request.tools) == 0:
            return request

        # Extract tool information
        tool_info = []
        for tool in runtime.tools:
            tool_info.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "tool": tool,
                }
            )

        # Build tool representation for the prompt
        tool_representation = "\n".join(
            f"- {info['name']}: {info['description']}" for info in tool_info
        )

        # Build system message
        system_message = f"""You are an agent that can use the following tools:
{tool_representation}
{self.system_prompt}"""

        if self.include_full_history:
            user_messages = [
                msg.content for msg in request.messages if isinstance(msg, HumanMessage)
            ]
            system_message += f"\nThe full conversation history is:\n{chr(10).join(user_messages)}"

        if self.max_tools is not None:
            system_message += f" You can select up to {self.max_tools} tools."

        # Get the latest user message
        latest_message = request.messages[-1] if request.messages else None
        user_content = (
            latest_message.content
            if isinstance(latest_message, HumanMessage) and isinstance(latest_message.content, str)
            else json.dumps(latest_message.content)
            if latest_message
            else ""
        )

        # Create tool selection model
        tool_selection_model = (
            request.model
            if self.model is None
            else init_chat_model(self.model)
            if isinstance(self.model, str)
            else self.model
        )

        valid_tool_names = [info["name"] for info in tool_info]
        structured_model = tool_selection_model.with_structured_output(ToolSelectionSchema)

        attempts = 0
        selected_tool_names: list[str] = valid_tool_names.copy()

        while attempts <= self.max_retries:
            try:
                response = structured_model.invoke(
                    [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_content},
                    ]
                )

                selected_tool_names = response.selected_tools if response.selected_tools else []

                # Validate that selected tools exist
                invalid_tools = [
                    name for name in selected_tool_names if name not in valid_tool_names
                ]

                if len(selected_tool_names) == 0:
                    system_message += "\n\nNote: You have not selected any tools. Please select at least one tool."
                    attempts += 1
                elif (
                    len(invalid_tools) == 0
                    and self.max_tools is not None
                    and len(selected_tool_names) > self.max_tools
                ):
                    system_message += f"\n\nNote: You have selected more tools than the maximum allowed. You can select up to {self.max_tools} tools."
                    attempts += 1
                elif len(invalid_tools) == 0:
                    # Success
                    break
                elif attempts < self.max_retries:
                    # Retry with feedback about invalid tools
                    system_message += (
                        f"\n\nNote: The following tools are not available: "
                        f"{', '.join(invalid_tools)}. "
                        "Please select only from the available tools."
                    )
                    attempts += 1
                else:
                    # Filter out invalid tools on final attempt
                    selected_tool_names = [
                        name for name in selected_tool_names if name in valid_tool_names
                    ]
                    break
            except Exception:
                # Fall back to using all tools
                if attempts >= self.max_retries:
                    return request
                attempts += 1

        # Filter tools based on selection
        selected_tools = [info["name"] for info in tool_info if info["name"] in selected_tool_names]

        return ModelRequest(
            model=request.model,
            system_prompt=request.system_prompt,
            messages=request.messages,
            tool_choice=request.tool_choice,
            tools=selected_tools,
            response_format=request.response_format,
            model_settings=request.model_settings,
        )
