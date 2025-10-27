"""Tool emulator middleware for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage

from langchain.agents.middleware.types import AgentMiddleware
from langchain.chat_models.base import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.types import Command

    from langchain.tools import BaseTool
    from langchain.tools.tool_node import ToolCallRequest


class LLMToolEmulator(AgentMiddleware):
    """Middleware that emulates specified tools using an LLM instead of executing them.

    This middleware allows selective emulation of tools for testing purposes.
    By default (when tools=None), all tools are emulated. You can specify which
    tools to emulate by passing a list of tool names or BaseTool instances.

    Examples:
        Emulate all tools (default behavior):
        ```python
        from langchain.agents.middleware import LLMToolEmulator

        middleware = LLMToolEmulator()

        agent = create_agent(
            model="openai:gpt-4o",
            tools=[get_weather, get_user_location, calculator],
            middleware=[middleware],
        )
        ```

        Emulate specific tools by name:
        ```python
        middleware = LLMToolEmulator(tools=["get_weather", "get_user_location"])
        ```

        Use a custom model for emulation:
        ```python
        middleware = LLMToolEmulator(
            tools=["get_weather"], model="anthropic:claude-3-5-sonnet-latest"
        )
        ```

        Emulate specific tools by passing tool instances:
        ```python
        middleware = LLMToolEmulator(tools=[get_weather, get_user_location])
        ```
    """

    def __init__(
        self,
        *,
        tools: list[str | BaseTool] | None = None,
        model: str | BaseChatModel | None = None,
    ) -> None:
        """Initialize the tool emulator.

        Args:
            tools: List of tool names (str) or BaseTool instances to emulate.
                If None (default), ALL tools will be emulated.
                If empty list, no tools will be emulated.
            model: Model to use for emulation.
                Defaults to "anthropic:claude-3-5-sonnet-latest".
                Can be a model identifier string or BaseChatModel instance.
        """
        super().__init__()

        # Extract tool names from tools
        # None means emulate all tools
        self.emulate_all = tools is None
        self.tools_to_emulate: set[str] = set()

        if not self.emulate_all and tools is not None:
            for tool in tools:
                if isinstance(tool, str):
                    self.tools_to_emulate.add(tool)
                else:
                    # Assume BaseTool with .name attribute
                    self.tools_to_emulate.add(tool.name)

        # Initialize emulator model
        if model is None:
            self.model = init_chat_model("anthropic:claude-3-5-sonnet-latest", temperature=1)
        elif isinstance(model, BaseChatModel):
            self.model = model
        else:
            self.model = init_chat_model(model, temperature=1)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Emulate tool execution using LLM if tool should be emulated.

        Args:
            request: Tool call request to potentially emulate.
            handler: Callback to execute the tool (can be called multiple times).

        Returns:
            ToolMessage with emulated response if tool should be emulated,
            otherwise calls handler for normal execution.
        """
        tool_name = request.tool_call["name"]

        # Check if this tool should be emulated
        should_emulate = self.emulate_all or tool_name in self.tools_to_emulate

        if not should_emulate:
            # Let it execute normally by calling the handler
            return handler(request)

        # Extract tool information for emulation
        tool_args = request.tool_call["args"]
        tool_description = request.tool.description if request.tool else "No description available"

        # Build prompt for emulator LLM
        prompt = (
            f"You are emulating a tool call for testing purposes.\n\n"
            f"Tool: {tool_name}\n"
            f"Description: {tool_description}\n"
            f"Arguments: {tool_args}\n\n"
            f"Generate a realistic response that this tool would return "
            f"given these arguments.\n"
            f"Return ONLY the tool's output, no explanation or preamble. "
            f"Introduce variation into your responses."
        )

        # Get emulated response from LLM
        response = self.model.invoke([HumanMessage(prompt)])

        # Short-circuit: return emulated result without executing real tool
        return ToolMessage(
            content=response.content,
            tool_call_id=request.tool_call["id"],
            name=tool_name,
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Async version of wrap_tool_call.

        Emulate tool execution using LLM if tool should be emulated.

        Args:
            request: Tool call request to potentially emulate.
            handler: Async callback to execute the tool (can be called multiple times).

        Returns:
            ToolMessage with emulated response if tool should be emulated,
            otherwise calls handler for normal execution.
        """
        tool_name = request.tool_call["name"]

        # Check if this tool should be emulated
        should_emulate = self.emulate_all or tool_name in self.tools_to_emulate

        if not should_emulate:
            # Let it execute normally by calling the handler
            return await handler(request)

        # Extract tool information for emulation
        tool_args = request.tool_call["args"]
        tool_description = request.tool.description if request.tool else "No description available"

        # Build prompt for emulator LLM
        prompt = (
            f"You are emulating a tool call for testing purposes.\n\n"
            f"Tool: {tool_name}\n"
            f"Description: {tool_description}\n"
            f"Arguments: {tool_args}\n\n"
            f"Generate a realistic response that this tool would return "
            f"given these arguments.\n"
            f"Return ONLY the tool's output, no explanation or preamble. "
            f"Introduce variation into your responses."
        )

        # Get emulated response from LLM (using async invoke)
        response = await self.model.ainvoke([HumanMessage(prompt)])

        # Short-circuit: return emulated result without executing real tool
        return ToolMessage(
            content=response.content,
            tool_call_id=request.tool_call["id"],
            name=tool_name,
        )
