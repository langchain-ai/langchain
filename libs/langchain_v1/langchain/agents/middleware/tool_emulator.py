"""Tool emulator middleware for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage

from langchain.agents.middleware.types import AgentMiddleware
from langchain.chat_models.base import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Generator

    from langgraph.runtime import Runtime
    from langgraph.types import Command
    from langgraph.typing import ContextT

    from langchain.tools import BaseTool
    from langchain.tools.tool_node import ToolCallRequest


class ToolEmulator(AgentMiddleware):
    """Middleware that emulates specified tools using an LLM instead of executing them.

    This middleware allows selective emulation of tools for testing purposes.
    Only tools specified in tools_to_emulate will be emulated; others execute normally.

    Examples:
        Emulate specific tools by name:
        ```python
        from langchain.agents.middleware import ToolEmulator

        middleware = ToolEmulator(tools_to_emulate=["get_weather", "get_user_location"])

        agent = create_agent(
            model="openai:gpt-4o",
            tools=[get_weather, get_user_location, calculator],
            middleware=[middleware],
        )
        ```

        Use a custom model for emulation:
        ```python
        middleware = ToolEmulator(
            tools_to_emulate=["get_weather"],
            model="anthropic:claude-3-5-sonnet-latest"
        )
        ```

        Emulate all tools by passing tool instances:
        ```python
        middleware = ToolEmulator(tools_to_emulate=[get_weather, get_user_location])
        ```
    """

    def __init__(
        self,
        *,
        tools_to_emulate: list[str | BaseTool] | None = None,
        model: str | BaseChatModel | None = None,
    ) -> None:
        """Initialize the tool emulator.

        Args:
            tools_to_emulate: List of tool names (str) or BaseTool instances to emulate.
                If None or empty, no tools will be emulated (middleware does nothing).
            model: Model to use for emulation. Defaults to "anthropic:claude-3-5-sonnet-latest".
                Can be a model identifier string or BaseChatModel instance.
        """
        super().__init__()

        # Extract tool names from tools_to_emulate
        self.tools_to_emulate: set[str] = set()
        if tools_to_emulate:
            for tool in tools_to_emulate:
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

    def on_tool_call(
        self,
        request: ToolCallRequest,
        state: Any,  # noqa: ARG002
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> Generator[ToolCallRequest | ToolMessage | Command, ToolMessage | Command, None]:
        """Emulate tool execution using LLM if tool is in tools_to_emulate.

        Args:
            request: Tool call request to potentially emulate.
            state: Current agent state.
            runtime: LangGraph runtime.

        Yields:
            ToolMessage with emulated response if tool should be emulated,
            otherwise yields the original request for normal execution.
        """
        tool_name = request.tool_call["name"]

        # Check if this tool should be emulated
        if tool_name not in self.tools_to_emulate:
            # Let it execute normally by yielding the request
            yield request
            return

        # Extract tool information for emulation
        tool_args = request.tool_call["args"]
        tool_description = request.tool.description

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
        yield ToolMessage(
            content=response.content,
            tool_call_id=request.tool_call["id"],
            name=tool_name,
        )
