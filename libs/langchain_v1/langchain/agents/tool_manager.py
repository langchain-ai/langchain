"""Dynamic tool management for agents.

Provides the `DynamicToolManager` middleware that enables adding and removing
tools after agent creation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, StructuredTool

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langgraph.prebuilt.tool_node import ToolNode


class DynamicToolManager(AgentMiddleware):
    """Middleware that supports dynamic tool addition and removal after agent creation.

    This middleware is automatically included by `create_agent()` and enables
    adding or removing tools at any point after the agent is created. Dynamic
    tools are injected into model requests and registered with the tool node
    for execution.

    .. note::
        The agent must be created with at least one tool for dynamic tool
        management to work, since the tool execution node is only added to the
        graph when tools are present at creation time.

    Examples:
        ```python
        agent = create_agent(model="openai:gpt-4o", tools=[static_tool])

        # Add a tool after creation
        agent.add_tool(my_new_tool)

        # Remove a tool
        agent.remove_tool("my_new_tool")

        # Batch operations
        agent.add_tools([tool_a, tool_b])
        agent.remove_tools(["tool_a", "tool_b"])
        ```
    """

    def __init__(self) -> None:
        """Initialize with an empty dynamic tool set."""
        self._tools: dict[str, BaseTool] = {}
        self._tool_node: ToolNode | None = None

    def set_tool_node(self, tool_node: ToolNode | None) -> None:
        """Set the ToolNode reference for direct tool registration.

        Called by `create_agent()` to provide the ToolNode instance so that
        dynamically added tools can be registered for execution.

        Args:
            tool_node: The ToolNode instance from the compiled graph.
        """
        self._tool_node = tool_node

    @property
    def name(self) -> str:
        """The name of this middleware instance."""
        return "_dynamic_tool_manager"

    @property
    def dynamic_tools(self) -> list[BaseTool]:
        """Return the list of currently registered dynamic tools."""
        return list(self._tools.values())

    @property
    def dynamic_tool_names(self) -> set[str]:
        """Return the set of currently registered dynamic tool names."""
        return set(self._tools.keys())

    def add_tool(self, tool: BaseTool | Callable[..., Any]) -> None:
        """Add a tool dynamically.

        Args:
            tool: A `BaseTool` instance or a callable to convert into a tool.

        Raises:
            RuntimeError: If no tool node is available (agent created with no tools).
        """
        if not isinstance(tool, BaseTool):
            tool = StructuredTool.from_function(tool)
        if self._tool_node is None:
            msg = (
                "Cannot add dynamic tools to an agent created without any tools. "
                "Pass at least one tool to create_agent() to enable dynamic tool "
                "management."
            )
            raise RuntimeError(msg)
        self._tools[tool.name] = tool
        self._tool_node.tools_by_name[tool.name] = tool

    def remove_tool(self, tool_name: str) -> None:
        """Remove a dynamically added tool by name.

        Args:
            tool_name: The name of the tool to remove.

        Raises:
            ValueError: If the tool name is not found in dynamic tools.
        """
        if tool_name not in self._tools:
            available = sorted(self._tools.keys())
            msg = (
                f"Tool '{tool_name}' not found in dynamic tools. "
                f"Available dynamic tools: {available}"
            )
            raise ValueError(msg)
        del self._tools[tool_name]
        if self._tool_node is not None:
            self._tool_node.tools_by_name.pop(tool_name, None)

    def add_tools(self, tools: Sequence[BaseTool | Callable[..., Any]]) -> None:
        """Add multiple tools dynamically.

        Args:
            tools: A sequence of `BaseTool` instances or callables.

        Raises:
            RuntimeError: If no tool node is available (agent created with no tools).
        """
        for t in tools:
            self.add_tool(t)

    def remove_tools(self, tool_names: Sequence[str]) -> None:
        """Remove multiple dynamically added tools by name.

        Args:
            tool_names: A sequence of tool names to remove.

        Raises:
            ValueError: If any tool name is not found in dynamic tools.
        """
        for name in tool_names:
            self.remove_tool(name)

    # -- Middleware hooks --

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelCallResult:
        """Add dynamic tools to the model request before execution."""
        if self._tools:
            updated = request.override(tools=[*request.tools, *self._tools.values()])
            return handler(updated)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelCallResult:
        """Add dynamic tools to the model request before async execution."""
        if self._tools:
            updated = request.override(tools=[*request.tools, *self._tools.values()])
            return await handler(updated)
        return await handler(request)
