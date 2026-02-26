"""Utilities for wrapping agents as tools, enabling nested agent loop patterns."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import CallbackManagerForToolRun  # noqa: TC002
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from pydantic import Field

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


class _AgentTool(BaseTool):
    """A tool that wraps a compiled agent graph, creating a nested agent loop.

    When invoked by an outer agent, this tool runs the inner agent's full
    loop (prompt -> model -> tools -> ... -> response) and returns the final
    AI message content as the tool result.
    """

    agent: Any = Field(exclude=True)
    thread_id_prefix: str = ""
    _invocation_count: int = 0

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,  # noqa: ARG002
        **_kwargs: Any,
    ) -> str:
        self._invocation_count += 1
        config: dict[str, Any] = {}
        if self.thread_id_prefix:
            config["configurable"] = {
                "thread_id": f"{self.thread_id_prefix}-{self._invocation_count}"
            }

        result = self.agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config or None,
        )
        return _extract_response(result)

    async def _arun(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,  # noqa: ARG002
        **_kwargs: Any,
    ) -> str:
        self._invocation_count += 1
        config: dict[str, Any] = {}
        if self.thread_id_prefix:
            config["configurable"] = {
                "thread_id": f"{self.thread_id_prefix}-{self._invocation_count}"
            }

        result = await self.agent.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config=config or None,
        )
        return _extract_response(result)


def _extract_response(result: dict[str, Any]) -> str:
    """Extract the final AI response text from an agent result."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = [
                    part if isinstance(part, str) else part.get("text", "") for part in content
                ]
                return "".join(text_parts)
    return ""


def create_agent_tool(
    agent: CompiledStateGraph,
    *,
    name: str,
    description: str,
    thread_id_prefix: str = "",
) -> BaseTool:
    """Create a tool from a compiled agent graph for nested agent loop patterns.

    This wraps a compiled agent (the output of
    [`create_agent`][langchain.agents.create_agent]) as a
    [`BaseTool`][langchain_core.tools.BaseTool] that can be passed to another agent.
    When the outer agent invokes this tool, the inner agent runs its full loop
    (model -> tools -> model -> ... -> final response), creating a **loop-in-loop**
    pattern where agents can delegate subtasks to specialized sub-agents.

    !!! warning "Experimental"

        This feature is experimental and may change in future releases.

    Args:
        agent: A compiled agent graph, typically produced by `create_agent`.
        name: The tool name visible to the outer agent.
        description: A description of what the sub-agent does. The outer agent
            uses this to decide when to delegate work to the sub-agent.
        thread_id_prefix: Optional prefix for generating unique thread IDs per
            invocation. When set together with a checkpointer on the inner agent,
            each invocation gets its own conversation thread.

    Returns:
        A tool that, when called with a `query` string, runs the inner agent
        and returns its final AI response.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.tool import create_agent_tool

        # Create a specialized research sub-agent
        researcher = create_agent(
            model="openai:gpt-4o",
            tools=[web_search, summarize],
            system_prompt="You are a research assistant.",
        )

        # Wrap it as a tool for the orchestrator
        research_tool = create_agent_tool(
            researcher,
            name="research",
            description="Delegate research tasks to a specialized research agent.",
        )

        # Create an orchestrator that can use the researcher
        orchestrator = create_agent(
            model="openai:gpt-4o",
            tools=[research_tool, write_report],
            system_prompt="You coordinate research and writing tasks.",
        )
        ```
    """
    return _AgentTool(
        name=name,
        description=description,
        agent=agent,
        thread_id_prefix=thread_id_prefix,
    )
