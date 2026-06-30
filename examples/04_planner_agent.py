"""Planner agent using `TodoListMiddleware` (one-shot).

Demonstrates middleware: `TodoListMiddleware` gives the agent a `write_todos` tool, so
it decomposes a multi-step goal into an explicit checklist and tracks progress as it
works. The resulting plan is read back from `result["todos"]`.

Run from `libs/langchain_v1`:

    uv run --with langchain-openai python ../../examples/04_planner_agent.py

Requires `OPENAI_API_KEY` in the environment.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

MODEL = "openai:gpt-4o-mini"


@tool
def web_search(query: str) -> str:
    """Look up information on the web (stubbed for this example).

    Args:
        query: The search query.

    Returns:
        A canned search-result snippet.
    """
    return f"[stub search results for {query!r}]"


def build_agent() -> CompiledStateGraph:
    """Build an agent that plans with a todo list before acting."""
    model = init_chat_model(MODEL)
    return create_agent(
        model,
        tools=[web_search],
        middleware=[TodoListMiddleware()],
    )


def main() -> None:
    """Give the agent a multi-step goal and print its plan plus the final answer."""
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY before running this example.")

    agent = build_agent()
    goal = (
        "Plan a two-part blog series introducing LangChain agents: first research "
        "the key concepts, then draft a section outline for each part."
    )
    print(f"Goal: {goal}\n")
    result = agent.invoke({"messages": [{"role": "user", "content": goal}]})

    todos = result.get("todos")
    if todos:
        print("Plan (todos):")
        for i, todo in enumerate(todos, start=1):
            status = todo.get("status", "?") if isinstance(todo, dict) else "?"
            content = todo.get("content", todo) if isinstance(todo, dict) else todo
            print(f"  {i}. [{status}] {content}")
        print()

    print(f"Final answer:\n{result['messages'][-1].content}")


if __name__ == "__main__":
    main()
