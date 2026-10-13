"""A server that needs input mid-call, answered by a human.

Some MCP tools cannot finish without asking something. `MCPAdapter` surfaces the
question as a LangGraph `interrupt()`, so the person already reviewing the
agent's work answers it and the run resumes.

The adapter arms every client it builds to advertise the elicitation capability,
so this works without any opt-in — a server only asks a client that made the
promise on the wire, and the adapter makes it for you.

    uv run examples/mcp/elicitation.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from _servers import booking_server
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter


async def main() -> None:
    """Book a table, answering the server's question from here."""
    async with MCPAdapter(booking_server()) as adapter:
        tools = await adapter.list_tools()

    # Resuming needs persistence, so the interrupted run has somewhere to wait.
    agent = create_agent("anthropic:claude-sonnet-5", tools, checkpointer=InMemorySaver())
    config: Any = {"configurable": {"thread_id": "booking-1"}}

    paused = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Book a table for 4."}]}, config
    )

    [interrupt] = paused["__interrupt__"]
    [question] = interrupt.value["requests"]
    print(f"server asks ({question['mode']}): {question['message']}")

    # Answers are keyed by the server's own request key, so nothing has to be
    # tracked across the pause. `decline` or `cancel` would refuse instead.
    answer = {"action": "accept", "content": {"date": "2026-09-14"}}
    resumed = await agent.ainvoke(Command(resume={"responses": {question["key"]: answer}}), config)

    print("agent:", resumed["messages"][-1].text)


if __name__ == "__main__":
    asyncio.run(main())
