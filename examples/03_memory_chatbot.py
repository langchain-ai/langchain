"""Interactive chatbot with conversation memory (multi-turn).

Demonstrates `checkpointer`: with an `InMemorySaver` and a fixed `thread_id`, the
agent remembers earlier turns in the same conversation. Send a couple of messages,
then ask about something from your first message to see the memory work. Type `exit`
(or `quit`) to stop.

Run from `libs/langchain_v1`:

    uv run --with langchain-openai python ../../examples/03_memory_chatbot.py

Requires `OPENAI_API_KEY` in the environment.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

MODEL = "openai:gpt-4o-mini"
THREAD_ID = "demo-conversation"


def build_agent() -> CompiledStateGraph:
    """Build a chat agent whose memory is backed by an in-memory checkpointer."""
    model = init_chat_model(MODEL)
    return create_agent(
        model,
        tools=[],
        system_prompt="You are a concise, friendly assistant.",
        checkpointer=InMemorySaver(),
    )


def main() -> None:
    """Run an interactive chat loop that remembers prior turns on one thread."""
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY before running this example.")

    agent = build_agent()
    # A fixed thread_id ties every turn to the same saved conversation history.
    config = {"configurable": {"thread_id": THREAD_ID}}
    print("Chat with memory. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        print(f"bot> {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
