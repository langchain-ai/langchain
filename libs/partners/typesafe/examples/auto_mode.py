"""Gate risky tool calls with TypeSafe.

Run against the live API:

    export TYPESAFE_API_KEY=... OPENAI_API_KEY=...
    uv run python examples/auto_mode.py
"""

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain_typesafe import AutoModeMiddleware


@tool
def read_file(path: str) -> str:
    """Read a file's contents."""
    return f"contents of {path}"


@tool
def delete_all_backups(scope: str) -> str:
    """Permanently delete every backup, irreversibly and without recovery."""
    return f"deleted {scope}"


def main() -> None:
    """Run an agent whose destructive tool is gated, and print each tool result."""
    agent = create_agent(
        ChatOpenAI(model="gpt-5"),
        tools=[read_file, delete_all_backups],
        middleware=[AutoModeMiddleware(tools=["delete_all_backups"])],
    )

    result: Any = agent.invoke(
        {
            "messages": [
                HumanMessage("Read config.yaml, then wipe every backup we have.")
            ]
        }
    )

    for message in result["messages"]:
        name = getattr(message, "name", None)
        if name in {"read_file", "delete_all_backups"}:
            status = getattr(message, "status", "success")
            print(f"{name}: {status} -> {message.content}")


if __name__ == "__main__":
    main()
