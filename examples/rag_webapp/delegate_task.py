"""Simple helper to delegate tasks to Agent Zero."""

from typing import Any

try:  # pragma: no cover - optional dependency
    from agent_zero.core.agent import Agent
except Exception:  # pragma: no cover
    Agent = None


def delegate_task(description: str) -> Any:
    """Run a task using Agent Zero if available."""
    if Agent is None:
        raise RuntimeError("agent-zero not installed")
    agent = Agent(
        model=None,
        history=[],
        name="task-agent",
        call_id="cli",
        conversation_knowledge={},
    )
    if hasattr(agent, "llm"):
        return agent.llm.invoke(description)
    return description
