"""Base prompt templates and common system messages for agents."""
from __future__ import annotations

EXPERT_SYSTEM_PROMPT = (
    "You are an expert content strategist and copywriter. Produce concise, high-quality, "
    "actionable outputs suitable for social media platforms. Be factual, avoid hallucination, "
    "and when unsure, say you don't know."
)


def wrap_user_prompt(task_description: str, context: str | None = None) -> str:
    """Return a full prompt combining system instructions, context and task.

    Args:
        task_description: Specific instructions for the agent.
        context: Optional additional content (transcript, metadata).
    """
    base = EXPERT_SYSTEM_PROMPT + "\n\nTask:\n" + task_description
    if context:
        base += "\n\nContext:\n" + context
    return base
