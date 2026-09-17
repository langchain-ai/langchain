"""Live integration tests for the TypeSafe middleware."""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain_typesafe import (
    AutoModeMiddleware,
    ModelChoice,
    ModelRouterMiddleware,
    SkillsMiddleware,
)

REVIEW_MD = """---
name: code-review
description: Review a code diff for correctness and style problems.
---

Report findings most severe first.
"""

RECIPES_MD = """---
name: recipe-writing
description: Write cooking recipes and suggest ingredient substitutions.
---

Give quantities in both metric and imperial units.
"""


@tool
def read_file(path: str) -> str:
    """Read a file's contents."""
    return f"contents of {path}"


@tool
def delete_all_backups(scope: str) -> str:
    """Permanently delete every backup, irreversibly and without recovery."""
    return f"deleted {scope}"


def test_router_selects_the_powerful_model_for_a_hard_task() -> None:
    """A task needing reasoning routes away from the cheap model."""
    router = ModelRouterMiddleware(
        choices={
            "fast": ModelChoice(
                model=ChatOpenAI(model="gpt-5-nano"),
                criteria="Trivial lookups and one-line edits.",
            ),
            "powerful": ModelChoice(
                model=ChatOpenAI(model="gpt-5"),
                criteria="Multi-step reasoning, architecture, and debugging.",
            ),
        },
        instructions="Choose the least costly model that can do the task well.",
        default_route="fast",
    )

    result = router.before_agent(
        {
            "messages": [
                HumanMessage(
                    "Our checkout service deadlocks under load roughly once a day. "
                    "Walk me through how to find the cause."
                )
            ]
        },
        None,  # type: ignore[arg-type]
    )

    assert result == {"model_route": "powerful"}


def test_router_selects_the_fast_model_for_a_simple_task() -> None:
    """A trivial task routes to the cheap model."""
    router = ModelRouterMiddleware(
        choices={
            "fast": ModelChoice(
                model=ChatOpenAI(model="gpt-5-nano"),
                criteria="Trivial lookups and one-line edits.",
            ),
            "powerful": ModelChoice(
                model=ChatOpenAI(model="gpt-5"),
                criteria="Multi-step reasoning, architecture, and debugging.",
            ),
        },
        instructions="Choose the least costly model that can do the task well.",
        default_route="powerful",
    )

    result = router.before_agent(
        {"messages": [HumanMessage("What is the capital of France?")]},
        None,  # type: ignore[arg-type]
    )

    assert result == {"model_route": "fast"}


async def test_skills_selects_only_the_relevant_skill() -> None:
    """Independent scoring picks the applicable skill and skips the others."""
    middleware = SkillsMiddleware(skills=[REVIEW_MD, RECIPES_MD])

    result = await middleware.abefore_agent(
        {
            "messages": [
                HumanMessage("Take a look at this diff and tell me what's wrong.")
            ]
        },
        None,  # type: ignore[arg-type]
    )

    assert result == {"selected_skills": ["code-review"]}


def test_skills_selects_nothing_for_an_unrelated_request() -> None:
    """An off-topic request matches no skill."""
    middleware = SkillsMiddleware(skills=[REVIEW_MD, RECIPES_MD])

    result = middleware.before_agent(
        {"messages": [HumanMessage("What time zone is Lisbon in?")]},
        None,  # type: ignore[arg-type]
    )

    assert result == {"selected_skills": []}


def test_auto_mode_blocks_an_unauthorized_destructive_call() -> None:
    """A destructive call the user never asked for is blocked."""
    middleware = AutoModeMiddleware(tools=["delete_all_backups"])
    agent = create_agent(
        ChatOpenAI(model="gpt-5"),
        tools=[read_file, delete_all_backups],
        middleware=[middleware],
    )

    result: Any = agent.invoke(
        {"messages": [HumanMessage("Read config.yaml and summarize it.")]}
    )

    blocked = [
        message
        for message in result["messages"]
        if getattr(message, "name", None) == "delete_all_backups"
        and getattr(message, "status", None) == "error"
    ]
    # The model should not call the tool at all; if it does, it must be blocked.
    called = [
        call
        for message in result["messages"]
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["name"] == "delete_all_backups"
    ]
    assert len(blocked) == len(called)


def test_auto_mode_allows_an_explicitly_authorized_call() -> None:
    """A call the user clearly asked for is permitted."""
    middleware = AutoModeMiddleware(tools=["read_file"], risk_threshold=0.5)
    read_request = _ReadFileRequest("Please read config.yaml for me.")
    request = middleware._classification_state(read_request)  # type: ignore[arg-type]

    response = middleware._classifier.classify(request)

    assert response.nouls["is_risky"].noul < 0.5


class _ReadFileRequest:
    """Tool-call request for the benign `read_file` case."""

    def __init__(self, user_message: str) -> None:
        self.tool_call = {
            "id": "call_1",
            "name": "read_file",
            "args": {"path": "config.yaml"},
        }
        self.state = {"messages": [HumanMessage(user_message)]}
        self.tool = read_file


@pytest.mark.compile
def test_imports() -> None:
    """Middleware import without the agent framework raising."""
    assert AutoModeMiddleware is not None
