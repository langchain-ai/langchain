"""Tests for agent middleware behavior."""

from langchain.agents.factory import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models.fake import FakeListLLM



_EXPECT_SINGLE_MIDDLEWARE = "Expected exactly one TodoListMiddleware instance"
_EXPECT_OVERRIDE = "Expected user middleware to override default"


def test_duplicate_middleware_name_last_wins() -> None:
    """User-provided middleware should override auto-injected middleware."""
    llm = FakeListLLM(responses=["ok"])

    default_middleware = TodoListMiddleware(system_prompt="default")
    custom_middleware = TodoListMiddleware(system_prompt="custom")

    agent = create_agent(
        model=llm,
        tools=[],
        middleware=[default_middleware, custom_middleware],
    )

    todos = [m for m in agent.middleware if m.name == default_middleware.name]

    if len(todos) != 1:
        raise AssertionError(_EXPECT_SINGLE_MIDDLEWARE)

    if todos[0].system_prompt != "custom":
        raise AssertionError(_EXPECT_OVERRIDE)
