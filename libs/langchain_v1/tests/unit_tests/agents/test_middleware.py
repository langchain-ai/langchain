"""Tests for agent middleware behavior."""

from langchain.agents.factory import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models.fake import FakeListLLM


def test_duplicate_middleware_name_last_wins() -> None:
    """User-provided middleware should override auto-injected middleware."""
    llm = FakeListLLM(responses=["ok"])

    default_middleware = TodoListMiddleware(system_prompt="DEFAULT")
    custom_middleware = TodoListMiddleware(system_prompt="CUSTOM")

    agent = create_agent(
        model=llm,
        tools=[],
        middleware=[default_middleware, custom_middleware],
    )

    # Invoke once to force middleware execution
    agent.invoke({"input": "test"})

    # FakeListLLM records prompts it receives
    messages = llm.prompts[0]

    # Ensure the custom system prompt was used
    system_messages = [m.content for m in messages if m.type == "system"]

    if not any("CUSTOM" in content for content in system_messages):
        raise AssertionError("Expected custom TodoListMiddleware system prompt to be used")
