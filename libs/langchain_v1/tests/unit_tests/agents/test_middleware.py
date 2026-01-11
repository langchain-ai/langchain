"""Tests for agent middleware behavior."""

from langchain.agents.factory import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models.fake import FakeMessagesListChatModel


_EXPECT_OVERRIDE = "Expected custom TodoListMiddleware system prompt to be used"


def test_duplicate_middleware_name_last_wins() -> None:
    """User-provided middleware should override auto-injected middleware."""
    llm = FakeMessagesListChatModel(responses=["ok"])

    default_middleware = TodoListMiddleware(system_prompt="DEFAULT")
    custom_middleware = TodoListMiddleware(system_prompt="CUSTOM")

    agent = create_agent(
        model=llm,
        tools=[],
        middleware=[default_middleware, custom_middleware],
    )

    # Invoke once to force middleware execution
    agent.invoke({"input": "test"})

    # Inspect messages sent to the model
    messages = llm.messages[0]
    system_messages = [m.content for m in messages if m.type == "system"]

    if not any("CUSTOM" in content for content in system_messages):
        raise AssertionError(_EXPECT_OVERRIDE)
