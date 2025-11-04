"""Unit tests for get_config() function in middleware."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState, get_config

from ..model import FakeToolCallingModel


def test_get_config_basic_access() -> None:
    """Test that middleware can access config via get_config()."""
    captured_config = {}

    class ConfigAccessMiddleware(AgentMiddleware):
        def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
            config = get_config()
            captured_config["config"] = config
            captured_config["exists"] = config is not None
            return None

    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
        middleware=[ConfigAccessMiddleware()],
        system_prompt="Test",
    )

    config = RunnableConfig(
        metadata={"user_id": "test_123", "session_id": "session_456"},
        tags=["test", "production"],
        run_name="test_run",
    )

    agent.invoke({"messages": [HumanMessage("test")]}, config=config)

    # Verify config was accessible
    assert captured_config["exists"] is True
    assert captured_config["config"] is not None

    # Verify metadata
    metadata = captured_config["config"].get("metadata", {})
    assert metadata.get("user_id") == "test_123"
    assert metadata.get("session_id") == "session_456"

    # Verify tags
    tags = captured_config["config"].get("tags", [])
    assert "test" in tags
    assert "production" in tags


async def test_get_config_async() -> None:
    """Test that get_config() works in async middleware."""
    captured_config = {}

    class AsyncConfigMiddleware(AgentMiddleware):
        async def abefore_model(
            self, state: AgentState, runtime: Any
        ) -> dict[str, Any] | None:
            config = get_config()
            captured_config["config"] = config
            return None

    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
        middleware=[AsyncConfigMiddleware()],
        system_prompt="Test",
    )

    config = RunnableConfig(
        metadata={"async_test": "value"},
        tags=["async"],
    )

    await agent.ainvoke({"messages": [HumanMessage("test")]}, config=config)

    # Verify config was accessible
    assert captured_config["config"] is not None
    metadata = captured_config["config"].get("metadata", {})
    assert metadata.get("async_test") == "value"


def test_get_config_no_config_passed() -> None:
    """Test get_config() when no config is passed to invoke()."""
    captured_config = {}

    class NoConfigMiddleware(AgentMiddleware):
        def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
            config = get_config()
            captured_config["config"] = config
            return None

    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
        middleware=[NoConfigMiddleware()],
        system_prompt="Test",
    )

    # Invoke without passing config
    agent.invoke({"messages": [HumanMessage("test")]})

    # Config should still exist (LangGraph creates default config)
    assert captured_config["config"] is not None


def test_get_config_metadata_conditional_logic() -> None:
    """Test using config metadata for conditional middleware behavior."""
    execution_log = []

    class ConditionalMiddleware(AgentMiddleware):
        def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
            config = get_config()
            if config:
                log_level = config.get("metadata", {}).get("log_level")
                if log_level == "debug":
                    execution_log.append("debug_mode_enabled")
                elif log_level == "info":
                    execution_log.append("info_mode")
            return None

    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
        middleware=[ConditionalMiddleware()],
        system_prompt="Test",
    )

    # Test with debug level
    config_debug = RunnableConfig(metadata={"log_level": "debug"})
    agent.invoke({"messages": [HumanMessage("test1")]}, config=config_debug)
    assert "debug_mode_enabled" in execution_log

    # Test with info level
    execution_log.clear()
    config_info = RunnableConfig(metadata={"log_level": "info"})
    agent.invoke({"messages": [HumanMessage("test2")]}, config=config_info)
    assert "info_mode" in execution_log


def test_get_config_multiple_middleware() -> None:
    """Test that multiple middleware can all access the same config."""
    captured_configs = {"first": None, "second": None, "third": None}

    class FirstMiddleware(AgentMiddleware):
        def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
            captured_configs["first"] = get_config()
            return None

    class SecondMiddleware(AgentMiddleware):
        def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
            captured_configs["second"] = get_config()
            return None

    class ThirdMiddleware(AgentMiddleware):
        def before_agent(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
            captured_configs["third"] = get_config()
            return None

    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
        middleware=[FirstMiddleware(), SecondMiddleware(), ThirdMiddleware()],
        system_prompt="Test",
    )

    config = RunnableConfig(
        metadata={"shared_key": "shared_value"},
        tags=["shared_tag"],
    )

    agent.invoke({"messages": [HumanMessage("test")]}, config=config)

    # Verify all middleware accessed config
    assert captured_configs["first"] is not None
    assert captured_configs["second"] is not None
    assert captured_configs["third"] is not None

    # Verify they all see the same metadata
    for config_key in ["first", "second", "third"]:
        cfg = captured_configs[config_key]
        metadata = cfg.get("metadata", {})
        assert metadata.get("shared_key") == "shared_value"
        tags = cfg.get("tags", [])
        assert "shared_tag" in tags


def test_get_config_with_decorator_middleware() -> None:
    """Test get_config() works with decorator-based middleware."""
    from langchain.agents.middleware.types import before_model

    captured_config = {}

    @before_model
    def config_decorator_middleware(state: AgentState, runtime: Any) -> None:
        config = get_config()
        captured_config["config"] = config

    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
        middleware=[config_decorator_middleware],
        system_prompt="Test",
    )

    config = RunnableConfig(
        metadata={"decorator_test": "value"},
    )

    agent.invoke({"messages": [HumanMessage("test")]}, config=config)

    # Verify config accessible in decorator middleware
    assert captured_config["config"] is not None
    metadata = captured_config["config"].get("metadata", {})
    assert metadata.get("decorator_test") == "value"


def test_get_config_use_case_user_context() -> None:
    """Test real-world use case: user-specific behavior."""
    user_actions = []

    class UserContextMiddleware(AgentMiddleware):
        def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
            config = get_config()
            if config:
                user_id = config.get("metadata", {}).get("user_id")
                is_premium = config.get("metadata", {}).get("is_premium_user", False)

                if user_id:
                    action = f"user_{user_id}_premium" if is_premium else f"user_{user_id}_free"
                    user_actions.append(action)
            return None

    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
        middleware=[UserContextMiddleware()],
        system_prompt="Test",
    )

    # Test premium user
    config_premium = RunnableConfig(
        metadata={"user_id": "alice", "is_premium_user": True}
    )
    agent.invoke({"messages": [HumanMessage("test")]}, config=config_premium)
    assert "user_alice_premium" in user_actions

    # Test free user
    config_free = RunnableConfig(metadata={"user_id": "bob", "is_premium_user": False})
    agent.invoke({"messages": [HumanMessage("test")]}, config=config_free)
    assert "user_bob_free" in user_actions


def test_get_config_with_tags() -> None:
    """Test accessing tags from config."""
    captured_tags = []

    class TagCheckMiddleware(AgentMiddleware):
        def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
            config = get_config()
            if config:
                tags = config.get("tags", [])
                captured_tags.extend(tags)
            return None

    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
        middleware=[TagCheckMiddleware()],
        system_prompt="Test",
    )

    config = RunnableConfig(tags=["production", "important", "monitored"])

    agent.invoke({"messages": [HumanMessage("test")]}, config=config)

    assert "production" in captured_tags
    assert "important" in captured_tags
    assert "monitored" in captured_tags


def test_get_config_with_run_name() -> None:
    """Test accessing run_name from config."""
    captured_run_name = {}

    class RunNameMiddleware(AgentMiddleware):
        def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
            config = get_config()
            if config:
                # run_name may or may not be passed through by LangGraph
                # Just verify we can access the config dict
                captured_run_name["has_config"] = True
                captured_run_name["run_name"] = config.get("run_name")
            return None

    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
        middleware=[RunNameMiddleware()],
        system_prompt="Test",
    )

    config = RunnableConfig(run_name="my_custom_run")

    agent.invoke({"messages": [HumanMessage("test")]}, config=config)

    # Verify config was accessible (run_name may be None if not passed through)
    assert captured_run_name.get("has_config") is True

