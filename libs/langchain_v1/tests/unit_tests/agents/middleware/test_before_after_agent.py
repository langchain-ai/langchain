"""Unit tests for before_agent and after_agent middleware hooks."""

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    after_agent,
    after_model,
    before_model,
    before_agent,
)
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime

from ..model import FakeToolCallingModel


@tool
def sample_tool(query: str) -> str:
    """A sample tool for testing."""
    return f"Result for: {query}"


class TestAgentMiddlewareHooks:
    """Test before_agent and after_agent middleware hooks."""

    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.parametrize("hook_type", ["before", "after"])
    async def test_hook_execution(self, is_async: bool, hook_type: str) -> None:
        """Test that agent hooks are called in both sync and async modes."""
        execution_log: list[str] = []

        if is_async:
            if hook_type == "before":

                @before_agent
                async def log_hook(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
                    execution_log.append(f"{hook_type}_agent_called")
                    execution_log.append(f"message_count: {len(state['messages'])}")
                    return None
            else:

                @after_agent
                async def log_hook(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
                    execution_log.append(f"{hook_type}_agent_called")
                    execution_log.append(f"message_count: {len(state['messages'])}")
                    return None
        else:
            if hook_type == "before":

                @before_agent
                def log_hook(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
                    execution_log.append(f"{hook_type}_agent_called")
                    execution_log.append(f"message_count: {len(state['messages'])}")
                    return None
            else:

                @after_agent
                def log_hook(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
                    execution_log.append(f"{hook_type}_agent_called")
                    execution_log.append(f"message_count: {len(state['messages'])}")
                    return None

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, tools=[], middleware=[log_hook])

        if is_async:
            await agent.ainvoke({"messages": [HumanMessage("Hi")]})
        else:
            agent.invoke({"messages": [HumanMessage("Hi")]})

        assert f"{hook_type}_agent_called" in execution_log
        assert any("message_count:" in log for log in execution_log)

    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.parametrize("hook_type", ["before", "after"])
    async def test_hook_with_class_inheritance(self, is_async: bool, hook_type: str) -> None:
        """Test agent hooks using class inheritance in both sync and async modes."""
        execution_log: list[str] = []

        if is_async:

            class CustomMiddleware(AgentMiddleware):
                async def abefore_agent(
                    self, state: AgentState, runtime: Runtime
                ) -> dict[str, Any] | None:
                    if hook_type == "before":
                        execution_log.append("hook_called")
                    return None

                async def aafter_agent(
                    self, state: AgentState, runtime: Runtime
                ) -> dict[str, Any] | None:
                    if hook_type == "after":
                        execution_log.append("hook_called")
                    return None
        else:

            class CustomMiddleware(AgentMiddleware):
                def before_agent(
                    self, state: AgentState, runtime: Runtime
                ) -> dict[str, Any] | None:
                    if hook_type == "before":
                        execution_log.append("hook_called")
                    return None

                def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
                    if hook_type == "after":
                        execution_log.append("hook_called")
                    return None

        middleware = CustomMiddleware()
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, tools=[], middleware=[middleware])

        if is_async:
            await agent.ainvoke({"messages": [HumanMessage("Test")]})
        else:
            agent.invoke({"messages": [HumanMessage("Test")]})

        assert "hook_called" in execution_log


class TestAgentHooksCombined:
    """Test before_agent and after_agent hooks working together."""

    @pytest.mark.parametrize("is_async", [False, True])
    async def test_execution_order(self, is_async: bool) -> None:
        """Test that before_agent executes before after_agent in both sync and async modes."""
        execution_log: list[str] = []

        if is_async:

            @before_agent
            async def log_before(state: AgentState, runtime: Runtime) -> None:
                execution_log.append("before")

            @after_agent
            async def log_after(state: AgentState, runtime: Runtime) -> None:
                execution_log.append("after")
        else:

            @before_agent
            def log_before(state: AgentState, runtime: Runtime) -> None:
                execution_log.append("before")

            @after_agent
            def log_after(state: AgentState, runtime: Runtime) -> None:
                execution_log.append("after")

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, tools=[], middleware=[log_before, log_after])

        if is_async:
            await agent.ainvoke({"messages": [HumanMessage("Test")]})
        else:
            agent.invoke({"messages": [HumanMessage("Test")]})

        assert execution_log == ["before", "after"]

    def test_state_passthrough(self) -> None:
        """Test that state modifications in before_agent are visible to after_agent."""

        @before_agent
        def modify_in_before(state: AgentState, runtime: Runtime) -> dict[str, Any]:
            return {"messages": [HumanMessage("Added by before_agent")]}

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(model=model, tools=[], middleware=[modify_in_before])
        result = agent.invoke({"messages": [HumanMessage("Original")]})

        message_contents = [msg.content for msg in result["messages"]]
        assert message_contents[1] == "Added by before_agent"

    def test_multiple_middleware_instances(self) -> None:
        """Test multiple before_agent and after_agent middleware instances."""
        execution_log = []

        @before_agent
        def before_one(state: AgentState, runtime: Runtime) -> None:
            execution_log.append("before_1")

        @before_agent
        def before_two(state: AgentState, runtime: Runtime) -> None:
            execution_log.append("before_2")

        @after_agent
        def after_one(state: AgentState, runtime: Runtime) -> None:
            execution_log.append("after_1")

        @after_agent
        def after_two(state: AgentState, runtime: Runtime) -> None:
            execution_log.append("after_2")

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))
        agent = create_agent(
            model=model, tools=[], middleware=[before_one, before_two, after_one, after_two]
        )
        agent.invoke({"messages": [HumanMessage("Test")]})

        assert execution_log == ["before_1", "before_2", "after_2", "after_1"]

    def test_agent_hooks_run_once_with_multiple_model_calls(self) -> None:
        """Test that before_agent and after_agent run only once per thread.

        This test verifies that agent-level hooks (before_agent, after_agent) execute
        exactly once per agent invocation, regardless of how many tool calling loops occur.
        This is different from model-level hooks (before_model, after_model) which run
        on every model invocation within the tool calling loop.
        """
        execution_log = []

        @before_agent
        def log_before_agent(state: AgentState, runtime: Runtime) -> None:
            execution_log.append("before_agent")

        @before_model
        def log_before_model(state: AgentState, runtime: Runtime) -> None:
            execution_log.append("before_model")

        @after_agent
        def log_after_agent(state: AgentState, runtime: Runtime) -> None:
            execution_log.append("after_agent")

        @after_model
        def log_after_model(state: AgentState, runtime: Runtime) -> None:
            execution_log.append("after_model")

        # Model will call a tool twice, then respond with final answer
        # This creates 3 model invocations total, but agent hooks should still run once
        model = FakeToolCallingModel(
            tool_calls=[
                [{"name": "sample_tool", "args": {"query": "first"}, "id": "1"}],
                [{"name": "sample_tool", "args": {"query": "second"}, "id": "2"}],
                [],  # Third call returns no tool calls (final answer)
            ]
        )

        agent = create_agent(
            model=model,
            tools=[sample_tool],
            middleware=[log_before_agent, log_before_model, log_after_model, log_after_agent],
        )

        agent.invoke(
            {"messages": [HumanMessage("Test")]}, config={"configurable": {"thread_id": "abc"}}
        )

        assert execution_log == [
            "before_agent",
            "before_model",
            "after_model",
            "before_model",
            "after_model",
            "before_model",
            "after_model",
            "after_agent",
        ]

        agent.invoke(
            {"messages": [HumanMessage("Test")]}, config={"configurable": {"thread_id": "abc"}}
        )

        assert execution_log == [
            "before_agent",
            "before_model",
            "after_model",
            "before_model",
            "after_model",
            "before_model",
            "after_model",
            "after_agent",
            "before_agent",
            "before_model",
            "after_model",
            "before_model",
            "after_model",
            "before_model",
            "after_model",
            "after_agent",
        ]
