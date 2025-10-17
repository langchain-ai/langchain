"""Unit tests for before_tool and after_tool middleware hooks."""

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    after_tool,
    before_tool,
)
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

from tests.unit_tests.agents.test_middleware_agent import FakeToolCallingModel


@tool
def sample_tool(query: str) -> str:
    """A sample tool for testing."""
    return f"Result for: {query}"


@tool
def error_tool() -> str:
    """A tool that always raises an error."""
    raise ValueError("Tool error")


class TestToolMiddlewareHooks:
    """Test before_tool and after_tool middleware hooks."""

    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.parametrize("hook_type", ["before", "after"])
    async def test_hook_execution(self, is_async: bool, hook_type: str) -> None:
        """Test that tool hooks are called in both sync and async modes."""
        execution_log: list[str] = []

        if hook_type == "before":

            @before_tool
            def log_hook(
                state: AgentState, runtime: Runtime, request
            ) -> dict[str, Any] | None:
                execution_log.append(f"{hook_type}_tool_called")
                execution_log.append(f"tool_name: {request.tool_call['name']}")
                return None
        else:

            @after_tool
            def log_hook(
                state: AgentState, runtime: Runtime, request, response
            ) -> dict[str, Any] | None:
                execution_log.append(f"{hook_type}_tool_called")
                execution_log.append(f"tool_name: {request.tool_call['name']}")
                return None

        # Model will call a tool once, then respond with final answer
        model = FakeToolCallingModel(
            tool_calls=[[{"name": "sample_tool", "args": {"query": "test"}, "id": "1"}], []]
        )
        agent = create_agent(model=model, tools=[sample_tool], middleware=[log_hook], checkpointer=InMemorySaver())

        agent.invoke({"messages": [HumanMessage("Hi")]}, {"configurable": {"thread_id": "test"}})

        assert f"{hook_type}_tool_called" in execution_log
        assert any("tool_name: sample_tool" in log for log in execution_log)

    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.parametrize("hook_type", ["before", "after"])
    async def test_hook_with_class_inheritance(self, is_async: bool, hook_type: str) -> None:
        """Test tool hooks using class inheritance in both sync and async modes."""
        execution_log: list[str] = []

        if is_async:

            class CustomMiddleware(AgentMiddleware):
                async def abefore_tool(
                    self, state: AgentState, runtime: Runtime, request
                ) -> dict[str, Any] | None:
                    if hook_type == "before":
                        execution_log.append("hook_called")
                        execution_log.append(f"tool: {request.tool_call['name']}")
                    return None

                async def aafter_tool(
                    self, state: AgentState, runtime: Runtime, request, response
                ) -> dict[str, Any] | None:
                    if hook_type == "after":
                        execution_log.append("hook_called")
                        execution_log.append(f"tool: {request.tool_call['name']}")
                    return None
        else:

            class CustomMiddleware(AgentMiddleware):
                def before_tool(
                    self, state: AgentState, runtime: Runtime, request
                ) -> dict[str, Any] | None:
                    if hook_type == "before":
                        execution_log.append("hook_called")
                        execution_log.append(f"tool: {request.tool_call['name']}")
                    return None

                def after_tool(
                    self, state: AgentState, runtime: Runtime, request, response
                ) -> dict[str, Any] | None:
                    if hook_type == "after":
                        execution_log.append("hook_called")
                        execution_log.append(f"tool: {request.tool_call['name']}")
                    return None

        middleware = CustomMiddleware()
        model = FakeToolCallingModel(
            tool_calls=[[{"name": "sample_tool", "args": {"query": "test"}, "id": "1"}]]
        )
        agent = create_agent(model=model, tools=[sample_tool], middleware=[middleware], checkpointer=InMemorySaver())

        if is_async:
            await agent.ainvoke({"messages": [HumanMessage("Test")]}, {"configurable": {"thread_id": "test"}})
        else:
            agent.invoke({"messages": [HumanMessage("Test")]}, {"configurable": {"thread_id": "test"}})

        assert "hook_called" in execution_log
        assert any("tool: sample_tool" in log for log in execution_log)


class TestToolHooksCombined:
    """Test before_tool and after_tool hooks working together."""

    @pytest.mark.parametrize("is_async", [False, True])
    async def test_execution_order(self, is_async: bool) -> None:
        """Test that before_tool executes before after_tool in both sync and async modes."""
        execution_log: list[str] = []

        if is_async:

            @before_tool
            async def log_before(state: AgentState, runtime: Runtime, request) -> None:
                execution_log.append("before")

            @after_tool
            async def log_after(state: AgentState, runtime: Runtime, request, response) -> None:
                execution_log.append("after")
        else:

            @before_tool
            def log_before(state: AgentState, runtime: Runtime, request) -> None:
                execution_log.append("before")

            @after_tool
            def log_after(state: AgentState, runtime: Runtime, request, response) -> None:
                execution_log.append("after")

        model = FakeToolCallingModel(
            tool_calls=[[{"name": "sample_tool", "args": {"query": "test"}, "id": "1"}]]
        )
        agent = create_agent(model=model, tools=[sample_tool], middleware=[log_before, log_after], checkpointer=InMemorySaver())

        if is_async:
            await agent.ainvoke({"messages": [HumanMessage("Test")]}, {"configurable": {"thread_id": "test"}})
        else:
            agent.invoke({"messages": [HumanMessage("Test")]}, {"configurable": {"thread_id": "test"}})

        # Verify before and after hooks are called for each tool invocation
        assert execution_log == ["before", "after"]

    def test_multiple_tool_calls(self) -> None:
        """Test that hooks are called for each tool invocation."""
        execution_log = []

        @before_tool
        def log_before(state: AgentState, runtime: Runtime, request) -> None:
            execution_log.append(f"before_{request.tool_call['name']}")

        @after_tool
        def log_after(state: AgentState, runtime: Runtime, request, response) -> None:
            execution_log.append(f"after_{request.tool_call['name']}")

        # Model will call tools twice with different names
        model = FakeToolCallingModel(
            tool_calls=[
                [{"name": "sample_tool", "args": {"query": "first"}, "id": "1"}],
                [{"name": "sample_tool", "args": {"query": "second"}, "id": "2"}],
            ]
        )

        agent = create_agent(
            model=model, tools=[sample_tool], middleware=[log_before, log_after], checkpointer=InMemorySaver()
        )
        agent.invoke({"messages": [HumanMessage("Test")]}, {"configurable": {"thread_id": "test"}})

        assert execution_log == [
            "before_sample_tool",
            "after_sample_tool",
            "before_sample_tool",
            "after_sample_tool",
        ]

    def test_tool_error_handling(self) -> None:
        """Test that after_tool hook still runs even when tool raises an error."""
        execution_log = []

        @before_tool
        def log_before(state: AgentState, runtime: Runtime, request) -> None:
            execution_log.append("before_error_tool")

        @after_tool
        def log_after(state: AgentState, runtime: Runtime, request, response) -> None:
            execution_log.append("after_error_tool")
            # Response should contain error information
            if hasattr(response, 'content'):
                execution_log.append(f"response_content: {response.content}")

        # Use a tool that always raises an error
        model = FakeToolCallingModel(
            tool_calls=[[{"name": "error_tool", "args": {}, "id": "1"}]]
        )

        agent = create_agent(
            model=model, tools=[error_tool], middleware=[log_before, log_after], checkpointer=InMemorySaver()
        )

        # Should not raise an exception due to error handling
        result = agent.invoke({"messages": [HumanMessage("Test")]}, {"configurable": {"thread_id": "test"}})

        # Both hooks should still be called
        assert "before_error_tool" in execution_log
        assert "after_error_tool" in execution_log

    def test_state_modification_in_before_tool(self) -> None:
        """Test that state modifications in before_tool are preserved."""
        execution_log = []

        @before_tool
        def add_metadata(state: AgentState, runtime: Runtime, request) -> dict[str, Any]:
            execution_log.append("before_tool_called")
            return {"tool_call_metadata": {"tool": request.tool_call["name"]}}

        @after_tool
        def check_metadata(state: AgentState, runtime: Runtime, request, response) -> None:
            execution_log.append("after_tool_called")
            if "tool_call_metadata" in state:
                execution_log.append(f"metadata_found: {state['tool_call_metadata']['tool']}")

        model = FakeToolCallingModel(
            tool_calls=[[{"name": "sample_tool", "args": {"query": "test"}, "id": "1"}]]
        )

        agent = create_agent(
            model=model, tools=[sample_tool], middleware=[add_metadata, check_metadata], checkpointer=InMemorySaver()
        )
        agent.invoke({"messages": [HumanMessage("Test")]}, {"configurable": {"thread_id": "test"}})

        assert "before_tool_called" in execution_log
        assert "after_tool_called" in execution_log
        assert "metadata_found: sample_tool" in execution_log

    def test_multiple_middleware_instances(self) -> None:
        """Test multiple before_tool and after_tool middleware instances."""
        execution_log = []

        @before_tool
        def before_one(state: AgentState, runtime: Runtime, request) -> None:
            execution_log.append("before_1")

        @before_tool
        def before_two(state: AgentState, runtime: Runtime, request) -> None:
            execution_log.append("before_2")

        @after_tool
        def after_one(state: AgentState, runtime: Runtime, request, response) -> None:
            execution_log.append("after_1")

        @after_tool
        def after_two(state: AgentState, runtime: Runtime, request, response) -> None:
            execution_log.append("after_2")

        model = FakeToolCallingModel(
            tool_calls=[[{"name": "sample_tool", "args": {"query": "test"}, "id": "1"}]]
        )

        agent = create_agent(
            model=model,
            tools=[sample_tool],
            middleware=[before_one, before_two, after_one, after_two],
            checkpointer=InMemorySaver(),
        )
        agent.invoke({"messages": [HumanMessage("Test")]}, {"configurable": {"thread_id": "test"}})

        # Verify composition order (first registered = outermost)
        assert execution_log == ["before_1", "before_2", "after_2", "after_1"]

    def test_response_access_in_after_tool(self) -> None:
        """Test that after_tool hooks can access the tool response."""
        execution_log = []
        tool_results = []

        @after_tool
        def capture_response(state: AgentState, runtime: Runtime, request, response) -> None:
            execution_log.append("after_tool_called")
            tool_results.append(response.content if hasattr(response, 'content') else str(response))

        model = FakeToolCallingModel(
            tool_calls=[[{"name": "sample_tool", "args": {"query": "test_query"}, "id": "1"}]]
        )

        agent = create_agent(
            model=model, tools=[sample_tool], middleware=[capture_response], checkpointer=InMemorySaver()
        )
        agent.invoke({"messages": [HumanMessage("Test")]}, {"configurable": {"thread_id": "test"}})

        assert "after_tool_called" in execution_log
        assert "Result for: test_query" in tool_results

    @pytest.mark.parametrize("is_async", [False, True])
    async def test_decorator_parameters(self, is_async: bool) -> None:
        """Test that decorator parameters work correctly."""
        execution_log = []

        if is_async:

            @before_tool(can_jump_to=["end"], name="CustomBeforeTool")
            async def custom_before(state: AgentState, runtime: Runtime, request) -> dict[str, Any]:
                execution_log.append("custom_before")
                return None

            @after_tool(state_schema=AgentState, name="CustomAfterTool")
            async def custom_after(state: AgentState, runtime: Runtime, request, response) -> None:
                execution_log.append("custom_after")
        else:

            @before_tool(can_jump_to=["end"], name="CustomBeforeTool")
            def custom_before(state: AgentState, runtime: Runtime, request) -> dict[str, Any]:
                execution_log.append("custom_before")
                return None

            @after_tool(state_schema=AgentState, name="CustomAfterTool")
            def custom_after(state: AgentState, runtime: Runtime, request, response) -> None:
                execution_log.append("custom_after")

        # Verify middleware names
        assert custom_before.name == "CustomBeforeTool"
        assert custom_after.name == "CustomAfterTool"

        model = FakeToolCallingModel(
            tool_calls=[[{"name": "sample_tool", "args": {"query": "test"}, "id": "1"}]]
        )

        agent = create_agent(
            model=model, tools=[sample_tool], middleware=[custom_before, custom_after], checkpointer=InMemorySaver()
        )

        if is_async:
            await agent.ainvoke({"messages": [HumanMessage("Test")]}, {"configurable": {"thread_id": "test"}})
        else:
            agent.invoke({"messages": [HumanMessage("Test")]}, {"configurable": {"thread_id": "test"}})

        assert "custom_before" in execution_log
        assert "custom_after" in execution_log