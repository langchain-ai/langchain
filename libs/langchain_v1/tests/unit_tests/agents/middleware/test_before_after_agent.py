"""Unit tests for before_agent and after_agent middleware hooks."""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    after_agent,
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


class TestBeforeAgentBasic:
    """Test basic before_agent functionality."""

    def test_sync_before_agent_execution(self) -> None:
        """Test that before_agent hook is called synchronously."""
        execution_log = []

        @before_agent
        def log_before_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            execution_log.append("before_agent_called")
            execution_log.append(f"message_count: {len(state['messages'])}")
            return None

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))

        agent = create_agent(model=model, tools=[], middleware=[log_before_agent])

        agent.invoke({"messages": [HumanMessage("Hi")]})

        assert "before_agent_called" in execution_log
        assert "message_count: 1" in execution_log

    async def test_async_before_agent_execution(self) -> None:
        """Test that before_agent hook is called asynchronously."""
        execution_log = []

        @before_agent
        async def async_log_before_agent(
            state: AgentState, runtime: Runtime
        ) -> dict[str, Any] | None:
            execution_log.append("async_before_agent_called")
            execution_log.append(f"message_count: {len(state['messages'])}")
            return None

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello!")]))

        agent = create_agent(model=model, tools=[], middleware=[async_log_before_agent])

        await agent.ainvoke({"messages": [HumanMessage("Hi")]})

        assert "async_before_agent_called" in execution_log
        assert "message_count: 1" in execution_log

    def test_before_agent_state_modification(self) -> None:
        """Test that before_agent can modify state."""

        @before_agent
        def add_metadata(state: AgentState, runtime: Runtime) -> dict[str, Any]:
            return {"messages": [HumanMessage("Injected by middleware")]}

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))

        agent = create_agent(model=model, tools=[], middleware=[add_metadata])

        result = agent.invoke({"messages": [HumanMessage("Original")]})

        # Should have original + injected + AI response
        assert len(result["messages"]) >= 2
        message_contents = [msg.content for msg in result["messages"]]
        assert "Injected by middleware" in message_contents

    def test_before_agent_with_class_inheritance(self) -> None:
        """Test before_agent using class inheritance."""
        execution_log = []

        class CustomBeforeAgentMiddleware(AgentMiddleware):
            def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
                execution_log.append("class_before_agent_called")
                return None

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))

        agent = create_agent(model=model, tools=[], middleware=[CustomBeforeAgentMiddleware()])

        agent.invoke({"messages": [HumanMessage("Test")]})

        assert "class_before_agent_called" in execution_log

    async def test_before_agent_with_async_class_inheritance(self) -> None:
        """Test async before_agent using class inheritance."""
        execution_log = []

        class CustomAsyncBeforeAgentMiddleware(AgentMiddleware):
            async def abefore_agent(
                self, state: AgentState, runtime: Runtime
            ) -> dict[str, Any] | None:
                execution_log.append("async_class_before_agent_called")
                return None

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))

        agent = create_agent(model=model, tools=[], middleware=[CustomAsyncBeforeAgentMiddleware()])

        await agent.ainvoke({"messages": [HumanMessage("Test")]})

        assert "async_class_before_agent_called" in execution_log


class TestAfterAgentBasic:
    """Test basic after_agent functionality."""

    def test_sync_after_agent_execution(self) -> None:
        """Test that after_agent hook is called synchronously."""
        execution_log = []

        @after_agent
        def log_after_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            execution_log.append("after_agent_called")
            execution_log.append(f"final_message_count: {len(state['messages'])}")
            return None

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Final response")]))

        agent = create_agent(model=model, tools=[], middleware=[log_after_agent])

        agent.invoke({"messages": [HumanMessage("Hi")]})

        assert "after_agent_called" in execution_log
        assert any("final_message_count:" in log for log in execution_log)

    async def test_async_after_agent_execution(self) -> None:
        """Test that after_agent hook is called asynchronously."""
        execution_log = []

        @after_agent
        async def async_log_after_agent(
            state: AgentState, runtime: Runtime
        ) -> dict[str, Any] | None:
            execution_log.append("async_after_agent_called")
            return None

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))

        agent = create_agent(model=model, tools=[], middleware=[async_log_after_agent])

        await agent.ainvoke({"messages": [HumanMessage("Hi")]})

        assert "async_after_agent_called" in execution_log

    def test_after_agent_state_modification(self) -> None:
        """Test that after_agent can modify state."""

        @after_agent
        def add_final_message(state: AgentState, runtime: Runtime) -> dict[str, Any]:
            return {"messages": [AIMessage("Added by after_agent")]}

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Model response")]))

        agent = create_agent(model=model, tools=[], middleware=[add_final_message])

        result = agent.invoke({"messages": [HumanMessage("Test")]})

        message_contents = [msg.content for msg in result["messages"]]
        assert "Added by after_agent" in message_contents

    def test_after_agent_with_class_inheritance(self) -> None:
        """Test after_agent using class inheritance."""
        execution_log = []

        class CustomAfterAgentMiddleware(AgentMiddleware):
            def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
                execution_log.append("class_after_agent_called")
                return None

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))

        agent = create_agent(model=model, tools=[], middleware=[CustomAfterAgentMiddleware()])

        agent.invoke({"messages": [HumanMessage("Test")]})

        assert "class_after_agent_called" in execution_log

    async def test_after_agent_with_async_class_inheritance(self) -> None:
        """Test async after_agent using class inheritance."""
        execution_log = []

        class CustomAsyncAfterAgentMiddleware(AgentMiddleware):
            async def aafter_agent(
                self, state: AgentState, runtime: Runtime
            ) -> dict[str, Any] | None:
                execution_log.append("async_class_after_agent_called")
                return None

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))

        agent = create_agent(model=model, tools=[], middleware=[CustomAsyncAfterAgentMiddleware()])

        await agent.ainvoke({"messages": [HumanMessage("Test")]})

        assert "async_class_after_agent_called" in execution_log


class TestBeforeAndAfterAgentCombined:
    """Test before_agent and after_agent hooks working together."""

    def test_execution_order(self) -> None:
        """Test that before_agent executes before after_agent."""
        execution_log = []

        @before_agent
        def log_before(state: AgentState, runtime: Runtime) -> None:
            execution_log.append("before")

        @after_agent
        def log_after(state: AgentState, runtime: Runtime) -> None:
            execution_log.append("after")

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))

        agent = create_agent(model=model, tools=[], middleware=[log_before, log_after])

        agent.invoke({"messages": [HumanMessage("Test")]})

        assert execution_log == ["before", "after"]

    async def test_async_execution_order(self) -> None:
        """Test async execution order of before_agent and after_agent."""
        execution_log = []

        @before_agent
        async def async_log_before(state: AgentState, runtime: Runtime) -> None:
            execution_log.append("async_before")

        @after_agent
        async def async_log_after(state: AgentState, runtime: Runtime) -> None:
            execution_log.append("async_after")

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))

        agent = create_agent(model=model, tools=[], middleware=[async_log_before, async_log_after])

        await agent.ainvoke({"messages": [HumanMessage("Test")]})

        assert execution_log == ["async_before", "async_after"]

    def test_state_passthrough(self) -> None:
        """Test that state modifications in before_agent are visible to after_agent."""
        collected_states = {}

        @before_agent
        def modify_in_before(state: AgentState, runtime: Runtime) -> dict[str, Any]:
            return {"messages": [HumanMessage("Modified by before_agent")]}

        @after_agent
        def capture_in_after(state: AgentState, runtime: Runtime) -> None:
            collected_states["messages"] = state["messages"]

        model = GenericFakeChatModel(messages=iter([AIMessage(content="Response")]))

        agent = create_agent(model=model, tools=[], middleware=[modify_in_before, capture_in_after])

        agent.invoke({"messages": [HumanMessage("Original")]})

        message_contents = [msg.content for msg in collected_states["messages"]]
        assert "Modified by before_agent" in message_contents

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

        assert "before_1" in execution_log
        assert "before_2" in execution_log
        assert "after_1" in execution_log
        assert "after_2" in execution_log

    def test_agent_hooks_run_once_with_multiple_model_calls(self) -> None:
        """Test that before_agent and after_agent run only once even with tool calls."""
        execution_log = []

        @before_agent
        def log_before_agent(state: AgentState, runtime: Runtime) -> None:
            execution_log.append("before_agent")

        @after_agent
        def log_after_agent(state: AgentState, runtime: Runtime) -> None:
            execution_log.append("after_agent")

        # Model will call a tool once, then respond with final answer
        model = FakeToolCallingModel(
            tool_calls=[
                [{"name": "sample_tool", "args": {"query": "test"}, "id": "1"}],
                [],  # Second call returns no tool calls (final answer)
            ]
        )

        agent = create_agent(
            model=model,
            tools=[sample_tool],
            middleware=[log_before_agent, log_after_agent],
        )

        agent.invoke({"messages": [HumanMessage("Test")]})

        # before_agent and after_agent should run exactly once
        assert execution_log.count("before_agent") == 1
        assert execution_log.count("after_agent") == 1
        # before_agent should run first, after_agent should run last
        assert execution_log[0] == "before_agent"
        assert execution_log[-1] == "after_agent"


class TestDecoratorParameters:
    """Test decorator parameters for before_agent and after_agent."""

    def test_before_agent_with_custom_name(self) -> None:
        """Test before_agent with custom middleware name."""

        @before_agent(name="CustomBeforeAgentMiddleware")
        def custom_named_before(state: AgentState, runtime: Runtime) -> None:
            pass

        assert custom_named_before.name == "CustomBeforeAgentMiddleware"

    def test_after_agent_with_custom_name(self) -> None:
        """Test after_agent with custom middleware name."""

        @after_agent(name="CustomAfterAgentMiddleware")
        def custom_named_after(state: AgentState, runtime: Runtime) -> None:
            pass

        assert custom_named_after.name == "CustomAfterAgentMiddleware"

    def test_before_agent_default_name(self) -> None:
        """Test that before_agent uses function name by default."""

        @before_agent
        def my_before_agent_function(state: AgentState, runtime: Runtime) -> None:
            pass

        assert my_before_agent_function.name == "my_before_agent_function"

    def test_after_agent_default_name(self) -> None:
        """Test that after_agent uses function name by default."""

        @after_agent
        def my_after_agent_function(state: AgentState, runtime: Runtime) -> None:
            pass

        assert my_after_agent_function.name == "my_after_agent_function"
