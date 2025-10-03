"""Tests for edge cases and error conditions in agents."""

import pytest
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.exceptions import LangChainException
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field

from .model import FakeToolCallingModel


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_agent_with_invalid_model_string(self) -> None:
        """Test agent creation with invalid model string."""
        with pytest.raises(Exception):  # Should raise some kind of error
            create_agent("invalid:model", [])

    def test_agent_with_none_tools(self) -> None:
        """Test agent with None tools."""
        model = FakeToolCallingModel()
        agent = create_agent(model, None)

        result = agent.invoke({"messages": [HumanMessage("Hello")]})
        assert "messages" in result

    def test_agent_with_empty_tools_list(self) -> None:
        """Test agent with empty tools list."""
        model = FakeToolCallingModel()
        agent = create_agent(model, [])

        result = agent.invoke({"messages": [HumanMessage("Hello")]})
        assert "messages" in result

    def test_agent_with_duplicate_middleware(self) -> None:
        """Test agent with duplicate middleware instances."""

        class TestMiddleware(AgentMiddleware):
            pass

        middleware1 = TestMiddleware()
        middleware2 = TestMiddleware()  # Same class, different instance

        model = FakeToolCallingModel()

        # Should raise an error about duplicate middleware
        with pytest.raises(AssertionError, match="Please remove duplicate middleware instances"):
            create_agent(model, [], middleware=[middleware1, middleware2])

    def test_agent_with_middleware_error(self) -> None:
        """Test agent with middleware that raises an error."""

        class ErrorMiddleware(AgentMiddleware):
            def before_model(self, state: AgentState, runtime) -> dict[str, Any]:
                raise ValueError("Middleware error")

        model = FakeToolCallingModel()
        agent = create_agent(model, [], middleware=[ErrorMiddleware()])

        with pytest.raises(ValueError, match="Middleware error"):
            agent.invoke({"messages": [HumanMessage("Hello")]})

    def test_agent_with_invalid_structured_output(self) -> None:
        """Test agent with invalid structured output configuration."""

        class InvalidSchema:
            pass  # Not a valid Pydantic model

        model = FakeToolCallingModel()

        # Should handle invalid schema gracefully
        with pytest.raises(Exception):
            create_agent(model, [], response_format=InvalidSchema)

    def test_agent_with_tool_exception(self) -> None:
        """Test agent handling tool exceptions."""

        @tool
        def error_tool(input: str) -> str:
            """Tool that always raises an exception."""
            raise Exception("Tool execution failed")

        model = FakeToolCallingModel(
            tool_calls=[[{"args": {"input": "test"}, "id": "1", "name": "error_tool"}], []]
        )

        agent = create_agent(model, [error_tool])
        result = agent.invoke({"messages": [HumanMessage("Use error tool")]})

        # Should handle tool exception gracefully
        assert "messages" in result
        messages = result["messages"]
        assert len(messages) >= 3  # Human + AI + Tool message
        assert isinstance(messages[2], ToolMessage)
        assert "Tool execution failed" in messages[2].content

    def test_agent_with_malformed_tool_call(self) -> None:
        """Test agent with malformed tool call from model."""
        model = FakeToolCallingModel(
            tool_calls=[[{"invalid": "tool_call"}], []]  # Malformed tool call
        )

        agent = create_agent(model, [])

        # Should handle malformed tool calls gracefully
        with pytest.raises(Exception):
            agent.invoke({"messages": [HumanMessage("Hello")]})

    def test_agent_with_empty_messages(self) -> None:
        """Test agent with empty messages list."""
        model = FakeToolCallingModel()
        agent = create_agent(model, [])

        result = agent.invoke({"messages": []})
        assert "messages" in result

    def test_agent_with_none_messages(self) -> None:
        """Test agent with None messages."""
        model = FakeToolCallingModel()
        agent = create_agent(model, [])

        with pytest.raises(Exception):
            agent.invoke({"messages": None})

    def test_agent_with_invalid_state(self) -> None:
        """Test agent with invalid state structure."""
        model = FakeToolCallingModel()
        agent = create_agent(model, [])

        with pytest.raises(Exception):
            agent.invoke({"invalid_key": "value"})

    def test_agent_with_large_message_history(self) -> None:
        """Test agent with large message history."""
        model = FakeToolCallingModel()
        agent = create_agent(model, [])

        # Create a large message history
        messages = [HumanMessage(f"Message {i}") for i in range(100)]

        result = agent.invoke({"messages": messages})
        assert "messages" in result

    def test_agent_with_special_characters(self) -> None:
        """Test agent with special characters in messages."""
        model = FakeToolCallingModel()
        agent = create_agent(model, [])

        special_message = HumanMessage("Hello! @#$%^&*()_+-=[]{}|;':\",./<>?")
        result = agent.invoke({"messages": [special_message]})
        assert "messages" in result

    def test_agent_with_unicode_messages(self) -> None:
        """Test agent with unicode messages."""
        model = FakeToolCallingModel()
        agent = create_agent(model, [])

        unicode_message = HumanMessage("Hello 世界! 🌍")
        result = agent.invoke({"messages": [unicode_message]})
        assert "messages" in result

    def test_agent_with_very_long_message(self) -> None:
        """Test agent with very long message."""
        model = FakeToolCallingModel()
        agent = create_agent(model, [])

        long_message = HumanMessage("A" * 10000)  # 10k character message
        result = agent.invoke({"messages": [long_message]})
        assert "messages" in result

    def test_agent_with_multiple_system_prompts(self) -> None:
        """Test agent behavior with multiple system prompts."""
        model = FakeToolCallingModel()
        agent = create_agent(model, [], system_prompt="First prompt")

        # Add system message to input
        messages = [HumanMessage("System: Second prompt"), HumanMessage("Hello")]

        result = agent.invoke({"messages": messages})
        assert "messages" in result

    def test_agent_with_tool_node_instead_of_list(self) -> None:
        """Test agent with ToolNode instead of tools list."""
        from langchain.tools import ToolNode

        @tool
        def test_tool(input: str) -> str:
            """Test tool."""
            return f"Result: {input}"

        tool_node = ToolNode([test_tool])
        model = FakeToolCallingModel()
        agent = create_agent(model, tool_node)

        result = agent.invoke({"messages": [HumanMessage("Hello")]})
        assert "messages" in result

    def test_agent_with_interrupt_before(self) -> None:
        """Test agent with interrupt_before configuration."""
        model = FakeToolCallingModel()
        agent = create_agent(model, [], interrupt_before=["model_request"])

        result = agent.invoke({"messages": [HumanMessage("Hello")]})
        assert "messages" in result

    def test_agent_with_interrupt_after(self) -> None:
        """Test agent with interrupt_after configuration."""
        model = FakeToolCallingModel()
        agent = create_agent(model, [], interrupt_after=["model_request"])

        result = agent.invoke({"messages": [HumanMessage("Hello")]})
        assert "messages" in result

    def test_agent_with_debug_mode(self) -> None:
        """Test agent with debug mode enabled."""
        model = FakeToolCallingModel()
        agent = create_agent(model, [], debug=True)

        result = agent.invoke({"messages": [HumanMessage("Hello")]})
        assert "messages" in result

    def test_agent_with_cache(self) -> None:
        """Test agent with cache configuration."""
        model = FakeToolCallingModel()
        cache = {}  # Simple dict cache
        agent = create_agent(model, [], cache=cache)

        result = agent.invoke({"messages": [HumanMessage("Hello")]})
        assert "messages" in result

    def test_agent_with_custom_name(self) -> None:
        """Test agent with custom name."""
        model = FakeToolCallingModel()
        agent = create_agent(model, [], name="CustomAgent")

        result = agent.invoke({"messages": [HumanMessage("Hello")]})
        assert "messages" in result

    def test_agent_with_context_schema(self) -> None:
        """Test agent with custom context schema."""
        from typing_extensions import TypedDict

        class CustomContext(TypedDict):
            user_id: str
            session_id: str

        model = FakeToolCallingModel()
        agent = create_agent(model, [], context_schema=CustomContext)

        result = agent.invoke({"messages": [HumanMessage("Hello")]})
        assert "messages" in result

    def test_agent_with_checkpointer_and_store(self) -> None:
        """Test agent with both checkpointer and store."""
        from langgraph.store.memory import InMemoryStore

        model = FakeToolCallingModel()
        checkpointer = InMemorySaver()
        store = InMemoryStore()

        agent = create_agent(model, [], checkpointer=checkpointer, store=store)

        result = agent.invoke({"messages": [HumanMessage("Hello")]})
        assert "messages" in result

    def test_agent_with_middleware_modifying_request(self) -> None:
        """Test agent with middleware that modifies model request."""

        class RequestModifyingMiddleware(AgentMiddleware):
            def modify_model_request(self, request, state, runtime):
                request.model_settings["custom_setting"] = "modified"
                return request

        model = FakeToolCallingModel()
        agent = create_agent(model, [], middleware=[RequestModifyingMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Hello")]})
        assert "messages" in result

    def test_agent_with_middleware_jumping_to_end(self) -> None:
        """Test agent with middleware that jumps to end."""

        class JumpToEndMiddleware(AgentMiddleware):
            def before_model(self, state: AgentState, runtime):
                return {"jump_to": "end"}

        model = FakeToolCallingModel()
        agent = create_agent(model, [], middleware=[JumpToEndMiddleware()])

        result = agent.invoke({"messages": [HumanMessage("Hello")]})
        assert "messages" in result
        # Should have minimal messages due to jumping to end
        assert len(result["messages"]) <= 2

    def test_agent_with_structured_output_error(self) -> None:
        """Test agent with structured output validation error."""

        class WeatherResponse(BaseModel):
            temperature: float = Field(description="Temperature in Fahrenheit")
            condition: str = Field(description="Weather condition")

        # Mock model that returns invalid structured output
        class InvalidStructuredModel(FakeToolCallingModel):
            def invoke(self, messages, **kwargs):
                # Return a message that doesn't match the schema
                return AIMessage(content="Invalid response")

        model = InvalidStructuredModel()
        agent = create_agent(model, [], response_format=ToolStrategy(schema=WeatherResponse))

        # Should handle structured output errors gracefully
        result = agent.invoke({"messages": [HumanMessage("What's the weather?")]})
        assert "messages" in result

    def test_agent_with_concurrent_invocations(self) -> None:
        """Test agent with concurrent invocations."""
        import asyncio

        model = FakeToolCallingModel()
        agent = create_agent(model, [])

        async def run_concurrent():
            tasks = [agent.ainvoke({"messages": [HumanMessage(f"Message {i}")]}) for i in range(5)]
            results = await asyncio.gather(*tasks)
            return results

        # Run concurrent invocations
        results = asyncio.run(run_concurrent())

        # All should succeed
        assert len(results) == 5
        for result in results:
            assert "messages" in result

    def test_agent_with_streaming(self) -> None:
        """Test agent streaming functionality."""
        model = FakeToolCallingModel()
        agent = create_agent(model, [])

        # Test streaming
        stream = agent.stream({"messages": [HumanMessage("Hello")]})
        chunks = list(stream)

        # Should have at least one chunk
        assert len(chunks) > 0
        # Each chunk should be a dict
        for chunk in chunks:
            assert isinstance(chunk, dict)
