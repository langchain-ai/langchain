"""Tests for custom state schema with default values in middleware."""

from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import Required, NotRequired
from typing import Annotated

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    add_messages,
)

from .model import FakeToolCallingModel


class CustomState(AgentState):
    """Custom state schema with additional fields."""
    
    messages: Required[Annotated[list, add_messages]]
    custom_field: bool = False
    another_field: str = "default_value"


def test_custom_state_schema_with_defaults_in_middleware():
    """Test that custom state fields with defaults are accessible in middleware."""
    accessed_values = []
    
    class StateAccessMiddleware(AgentMiddleware):
        """Middleware that accesses custom state fields."""
        
        def before_model(self, request: ModelRequest):
            # Should be able to access custom fields even if not in input
            accessed_values.append({
                "custom_field": request.state.get("custom_field"),
                "another_field": request.state.get("another_field"),
            })
    
    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
        tools=[],
        system_prompt="Test assistant",
        state_schema=CustomState,
        middleware=[StateAccessMiddleware()],
    )
    
    # Invoke with only messages, not providing custom fields
    result = agent.invoke({"messages": [HumanMessage(content="test")]})
    
    # Middleware should have accessed the default values
    assert len(accessed_values) == 1
    assert accessed_values[0]["custom_field"] is False
    assert accessed_values[0]["another_field"] == "default_value"
    
    # Result should contain the message
    assert len(result["messages"]) == 2
    assert isinstance(result["messages"][0], HumanMessage)
    assert isinstance(result["messages"][1], AIMessage)


def test_custom_state_schema_with_provided_values():
    """Test that explicitly provided custom state values override defaults."""
    accessed_values = []
    
    class StateAccessMiddleware(AgentMiddleware):
        """Middleware that accesses custom state fields."""
        
        def before_model(self, request: ModelRequest):
            accessed_values.append({
                "custom_field": request.state.get("custom_field"),
                "another_field": request.state.get("another_field"),
            })
    
    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
        tools=[],
        system_prompt="Test assistant",
        state_schema=CustomState,
        middleware=[StateAccessMiddleware()],
    )
    
    # Invoke with custom field values provided
    result = agent.invoke({
        "messages": [HumanMessage(content="test")],
        "custom_field": True,
        "another_field": "custom_value",
    })
    
    # Middleware should have accessed the provided values
    assert len(accessed_values) == 1
    assert accessed_values[0]["custom_field"] is True
    assert accessed_values[0]["another_field"] == "custom_value"


def test_custom_state_schema_direct_access_no_keyerror():
    """Test that direct dictionary access to custom fields doesn't raise KeyError."""
    
    class DirectAccessMiddleware(AgentMiddleware):
        """Middleware that directly accesses custom state fields."""
        
        def before_model(self, request: ModelRequest):
            # This should NOT raise KeyError with the fix
            custom_field = request.state["custom_field"]
            another_field = request.state["another_field"]
            assert custom_field is False
            assert another_field == "default_value"
    
    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
        tools=[],
        system_prompt="Test assistant",
        state_schema=CustomState,
        middleware=[DirectAccessMiddleware()],
    )
    
    # This should not raise KeyError
    result = agent.invoke({"messages": [HumanMessage(content="test")]})
    assert result is not None
