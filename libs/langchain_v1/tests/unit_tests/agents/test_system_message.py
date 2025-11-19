"""Tests for system_prompt support in create_agent and ModelRequest."""

import warnings

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest
from tests.unit_tests.agents.model import FakeToolCallingModel


def test_create_agent_accepts_string_system_prompt():
    """Test that create_agent accepts a string system_prompt."""
    model = FakeToolCallingModel()
    agent = create_agent(model, system_prompt="You are a helpful assistant")

    # Run the agent to ensure it works
    result = agent.invoke({"messages": [HumanMessage(content="Hello")]})
    assert "messages" in result


def test_create_agent_accepts_system_message():
    """Test that create_agent accepts a SystemMessage for system_prompt."""
    model = FakeToolCallingModel()
    system_msg = SystemMessage(content="You are a helpful assistant")
    agent = create_agent(model, system_prompt=system_msg)

    # Run the agent to ensure it works
    result = agent.invoke({"messages": [HumanMessage(content="Hello")]})
    assert "messages" in result


def test_model_request_deprecates_string_system_prompt(mock_runtime):
    """Test that ModelRequest raises deprecation warning for string system_prompt."""
    model = FakeToolCallingModel()

    # Expect a deprecation warning when passing a string
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        request = ModelRequest(
            model=model,
            system_prompt="You are a helpful assistant",
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={"messages": []},
            runtime=mock_runtime,
        )

        # Check that a deprecation warning was raised
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "system_prompt" in str(w[0].message)

        # Verify that the string was coerced to SystemMessage
        assert isinstance(request.system_prompt, SystemMessage)
        assert request.system_prompt.content == "You are a helpful assistant"


def test_model_request_accepts_system_message(mock_runtime):
    """Test that ModelRequest accepts SystemMessage without deprecation warning."""
    model = FakeToolCallingModel()
    system_msg = SystemMessage(content="You are a helpful assistant")

    # Should not raise any warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        request = ModelRequest(
            model=model,
            system_prompt=system_msg,
            messages=[],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={"messages": []},
            runtime=mock_runtime,
        )

        # No deprecation warning should be raised
        assert len(w) == 0

        # Verify that the SystemMessage is preserved
        assert isinstance(request.system_prompt, SystemMessage)
        assert request.system_prompt.content == "You are a helpful assistant"


def test_model_request_override_with_string(mock_runtime):
    """Test that ModelRequest.override() works with string system_prompt."""
    model = FakeToolCallingModel()
    system_msg = SystemMessage(content="Original prompt")

    request = ModelRequest(
        model=model,
        system_prompt=system_msg,
        messages=[],
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": []},
        runtime=mock_runtime,
    )

    # Override with a string - should trigger deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        new_request = request.override(system_prompt="New prompt")

        # Check for deprecation warning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)

        # Verify the override worked
        assert isinstance(new_request.system_prompt, SystemMessage)
        assert new_request.system_prompt.content == "New prompt"


@pytest.fixture
def mock_runtime():
    """Create a mock runtime for testing."""

    class MockRuntime:
        pass

    return MockRuntime()
