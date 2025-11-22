"""Test AgentRuntime's metadata, state management,
and lifecycle functionality"""

from datetime import datetime

from langchain.agents.runtime import (AgentMetadata, AgentRuntime, AgentState,
                                      AgentStatus)


def test_agent_metadata_to_dict():
    """Test metadata serialization functionality"""
    created_at = datetime(2024, 1, 1)
    metadata = AgentMetadata(
        name="test_agent",
        description="Test agent",
        agent_type="react",
        model_name="gpt-3.5-turbo",
        created_at=created_at,
    )
    result = metadata.to_dict()
    assert result["name"] == "test_agent"
    assert (
        result["created_at"] == "2024-01-01T00:00:00"
    )  # Verify datetime to ISO format


def test_agent_state_updates():
    """Test state update logic (call, complete)"""
    state = AgentState()
    assert state.status == AgentStatus.INITIALIZED
    assert state.call_count == 0

    # Simulate call
    state.update_on_call()
    assert state.status == AgentStatus.RUNNING
    assert state.call_count == 1
    assert state.last_call_time is not None  # Call time updated

    # Simulate successful completion
    state.update_on_complete(success=True)
    assert state.status == AgentStatus.SUCCEEDED
    assert state.error is None

    # Simulate failed completion
    state.update_on_complete(success=False, error="Test error")
    assert state.status == AgentStatus.FAILED
    assert state.error == "Test error"


def test_agent_runtime_lifecycle():
    """Test AgentRuntime's complete lifecycle hooks"""
    metadata = AgentMetadata(
        name="runtime_test",
        description="Runtime test agent",
        agent_type="structured",
        model_name="gpt-4",
    )
    runtime = AgentRuntime(metadata)

    # Initial state
    assert runtime.state.status == AgentStatus.INITIALIZED
    assert runtime.get_summary()["metadata"]["name"] == "runtime_test"

    # Simulate call
    runtime.on_call()
    assert runtime.state.status == AgentStatus.RUNNING
    assert runtime.state.call_count == 1

    # Simulate successful completion
    runtime.on_complete(success=True)
    assert runtime.state.status == AgentStatus.SUCCEEDED
