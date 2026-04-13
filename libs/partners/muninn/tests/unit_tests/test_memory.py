"""Tests for langchain-muninn integration."""

from langchain_muninn import MuninnMemory, MuninnEntityMemory


def test_import():
    """Test that the package can be imported."""
    assert MuninnMemory is not None
    assert MuninnEntityMemory is not None


def test_memory_variables():
    """Test that memory variables are defined."""
    memory = MuninnMemory(api_key="test_key")
    assert memory.memory_variables == ["history"]
    
    entity_memory = MuninnEntityMemory(api_key="test_key")
    assert entity_memory.memory_variables == ["entity_facts"]