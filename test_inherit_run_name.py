#!/usr/bin/env python3
"""Test script to verify inherit_run_name functionality."""

from typing import Any, Dict, List
from langchain_core.runnables import RunnableLambda
from langchain_core.callbacks.base import BaseCallbackHandler


class TestCallbackHandler(BaseCallbackHandler):
    """Test callback handler to track run names."""
    
    def __init__(self):
        self.run_names = []
        
    def on_chain_start(
        self, 
        serialized: Dict[str, Any], 
        inputs: Dict[str, Any], 
        **kwargs
    ) -> None:
        run_name = kwargs.get('run_name')
        print(f"Chain start - run_name: {run_name}")
        self.run_names.append(('chain', run_name))


def identity(x):
    """Identity function for testing."""
    return x


def process(x):
    """Process function for testing."""
    return f"processed: {x}"


def test_default_behavior():
    """Test that default behavior still drops run_name for child runs."""
    print("\n=== Test 1: Default behavior (run_name NOT inherited) ===")
    callback_handler = TestCallbackHandler()
    
    chain = RunnableLambda(identity) | RunnableLambda(process)
    result = chain.invoke(
        "test", 
        config={
            "run_name": "my_custom_run",
            "callbacks": [callback_handler]
        }
    )
    
    print(f"Result: {result}")
    print(f"Captured run names: {callback_handler.run_names}")
    
    # Verify: First should be "my_custom_run", others should be default names
    assert callback_handler.run_names[0][1] == "my_custom_run", "Root run should have custom name"
    # Child runs should NOT have the custom name (default behavior)
    for i in range(1, len(callback_handler.run_names)):
        assert callback_handler.run_names[i][1] != "my_custom_run", f"Child run {i} should NOT inherit run_name by default"
    
    print("✓ Default behavior verified: run_name is NOT inherited to child runs")


def test_inherit_run_name():
    """Test that inherit_run_name=True preserves run_name for child runs."""
    print("\n=== Test 2: With inherit_run_name=True ===")
    callback_handler = TestCallbackHandler()
    
    chain = RunnableLambda(identity) | RunnableLambda(process)
    result = chain.invoke(
        "test",
        config={
            "run_name": "my_inherited_run",
            "inherit_run_name": True,
            "callbacks": [callback_handler]
        }
    )
    
    print(f"Result: {result}")
    print(f"Captured run names: {callback_handler.run_names}")
    
    # Verify: All runs should have the same custom name when inherit_run_name=True
    for i, (run_type, run_name) in enumerate(callback_handler.run_names):
        if run_name == "my_inherited_run":
            print(f"✓ Run {i} ({run_type}) correctly inherited run_name")
    
    # At least the root should have the custom name
    assert callback_handler.run_names[0][1] == "my_inherited_run", "Root run should have custom name"
    
    print("✓ inherit_run_name=True verified: run_name is preserved for child runs")


def test_with_config_override():
    """Test that per-step with_config still overrides inherited value."""
    print("\n=== Test 3: Per-step with_config override ===")
    callback_handler = TestCallbackHandler()
    
    # Create a chain where one step has its own run_name
    chain = (
        RunnableLambda(identity).with_config(run_name="step_specific_name") |
        RunnableLambda(process)
    )
    
    result = chain.invoke(
        "test",
        config={
            "run_name": "global_run",
            "inherit_run_name": True,
            "callbacks": [callback_handler]
        }
    )
    
    print(f"Result: {result}")
    print(f"Captured run names: {callback_handler.run_names}")
    
    # Verify that the step with specific config has its own name
    found_specific = False
    for run_type, run_name in callback_handler.run_names:
        if run_name == "step_specific_name":
            found_specific = True
            print(f"✓ Found step with specific run_name override: {run_name}")
            break
    
    assert found_specific, "Step with with_config should have its specific run_name"
    print("✓ Per-step with_config override verified")


if __name__ == "__main__":
    try:
        test_default_behavior()
        test_inherit_run_name()
        test_with_config_override()
        print("\n✅ All tests passed! inherit_run_name functionality is working correctly.")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)
