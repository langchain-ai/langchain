#!/usr/bin/env python3
"""Test script to verify the context manager fix."""

from langchain_core.runnables.config import set_config_context
from langchain_core.runnables import RunnableConfig

def test_reuse_raises_error():
    """Test that reusing the same context manager raises RuntimeError."""
    config = RunnableConfig()
    ctx_manager = set_config_context(config)
    
    try:
        with ctx_manager as ctx1:
            print("✓ First enter successful")
            try:
                with ctx_manager as ctx2:
                    print("✗ Second enter should not succeed")
                    return False
            except RuntimeError as e:
                if "Cannot re-enter an already-entered context manager" in str(e):
                    print(f"✓ Caught expected RuntimeError: {e}")
                    return True
                else:
                    print(f"✗ Wrong error message: {e}")
                    return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_exit_without_enter():
    """Test that exiting without entering raises RuntimeError."""
    config = RunnableConfig()
    ctx_manager = set_config_context(config)
    
    try:
        ctx_manager.__exit__(None, None, None)
        print("✗ Exit without enter should raise RuntimeError")
        return False
    except RuntimeError as e:
        if "Cannot exit context manager that was not entered" in str(e):
            print(f"✓ Caught expected RuntimeError: {e}")
            return True
        else:
            print(f"✗ Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_normal_usage():
    """Test that normal usage still works correctly."""
    config = RunnableConfig(tags=["test"])
    
    try:
        with set_config_context(config) as ctx:
            print("✓ Normal usage works")
            # Context should be available
            if ctx is not None:
                print("✓ Context is available")
                return True
            else:
                print("✗ Context is None")
                return False
    except Exception as e:
        print(f"✗ Normal usage failed: {e}")
        return False

def test_nested_different_instances():
    """Test that using different instances in nested contexts works."""
    config1 = RunnableConfig(tags=["outer"])
    config2 = RunnableConfig(tags=["inner"])
    
    try:
        with set_config_context(config1) as ctx1:
            print("✓ Outer context entered")
            with set_config_context(config2) as ctx2:
                print("✓ Inner context entered")
                if ctx1 is not None and ctx2 is not None:
                    print("✓ Both contexts are available")
                    return True
                else:
                    print("✗ One or both contexts are None")
                    return False
    except Exception as e:
        print(f"✗ Nested contexts failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing context manager fix...")
    print("-" * 50)
    
    all_passed = True
    
    print("\n1. Testing reuse raises error:")
    if not test_reuse_raises_error():
        all_passed = False
    
    print("\n2. Testing exit without enter:")
    if not test_exit_without_enter():
        all_passed = False
    
    print("\n3. Testing normal usage:")
    if not test_normal_usage():
        all_passed = False
    
    print("\n4. Testing nested different instances:")
    if not test_nested_different_instances():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
