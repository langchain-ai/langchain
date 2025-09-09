#!/usr/bin/env python3
"""Verify the context manager fix is working."""

import sys
sys.path.insert(0, '/home/daytona/langchain/libs/core')

from langchain_core.runnables.config import set_config_context, ConfigContext, RunnableConfig

# Test 1: Check that set_config_context returns ConfigContext
config = RunnableConfig()
ctx = set_config_context(config)
print(f"1. Context manager type: {type(ctx).__name__}")
print(f"   Expected: ConfigContext, Got: {type(ctx).__name__}")
assert isinstance(ctx, ConfigContext), f"Expected ConfigContext, got {type(ctx)}"

# Test 2: Test reuse raises RuntimeError
print("\n2. Testing reuse raises RuntimeError:")
ctx_manager = set_config_context(config)
try:
    with ctx_manager as ctx1:
        print("   First enter successful")
        try:
            with ctx_manager as ctx2:
                print("   ERROR: Second enter should not succeed")
                assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            print(f"   SUCCESS: Caught expected RuntimeError: {e}")
            assert "Cannot re-enter an already-entered context manager" in str(e)
except Exception as e:
    print(f"   ERROR: Unexpected exception: {e}")
    raise

# Test 3: Test exit without enter
print("\n3. Testing exit without enter raises RuntimeError:")
ctx_manager = set_config_context(config)
try:
    ctx_manager.__exit__(None, None, None)
    print("   ERROR: Exit without enter should raise RuntimeError")
    assert False, "Should have raised RuntimeError"
except RuntimeError as e:
    print(f"   SUCCESS: Caught expected RuntimeError: {e}")
    assert "Cannot exit context manager that was not entered" in str(e)

# Test 4: Normal usage works
print("\n4. Testing normal usage:")
with set_config_context(config) as ctx:
    print("   SUCCESS: Normal usage works")
    assert ctx is not None

# Test 5: Nested different instances work
print("\n5. Testing nested different instances:")
config1 = RunnableConfig(tags=["outer"])
config2 = RunnableConfig(tags=["inner"])
with set_config_context(config1) as ctx1:
    with set_config_context(config2) as ctx2:
        print("   SUCCESS: Nested contexts work")
        assert ctx1 is not None
        assert ctx2 is not None

print("\n✅ All tests passed!")
