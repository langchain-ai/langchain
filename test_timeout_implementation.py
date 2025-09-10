#!/usr/bin/env python3
"""Simple test script to verify ToolNode timeout implementation works correctly."""

import asyncio
import sys
import os

# Add the langchain_v1 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs', 'langchain_v1'))

from langchain.agents import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import AIMessage


@tool
async def slow_tool(duration: float) -> str:
    """A tool that simulates a slow operation."""
    await asyncio.sleep(duration)
    return f"Completed after {duration} seconds"


@tool
async def fast_tool(message: str) -> str:
    """A tool that completes quickly."""
    await asyncio.sleep(0.1)
    return f"Processed: {message}"


async def test_timeout_triggers():
    """Test that timeout is triggered when tool takes too long."""
    print("Test 1: Testing timeout triggers...")
    tool_node = ToolNode([slow_tool], timeout=0.5)
    
    try:
        await tool_node.ainvoke({
            "messages": [
                AIMessage(
                    "Testing timeout",
                    tool_calls=[{
                        "name": "slow_tool",
                        "args": {"duration": 1.0},
                        "id": "test_timeout",
                    }],
                )
            ]
        })
        print("❌ Test 1 FAILED: Expected TimeoutError but none was raised")
        return False
    except asyncio.TimeoutError as e:
        if "Tool execution timed out after 0.5 seconds" in str(e) and "slow_tool" in str(e):
            print(f"✅ Test 1 PASSED: Timeout triggered correctly with message: {e}")
            return True
        else:
            print(f"❌ Test 1 FAILED: Timeout message incorrect: {e}")
            return False
    except Exception as e:
        print(f"❌ Test 1 FAILED: Unexpected error: {e}")
        return False


async def test_no_timeout_backward_compatibility():
    """Test that ToolNode works without timeout (backward compatibility)."""
    print("\nTest 2: Testing backward compatibility (no timeout)...")
    tool_node = ToolNode([fast_tool])
    
    try:
        result = await tool_node.ainvoke({
            "messages": [
                AIMessage(
                    "Test",
                    tool_calls=[{
                        "name": "fast_tool",
                        "args": {"message": "Hello"},
                        "id": "1",
                    }],
                )
            ]
        })
        
        if result["messages"][0].content == "Processed: Hello":
            print("✅ Test 2 PASSED: Tool executed successfully without timeout")
            return True
        else:
            print(f"❌ Test 2 FAILED: Unexpected result: {result}")
            return False
    except Exception as e:
        print(f"❌ Test 2 FAILED: Unexpected error: {e}")
        return False


async def test_successful_completion_within_timeout():
    """Test that tools complete successfully when within timeout."""
    print("\nTest 3: Testing successful completion within timeout...")
    tool_node = ToolNode([fast_tool], timeout=1.0)
    
    try:
        result = await tool_node.ainvoke({
            "messages": [
                AIMessage(
                    "Test",
                    tool_calls=[{
                        "name": "fast_tool",
                        "args": {"message": "Quick task"},
                        "id": "1",
                    }],
                )
            ]
        })
        
        if "Processed: Quick task" in result["messages"][0].content:
            print("✅ Test 3 PASSED: Tool completed successfully within timeout")
            return True
        else:
            print(f"❌ Test 3 FAILED: Unexpected result: {result}")
            return False
    except Exception as e:
        print(f"❌ Test 3 FAILED: Unexpected error: {e}")
        return False


async def test_multiple_tools_timeout():
    """Test timeout with multiple tools running in parallel."""
    print("\nTest 4: Testing timeout with multiple tools in parallel...")
    tool_node = ToolNode([fast_tool, slow_tool], timeout=0.3)
    
    try:
        await tool_node.ainvoke({
            "messages": [
                AIMessage(
                    "Test parallel",
                    tool_calls=[
                        {
                            "name": "fast_tool",
                            "args": {"message": "Fast 1"},
                            "id": "fast_1",
                        },
                        {
                            "name": "slow_tool",
                            "args": {"duration": 1.0},
                            "id": "slow_1",
                        },
                    ],
                )
            ]
        })
        print("❌ Test 4 FAILED: Expected TimeoutError but none was raised")
        return False
    except asyncio.TimeoutError as e:
        error_msg = str(e)
        if "Tool execution timed out" in error_msg and "fast_tool" in error_msg and "slow_tool" in error_msg:
            print(f"✅ Test 4 PASSED: Timeout triggered with multiple tools: {e}")
            return True
        else:
            print(f"❌ Test 4 FAILED: Timeout message incorrect: {e}")
            return False
    except Exception as e:
        print(f"❌ Test 4 FAILED: Unexpected error: {e}")
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing ToolNode Timeout Implementation")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(await test_timeout_triggers())
    results.append(await test_no_timeout_backward_compatibility())
    results.append(await test_successful_completion_within_timeout())
    results.append(await test_multiple_tools_timeout())
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests PASSED!")
        return 0
    else:
        print(f"❌ {total - passed} out of {total} tests FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
