#!/usr/bin/env python3
"""Test the actual cache fix in isolation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs', 'core'))

# Import the required classes
from langchain_core.outputs.chat_result import ChatResult
from langchain_core.outputs.generation import Generation
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.messages import AIMessage

def test_chat_result_with_generation_objects():
    """Test that our fix allows ChatResult to accept Generation objects."""
    print("Testing ChatResult creation with Generation objects...")
    
    # Create a Generation object (the problematic case)
    gen = Generation(
        text="hello world",
        generation_info={"test": "info"},
        type="Generation"
    )
    
    print(f"Created Generation: {gen}")
    print(f"Type: {type(gen).__name__}")
    
    # Try to create ChatResult directly - this should fail with original code
    try:
        result = ChatResult(generations=[gen])
        print(f"✗ Unexpected success - ChatResult accepted Generation objects: {result}")
        return False
    except Exception as e:
        print(f"✓ Expected failure - ChatResult rejected Generation objects: {e}")
    
    # Now test our conversion logic
    print("\nTesting conversion logic...")
    cache_val = [gen]
    converted_generations = []
    
    for g in cache_val:
        if isinstance(g, Generation) and not isinstance(g, ChatGeneration):
            # Convert Generation to ChatGeneration by creating an AIMessage
            # from the text content
            chat_gen = ChatGeneration(
                message=AIMessage(content=g.text),
                generation_info=g.generation_info,
            )
            converted_generations.append(chat_gen)
            print(f"✓ Converted Generation to ChatGeneration: {chat_gen}")
        else:
            # Already a ChatGeneration or other expected type
            converted_generations.append(g)
    
    # Now try to create ChatResult with converted generations
    try:
        result = ChatResult(generations=converted_generations)
        print(f"✓ Success - ChatResult accepted converted ChatGeneration objects: {result}")
        print(f"✓ Result content: {result.generations[0].message.content}")
        return True
    except Exception as e:
        print(f"✗ Failed - ChatResult still rejected converted objects: {e}")
        return False

if __name__ == "__main__":
    success = test_chat_result_with_generation_objects()
    sys.exit(0 if success else 1)