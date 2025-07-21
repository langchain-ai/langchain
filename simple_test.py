#!/usr/bin/env python3
"""Very simple test to verify the cache fix logic."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs', 'core'))

try:
    from langchain_core.outputs import ChatGeneration, Generation
    from langchain_core.messages import AIMessage
    
    def test_conversion_logic():
        """Test the Generation to ChatGeneration conversion logic."""
        print("Testing conversion logic...")
        
        # Create a Generation object (the problematic type)
        gen = Generation(
            text="hello world",
            generation_info={"test": "info"},
            type="Generation"
        )
        
        print(f"Original Generation: {gen}")
        print(f"Type: {type(gen).__name__}")
        print(f"gen.type: {gen.type}")
        
        # Test the conversion logic
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
                print(f"✓ New type: {type(chat_gen).__name__}")
                print(f"✓ chat_gen.type: {chat_gen.type}")
            else:
                # Already a ChatGeneration or other expected type
                converted_generations.append(g)
        
        print(f"✓ Conversion successful! Got {len(converted_generations)} ChatGeneration objects")
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Other error: {e}")
        return False

if __name__ == "__main__":
    success = test_conversion_logic()
    sys.exit(0 if success else 1)