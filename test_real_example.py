#!/usr/bin/env python3
"""
Test the fix with the actual example from issue #28848
"""

import sys
import os

# Add the paths to use our modified langchain
sys.path.insert(0, '/home/nav/langchain/libs/core')
sys.path.insert(0, '/home/nav/langchain/libs/partners/openai')

# Mock the OpenAI client to avoid API calls
import unittest.mock

# Mock openai module
mock_openai = unittest.mock.MagicMock()
sys.modules['openai'] = mock_openai

# Mock tiktoken
mock_tiktoken = unittest.mock.MagicMock()
mock_tiktoken.encoding_for_model.return_value.encode.return_value = []
sys.modules['tiktoken'] = mock_tiktoken

try:
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field
    from langchain.tools import StructuredTool

    # Test data from the original issue
    class ResponseModel(BaseModel):
        a_value: str = Field(description="This doesn't matter much")

    def a_func(val: int):
        return True

    a_tool = StructuredTool.from_function(
        func=a_func,
        name="A func", 
        description="A function you will need",
    )

    print("🧪 Testing the actual issue #28848 example...")
    
    # This is the exact code that was failing
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    print(f"✅ Created ChatOpenAI: {type(llm)}")
    
    structured_llm = llm.with_structured_output(ResponseModel)
    print(f"✅ Created structured_llm: {type(structured_llm)}")
    print(f"   Steps: {[type(step).__name__ for step in structured_llm.steps]}")
    
    # This was the line that failed before our fix
    try:
        llm_with_tools = structured_llm.bind_tools([a_tool])
        print("🎉 SUCCESS: The original failing code now works!")
        print(f"   Result type: {type(llm_with_tools)}")
        print(f"   Result steps: {[type(step).__name__ for step in llm_with_tools.steps]}")
        
    except AttributeError as e:
        print(f"❌ Still failing: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error during setup: {e}")
    sys.exit(1)

print("\n✅ All tests passed! Issue #28848 is fixed.")