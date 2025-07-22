#!/usr/bin/env python3
"""
Test script for the bind_tools + with_structured_output fix
"""

import sys
import os
sys.path.insert(0, '/home/nav/langchain/libs/core')
sys.path.insert(0, '/home/nav/langchain/libs/partners/openai')

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# Mock ChatOpenAI for testing without API calls
class MockChatOpenAI:
    def __init__(self, model="gpt-4o-mini", temperature=0):
        self.model = model
        self.temperature = temperature
        
    def with_structured_output(self, schema):
        # Simulate what the real method does - return a RunnableSequence
        from langchain_core.runnables import RunnableSequence, RunnableLambda
        
        # Create a mock bound model using RunnableLambda that has bind_tools
        def mock_model_func(x):
            return x
            
        mock_bound_model = RunnableLambda(mock_model_func)
        
        # Add bind_tools method to the mock
        def bind_tools_method(tools, **kwargs):
            print(f"✅ bind_tools called with {len(tools)} tools and kwargs: {kwargs}")
            # Return a new RunnableLambda to simulate successful binding
            return RunnableLambda(lambda x: f"bound_with_tools({x})")
            
        mock_bound_model.bind_tools = bind_tools_method
        mock_parser = RunnableLambda(lambda x: f"parsed({x})")
        
        # This simulates the structure that with_structured_output creates
        return RunnableSequence(mock_bound_model, mock_parser)
    
    def bind_tools(self, tools, **kwargs):
        print(f"✅ Direct bind_tools called with {len(tools)} tools")
        return self

# Test data
class ResponseModel(BaseModel):
    a_value: str = Field(description="This doesn't matter much")

def a_func(val: int):
    return True

a_tool = StructuredTool.from_function(
    func=a_func,
    name="A func",
    description="A function you will need",
)

def test_bind_tools_fix():
    print("🧪 Testing bind_tools + with_structured_output fix...")
    
    # Create the chain that was failing before
    llm = MockChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(ResponseModel)
    
    print(f"structured_llm type: {type(structured_llm)}")
    print(f"structured_llm steps: {[type(step).__name__ for step in structured_llm.steps]}")
    
    try:
        # This should work now!
        llm_with_tools = structured_llm.bind_tools([a_tool])
        print("✅ SUCCESS: bind_tools worked on RunnableSequence!")
        print(f"Result type: {type(llm_with_tools)}")
        return True
        
    except AttributeError as e:
        print(f"❌ FAILED: {e}")
        return False
    except Exception as e:
        print(f"❌ FAILED with unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_bind_tools_fix()
    sys.exit(0 if success else 1)