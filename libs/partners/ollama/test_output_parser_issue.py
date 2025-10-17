"""
Python reproduction of OutputParserException issue with Ollama
Original issue from JavaScript: OutputParserException with empty text parsing
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

# Initialize Ollama (equivalent to JavaScript version)
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    base_url="http://localhost:11434"
)

# Define the classification schema (equivalent to Zod schema)
class ClassificationSchema(BaseModel):
    """Extract sentiment, aggressiveness, and language from text"""
    
    sentiment: Literal["happy", "neutral", "sad"] = Field(
        description="The sentiment of the text"
    )
    aggressiveness: int = Field(
        description="Describes how aggressive the statement is on a scale from 1 to 5. The higher the number the more aggressive"
    )
    language: Literal["spanish", "english", "french", "german", "italian"] = Field(
        description="The language the text is written in"
    )


# Create the tagging prompt
tagging_prompt = ChatPromptTemplate.from_template(
    """Extract the desired information from the following passage.

Passage:
{input}
"""
)

# Create LLM with structured output
llm_with_structured_output = llm.with_structured_output(
    ClassificationSchema,
    method="function_calling"  # or try "json_schema"
)

# Test input (Spanish text)
test_input = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"

print("Testing Ollama structured output with Spanish text...")
print(f"Input: {test_input}\n")

try:
    # Format the prompt
    prompt = tagging_prompt.invoke({"input": test_input})
    
    # Get structured output
    result = llm_with_structured_output.invoke(prompt)
    
    print("✓ SUCCESS!")
    print(f"Result: {result}")
    print(f"\nParsed values:")
    print(f"  - Sentiment: {result.sentiment}")
    print(f"  - Aggressiveness: {result.aggressiveness}")
    print(f"  - Language: {result.language}")
    
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}")
    print(f"Error: {e}")
    print("\nThis is the OutputParserException the fix addresses!")


# Additional test to show the fix working
print("\n" + "="*60)
print("Testing with different methods:")
print("="*60)

methods = ["function_calling", "json_schema"]

for method in methods:
    print(f"\nMethod: {method}")
    try:
        llm_test = llm.with_structured_output(
            ClassificationSchema,
            method=method
        )
        result = llm_test.invoke(
            tagging_prompt.invoke({"input": test_input})
        )
        print(f"  ✓ Success: {result}")
    except Exception as e:
        print(f"  ✗ Failed: {type(e).__name__}: {str(e)[:100]}")
