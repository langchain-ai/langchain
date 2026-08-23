from collections import ChainMap, UserDict
from types import MappingProxyType

from langchain_core.prompts import PromptTemplate

def test_prompt_validation_with_various_mappings() -> None:
    """Test that _validate_input handles various Mapping types correctly."""
    template = "Hello {name}! Today is {day}."
    prompt = PromptTemplate.from_template(template)
    
    # Test cases with multi-variable input that were previously failing
    test_cases = [
        ("UserDict", UserDict({"name": "World", "day": "Monday"})),
        ("ChainMap", ChainMap({"name": "World", "day": "Monday"})),
        ("MappingProxyType", MappingProxyType({"name": "World", "day": "Monday"})),
    ]
    
    for label, input_data in test_cases:
        # Should not raise TypeError
        validated = prompt._validate_input(input_data)
        assert isinstance(validated, dict), f"Failed for {label}: expected dict output"
        assert validated == {"name": "World", "day": "Monday"}, f"Failed for {label}: incorrect content"

def test_prompt_invoke_with_various_mappings() -> None:
    """Test that invoke handles various Mapping types correctly."""
    template = "Hello {name}! Today is {day}."
    prompt = PromptTemplate.from_template(template)
    
    input_data = ChainMap({"name": "World", "day": "Monday"})
    # Should not raise TypeError
    result = prompt.invoke(input_data)
    assert result.to_string() == "Hello World! Today is Monday."
