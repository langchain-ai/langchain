\"\"\"
Unit tests for PydanticOutputParser schema instructions and validation bounds in langchain_core.output_parsers.
\"\"\"
import pytest
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class UserRecord(BaseModel):
    name: str = Field(description="User's full name")
    age: int = Field(description="User's age in years")

def test_pydantic_output_parser_format_instructions():
    parser = PydanticOutputParser(pydantic_object=UserRecord)
    instructions = parser.get_format_instructions()
    assert "UserRecord" in instructions or "name" in instructions
    assert "age" in instructions

def test_pydantic_output_parser_valid_json():
    parser = PydanticOutputParser(pydantic_object=UserRecord)
    raw = '{\"name\": \"Alice\", \"age\": 30}'
    result = parser.invoke(raw)
    assert isinstance(result, UserRecord)
    assert result.name == "Alice"
    assert result.age == 30