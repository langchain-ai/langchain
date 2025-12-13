from langchain_core.tools import tool
from pydantic import BaseModel

def test_tool_docstring_no_inheritance() -> None:
    """Test that class-based tools do not inherit docstrings from parents."""

    class ParentTool(BaseModel):
        """Parent Tool Description"""
        foo: str

    @tool
    class ChildTool(ParentTool):
        bar: str

    # Assert that the description is NOT the parent's docstring
    # It should be empty or default, but definitely not "Parent Tool Description"
    assert "Parent Tool Description" not in ChildTool.description
