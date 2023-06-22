"""Test functionality of converting tools to OpenAI Functions."""
from enum import Enum
from typing import Set, Union

from pydantic import BaseModel, Field

from langchain.agents import Tool
from langchain.tools import StructuredTool
from langchain.tools.convert_to_openai import (
    add_enum_properties,
    format_tool_to_openai_function,
)


class EnumStub1(str, Enum):
    example1 = "example1"
    example2 = "example2"
    example3 = "example3"


class EnumStub2(str, Enum):
    example3 = "example3"
    example4 = "example4"
    example5 = "example5"


class ArgStub(BaseModel):
    standard_enum: EnumStub1
    field_enum: EnumStub1 = Field(default=EnumStub1.example1)
    union_enum: Union[EnumStub1, EnumStub2]
    union_field_enum: Union[EnumStub1, EnumStub2] = Field(default=EnumStub1.example3)
    non_enum: Union[EnumStub1, str]  # type str engulfs EnumStub1


def _test_enum(enum_key: str, result: Set[Union[EnumStub1, EnumStub2]]) -> None:
    arg_stub_schema = ArgStub.schema()

    assert "properties" in arg_stub_schema
    assert "definitions" in arg_stub_schema
    assert enum_key in arg_stub_schema["properties"]

    enum_schema = arg_stub_schema["properties"][enum_key]
    add_enum_properties(
        prop=enum_schema, definitions=arg_stub_schema.get("definitions", {})
    )

    assert set(enum_schema["enum"]) == result


def test_add_standard_enum_properties() -> None:
    """Test that a standard enum fields can be added into functions."""

    # $ref case
    _test_enum(
        "standard_enum",
        {
            EnumStub1.example1,
            EnumStub1.example2,
            EnumStub1.example3,
        },
    )


def test_add_field_enum_properties() -> None:
    """Test that a enum fields with field info can be added into functions."""

    # allOf case
    _test_enum(
        "field_enum",
        {
            EnumStub1.example1,
            EnumStub1.example2,
            EnumStub1.example3,
        },
    )


def test_add_union_enum_properties() -> None:
    """Test union enums fields."""

    # anyOf case
    _test_enum(
        "union_enum",
        {
            EnumStub1.example1,
            EnumStub1.example2,
            EnumStub1.example3,
            EnumStub2.example3,
            EnumStub2.example4,
            EnumStub2.example5,
        },
    )


def test_add_union_field_enum_properties() -> None:
    """Test that a union enums fields with field info can be added into functions."""

    # anyOf case
    _test_enum(
        "union_field_enum",
        {
            EnumStub1.example1,
            EnumStub1.example2,
            EnumStub1.example3,
            EnumStub2.example3,
            EnumStub2.example4,
            EnumStub2.example5,
        },
    )


def test_add_non_enum_properties() -> None:
    """Test that a union that looks like enum but are not will not be added 
    into functions."""
    arg_stub_schema = ArgStub.schema()

    assert "properties" in arg_stub_schema
    assert "definitions" in arg_stub_schema
    assert "non_enum" in arg_stub_schema["properties"]

    enum_schema = arg_stub_schema["properties"]["non_enum"]
    add_enum_properties(
        prop=enum_schema, definitions=arg_stub_schema.get("definitions", {})
    )

    assert "enum" not in enum_schema


def test_tool_to_openai_function() -> None:
    """Test that a tool can be converted into functions."""

    def search(query: str) -> str:
        return ""

    tool = Tool(name="Search", func=search, description="Search tool")
    openai_func_desc = format_tool_to_openai_function(tool)

    assert "name" in openai_func_desc and openai_func_desc["name"] == "Search"
    assert (
        "description" in openai_func_desc
        and openai_func_desc["description"] == "Search tool"
    )
    assert "parameters" in openai_func_desc and openai_func_desc["parameters"] == {
        "type": "object",
        "properties": {
            "__arg1": {"title": "__arg1", "type": "string"},
        },
        "required": ["__arg1"],
    }


def test_structured_tool_to_openai_function() -> None:
    """Test that a structured tool can be converted into functions."""

    def concat(a: Union[EnumStub1, EnumStub2], b: EnumStub1) -> str:
        """Concatenate enum together!"""
        return a + b

    tool = StructuredTool.from_function(func=concat)
    openai_func_desc = format_tool_to_openai_function(tool)

    assert "name" in openai_func_desc and openai_func_desc["name"] == "concat"
    assert "parameters" in openai_func_desc and openai_func_desc["parameters"] == {
        "type": "object",
        "properties": {
            "a": {
                "title": "A",
                "enum": sorted(
                    {
                        EnumStub1.example1,
                        EnumStub1.example2,
                        EnumStub1.example3,
                        EnumStub2.example3,
                        EnumStub2.example4,
                        EnumStub2.example5,
                    }
                ),
            },
            "b": {
                "enum": sorted(
                    {
                        EnumStub1.example1,
                        EnumStub1.example2,
                        EnumStub1.example3,
                    }
                )
            },
        },
        "required": ["a", "b"],
    }
