"""Test PydanticOutputParser"""
from enum import Enum
from typing import Optional

from langchain_core.exceptions import OutputParserException
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.output_parsers.pydantic import PydanticOutputParser


class Actions(Enum):
    SEARCH = "Search"
    CREATE = "Create"
    UPDATE = "Update"
    DELETE = "Delete"


class TestModel(BaseModel):
    action: Actions = Field(description="Action to be performed")
    action_input: str = Field(description="Input to be used in the action")
    additional_fields: Optional[str] = Field(
        description="Additional fields", default=None
    )
    for_new_lines: str = Field(description="To be used to test newlines")


# Prevent pytest from trying to run tests on TestModel
TestModel.__test__ = False  # type: ignore[attr-defined]


DEF_RESULT = """{
    "action": "Update",
    "action_input": "The PydanticOutputParser class is powerful",
    "additional_fields": null,
    "for_new_lines": "not_escape_newline:\n escape_newline: \\n"
}"""

# action 'update' with a lowercase 'u' to test schema validation failure.
DEF_RESULT_FAIL = """{
    "action": "update",
    "action_input": "The PydanticOutputParser class is powerful",
    "additional_fields": null
}"""

DEF_EXPECTED_RESULT = TestModel(
    action=Actions.UPDATE,
    action_input="The PydanticOutputParser class is powerful",
    additional_fields=None,
    for_new_lines="not_escape_newline:\n escape_newline: \n",
)


def test_pydantic_output_parser() -> None:
    """Test PydanticOutputParser."""

    pydantic_parser: PydanticOutputParser[TestModel] = PydanticOutputParser(
        pydantic_object=TestModel
    )

    result = pydantic_parser.parse(DEF_RESULT)
    print("parse_result:", result)
    assert DEF_EXPECTED_RESULT == result


def test_pydantic_output_parser_fail() -> None:
    """Test PydanticOutputParser where completion result fails schema validation."""

    pydantic_parser: PydanticOutputParser[TestModel] = PydanticOutputParser(
        pydantic_object=TestModel
    )

    try:
        pydantic_parser.parse(DEF_RESULT_FAIL)
    except OutputParserException as e:
        print("parse_result:", e)
        assert "Failed to parse TestModel from completion" in str(e)
    else:
        assert False, "Expected OutputParserException"


def test_pydantic_output_parser_type_inference() -> None:
    """Test pydantic output parser type inference."""

    class SampleModel(BaseModel):
        foo: int
        bar: str

    # Ignoring mypy error that appears in python 3.8, but not 3.11.
    # This seems to be functionally correct, so we'll ignore the error.
    pydantic_parser = PydanticOutputParser(
        pydantic_object=SampleModel  # type: ignore[var-annotated]
    )
    schema = pydantic_parser.get_output_schema().schema()

    assert schema == {
        "properties": {
            "bar": {"title": "Bar", "type": "string"},
            "foo": {"title": "Foo", "type": "integer"},
        },
        "required": ["foo", "bar"],
        "title": "SampleModel",
        "type": "object",
    }
