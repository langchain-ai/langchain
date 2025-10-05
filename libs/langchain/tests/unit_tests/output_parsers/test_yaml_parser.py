"""Test yamlOutputParser."""

from enum import Enum
from typing import Optional

import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field

from langchain_classic.output_parsers.yaml import YamlOutputParser


class Actions(Enum):
    SEARCH = "Search"
    CREATE = "Create"
    UPDATE = "Update"
    DELETE = "Delete"


class TestModel(BaseModel):
    action: Actions = Field(description="Action to be performed")
    action_input: str = Field(description="Input to be used in the action")
    additional_fields: Optional[str] = Field(
        description="Additional fields",
        default=None,
    )
    for_new_lines: str = Field(description="To be used to test newlines")


# Prevent pytest from trying to run tests on TestModel
TestModel.__test__ = False  # type: ignore[attr-defined]


DEF_RESULT = """```yaml
---

action: Update
action_input: The yamlOutputParser class is powerful
additional_fields: null
for_new_lines: |
  not_escape_newline:
   escape_newline:

```"""
DEF_RESULT_NO_BACKTICKS = """
action: Update
action_input: The yamlOutputParser class is powerful
additional_fields: null
for_new_lines: |
  not_escape_newline:
   escape_newline:

"""

# action 'update' with a lowercase 'u' to test schema validation failure.
DEF_RESULT_FAIL = """```yaml
action: update
action_input: The yamlOutputParser class is powerful
additional_fields: null
```"""

DEF_EXPECTED_RESULT = TestModel(
    action=Actions.UPDATE,
    action_input="The yamlOutputParser class is powerful",
    additional_fields=None,
    for_new_lines="not_escape_newline:\n escape_newline:\n",
)


@pytest.mark.parametrize("result", [DEF_RESULT, DEF_RESULT_NO_BACKTICKS])
def test_yaml_output_parser(result: str) -> None:
    """Test yamlOutputParser."""
    yaml_parser: YamlOutputParser[TestModel] = YamlOutputParser(
        pydantic_object=TestModel,
    )

    model = yaml_parser.parse(result)
    print("parse_result:", result)  # noqa: T201
    assert model == DEF_EXPECTED_RESULT


def test_yaml_output_parser_fail() -> None:
    """Test YamlOutputParser where completion result fails schema validation."""
    yaml_parser: YamlOutputParser[TestModel] = YamlOutputParser(
        pydantic_object=TestModel,
    )

    with pytest.raises(OutputParserException) as exc_info:
        yaml_parser.parse(DEF_RESULT_FAIL)

    assert "Failed to parse TestModel from completion" in str(exc_info.value)


def test_yaml_output_parser_output_type() -> None:
    """Test YamlOutputParser OutputType."""
    yaml_parser = YamlOutputParser[TestModel](pydantic_object=TestModel)
    assert yaml_parser.OutputType is TestModel
