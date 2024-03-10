"""Test OpenAI Function chain."""
import pytest

from langchain.chains.openai_functions.utils import _convert_schema


@pytest.mark.parametrize("input_schema,expected_schema", [
    # In this case we ensure that the title of the main object is preserved in the 
    # schema convertion
    (
        {
            "properties": {
                "name": {
                    "title": "Name",
                    "type": "string"
                }
            },
            "required": [
                "name"
            ],
            "title": "Product that can be purchased",
            "type": "object"
        },
        {
            "type": "object",
            "properties": {
                "name": {
                    "title": "Name",
                    "type": "string"
                }
            },
            "required": [
                "name"
            ],
            "title": "Product that can be purchased"
        }
    ),
])
def test_convert_schema(input_schema, expected_schema) -> None:
    """Checks that the schemas are converted as expected"""
    assert _convert_schema(input_schema) == expected_schema
