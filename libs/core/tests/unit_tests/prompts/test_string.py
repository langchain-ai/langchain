import pytest
from packaging import version

from langchain_core.prompts.string import mustache_schema
from langchain_core.utils.pydantic import PYDANTIC_VERSION

PYDANTIC_VERSION_AT_LEAST_210 = version.parse("2.10") <= PYDANTIC_VERSION


@pytest.mark.skipif(
    PYDANTIC_VERSION_AT_LEAST_210,
    reason=(
        "Only test with most recent version of pydantic. "
        "Pydantic introduced small fixes to generated JSONSchema on minor versions."
    ),
)
def test_mustache_schema_parent_child() -> None:
    template = "{{x.y}} {{x}}"
    expected = {
        "$defs": {
            "x": {
                "properties": {"y": {"default": None, "title": "Y", "type": "string"}},
                "title": "x",
                "type": "object",
            }
        },
        "properties": {"x": {"$ref": "#/$defs/x", "default": None}},
        "title": "PromptInput",
        "type": "object",
    }
    actual = mustache_schema(template).model_json_schema()
    assert expected == actual
