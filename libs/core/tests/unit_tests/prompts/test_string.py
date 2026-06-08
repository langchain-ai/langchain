import pytest
from packaging import version

from langchain_core.prompts.string import (
    check_valid_template,
    get_template_variables,
    mustache_schema,
)
from langchain_core.utils.formatting import formatter
from langchain_core.utils.pydantic import PYDANTIC_VERSION

PYDANTIC_VERSION_AT_LEAST_29 = version.parse("2.9") <= PYDANTIC_VERSION


@pytest.mark.skipif(
    not PYDANTIC_VERSION_AT_LEAST_29,
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


def test_get_template_variables_mustache_nested() -> None:
    template = "Hello {{user.name}}, your role is {{user.role}}"
    template_format = "mustache"
    # Returns only the top-level key for mustache templates
    expected = ["user"]
    actual = get_template_variables(template, template_format)
    assert actual == expected


def test_get_template_variables_rejects_nested_replacement_field_in_format_spec() -> (
    None
):
    template = "{name:{name.__class__.__name__}}"

    with pytest.raises(ValueError, match="Nested replacement fields are not allowed"):
        get_template_variables(template, "f-string")


def test_formatter_rejects_nested_replacement_field_in_format_spec() -> None:
    template = "{name:{name.__class__.__name__}}"

    with pytest.raises(ValueError, match="Invalid format specifier"):
        formatter.format(template, name="hello")


def test_check_valid_template_rejects_nested_replacement_field_in_format_spec() -> None:
    template = "{name:{name.__class__.__name__}}"

    with pytest.raises(ValueError, match="Nested replacement fields are not allowed"):
        check_valid_template(template, "f-string", ["name"])


@pytest.mark.parametrize(
    ("template", "kwargs", "expected_variables", "expected_output"),
    [
        ("{value:.2f}", {"value": 3.14159}, ["value"], "3.14"),
        ("{value:>10}", {"value": "cat"}, ["value"], "       cat"),
        ("{value:*^10}", {"value": "cat"}, ["value"], "***cat****"),
        ("{value:,}", {"value": 1234567}, ["value"], "1,234,567"),
        ("{value:%}", {"value": 0.125}, ["value"], "12.500000%"),
        ("{value!r}", {"value": "cat"}, ["value"], "'cat'"),
    ],
)
def test_f_string_templates_allow_safe_format_specs(
    template: str,
    kwargs: dict[str, object],
    expected_variables: list[str],
    expected_output: str,
) -> None:
    assert get_template_variables(template, "f-string") == expected_variables
    assert formatter.format(template, **kwargs) == expected_output
