"""Test functionality related to prompts."""

from typing import Any, Union
from unittest import mock

import pydantic
import pytest
from syrupy import SnapshotAssertion

from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import PromptTemplateFormat
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from tests.unit_tests.pydantic_utils import _normalize_schema

PYDANTIC_VERSION = tuple(map(int, pydantic.__version__.split(".")))


def test_prompt_valid() -> None:
    """Test prompts can be constructed."""
    template = "This is a {foo} test."
    input_variables = ["foo"]
    prompt = PromptTemplate(input_variables=input_variables, template=template)
    assert prompt.template == template
    assert prompt.input_variables == input_variables


def test_from_file_encoding() -> None:
    """Test that we can load a template from a file with a non utf-8 encoding."""
    template = "This is a {foo} test with special character €."
    input_variables = ["foo"]

    # First write to a file using CP-1252 encoding.
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(delete=True, mode="w", encoding="cp1252") as f:
        f.write(template)
        f.flush()
        file_name = f.name

        # Now read from the file using CP-1252 encoding and test
        prompt = PromptTemplate.from_file(file_name, encoding="cp1252")
        assert prompt.template == template
        assert prompt.input_variables == input_variables

        # Now read from the file using UTF-8 encoding and test
        with pytest.raises(UnicodeDecodeError):
            PromptTemplate.from_file(file_name, encoding="utf-8")


def test_prompt_from_template() -> None:
    """Test prompts can be constructed from a template."""
    # Single input variable.
    template = "This is a {foo} test."
    prompt = PromptTemplate.from_template(template)
    expected_prompt = PromptTemplate(template=template, input_variables=["foo"])
    assert prompt == expected_prompt

    # Multiple input variables.
    template = "This {bar} is a {foo} test."
    prompt = PromptTemplate.from_template(template)
    expected_prompt = PromptTemplate(template=template, input_variables=["bar", "foo"])
    assert prompt == expected_prompt

    # Multiple input variables with repeats.
    template = "This {bar} is a {foo} test {foo}."
    prompt = PromptTemplate.from_template(template)
    expected_prompt = PromptTemplate(template=template, input_variables=["bar", "foo"])
    assert prompt == expected_prompt


def test_mustache_prompt_from_template(snapshot: SnapshotAssertion) -> None:
    """Test prompts can be constructed from a template."""
    # Single input variable.
    template = "This is a {{foo}} test."
    prompt = PromptTemplate.from_template(template, template_format="mustache")
    assert prompt.format(foo="bar") == "This is a bar test."
    assert prompt.input_variables == ["foo"]
    assert prompt.get_input_jsonschema() == {
        "title": "PromptInput",
        "type": "object",
        "properties": {"foo": {"title": "Foo", "type": "string", "default": None}},
    }

    # Multiple input variables.
    template = "This {{bar}} is a {{foo}} test."
    prompt = PromptTemplate.from_template(template, template_format="mustache")
    assert prompt.format(bar="baz", foo="bar") == "This baz is a bar test."
    assert prompt.input_variables == ["bar", "foo"]
    assert prompt.get_input_jsonschema() == {
        "title": "PromptInput",
        "type": "object",
        "properties": {
            "bar": {"title": "Bar", "type": "string", "default": None},
            "foo": {"title": "Foo", "type": "string", "default": None},
        },
    }

    # Multiple input variables with repeats.
    template = "This {{bar}} is a {{foo}} test {{&foo}}."
    prompt = PromptTemplate.from_template(template, template_format="mustache")
    assert prompt.format(bar="baz", foo="bar") == "This baz is a bar test bar."
    assert prompt.input_variables == ["bar", "foo"]
    assert prompt.get_input_jsonschema() == {
        "title": "PromptInput",
        "type": "object",
        "properties": {
            "bar": {"title": "Bar", "type": "string", "default": None},
            "foo": {"title": "Foo", "type": "string", "default": None},
        },
    }

    # Nested variables.
    template = "This {{obj.bar}} is a {{obj.foo}} test {{{foo}}}."
    prompt = PromptTemplate.from_template(template, template_format="mustache")
    assert prompt.format(obj={"bar": "foo", "foo": "bar"}, foo="baz") == (
        "This foo is a bar test baz."
    )
    assert prompt.input_variables == ["foo", "obj"]
    if PYDANTIC_VERSION >= (2, 9):
        assert _normalize_schema(prompt.get_input_jsonschema()) == snapshot(
            name="schema_0"
        )

    # . variables
    template = "This {{.}} is a test."
    prompt = PromptTemplate.from_template(template, template_format="mustache")
    assert prompt.format(foo="baz") == ("This {'foo': 'baz'} is a test.")
    assert prompt.input_variables == []
    assert prompt.get_input_jsonschema() == {
        "title": "PromptInput",
        "type": "object",
        "properties": {},
    }

    # section/context variables
    template = """This{{#foo}}
        {{bar}}
    {{/foo}}is a test."""
    prompt = PromptTemplate.from_template(template, template_format="mustache")
    assert prompt.format(foo={"bar": "yo"}) == (
        """This
        yo
    is a test."""
    )
    assert prompt.input_variables == ["foo"]
    if PYDANTIC_VERSION >= (2, 9):
        assert _normalize_schema(prompt.get_input_jsonschema()) == snapshot(
            name="schema_2"
        )

    # more complex nested section/context variables
    template = """This{{#foo}}
        {{bar}}
        {{#baz}}
            {{qux}}
        {{/baz}}
        {{quux}}
    {{/foo}}is a test."""
    prompt = PromptTemplate.from_template(template, template_format="mustache")
    assert prompt.format(
        foo={"bar": "yo", "baz": [{"qux": "wassup"}], "quux": "hello"}
    ) == (
        """This
        yo
            wassup
        hello
    is a test."""
    )
    assert prompt.input_variables == ["foo"]
    if PYDANTIC_VERSION >= (2, 9):
        assert _normalize_schema(prompt.get_input_jsonschema()) == snapshot(
            name="schema_3"
        )

    # triply nested section/context variables
    template = """This{{#foo}}
        {{bar}}
        {{#baz.qux}}
            {{#barfoo}}
                {{foobar}}
            {{/barfoo}}
            {{foobar}}
        {{/baz.qux}}
        {{quux}}
    {{/foo}}is a test."""
    prompt = PromptTemplate.from_template(template, template_format="mustache")
    assert prompt.format(
        foo={
            "bar": "yo",
            "baz": {
                "qux": [
                    {"foobar": "wassup"},
                    {"foobar": "yoyo", "barfoo": {"foobar": "hello there"}},
                ]
            },
            "quux": "hello",
        }
    ) == (
        """This
        yo
            wassup
                hello there
            yoyo
        hello
    is a test."""
    )
    assert prompt.input_variables == ["foo"]
    if PYDANTIC_VERSION >= (2, 9):
        assert _normalize_schema(prompt.get_input_jsonschema()) == snapshot(
            name="schema_4"
        )

    # section/context variables with repeats
    template = """This{{#foo}}
        {{bar}}
    {{/foo}}is a test."""
    prompt = PromptTemplate.from_template(template, template_format="mustache")
    assert prompt.format(foo=[{"bar": "yo"}, {"bar": "hello"}]) == (
        """This
        yo
    
        hello
    is a test."""  # noqa: W293
    )
    assert prompt.input_variables == ["foo"]
    if PYDANTIC_VERSION >= (2, 9):
        assert _normalize_schema(prompt.get_input_jsonschema()) == snapshot(
            name="schema_5"
        )
    template = """This{{^foo}}
        no foos
    {{/foo}}is a test."""
    prompt = PromptTemplate.from_template(template, template_format="mustache")
    assert prompt.format() == (
        """This
        no foos
    is a test."""
    )
    assert prompt.input_variables == ["foo"]
    assert prompt.get_input_jsonschema() == {
        "properties": {"foo": {"default": None, "title": "Foo", "type": "object"}},
        "title": "PromptInput",
        "type": "object",
    }


def test_prompt_from_template_with_partial_variables() -> None:
    """Test prompts can be constructed from a template with partial variables."""
    # given
    template = "This is a {foo} test {bar}."
    partial_variables = {"bar": "baz"}
    # when
    prompt = PromptTemplate.from_template(template, partial_variables=partial_variables)
    # then
    expected_prompt = PromptTemplate(
        template=template,
        input_variables=["foo"],
        partial_variables=partial_variables,
    )
    assert prompt == expected_prompt


def test_prompt_missing_input_variables() -> None:
    """Test error is raised when input variables are not provided."""
    template = "This is a {foo} test."
    input_variables: list = []
    with pytest.raises(ValueError):
        PromptTemplate(
            input_variables=input_variables, template=template, validate_template=True
        )
    assert PromptTemplate(
        input_variables=input_variables, template=template
    ).input_variables == ["foo"]


def test_prompt_empty_input_variable() -> None:
    """Test error is raised when empty string input variable."""
    with pytest.raises(ValueError):
        PromptTemplate(input_variables=[""], template="{}", validate_template=True)


def test_prompt_wrong_input_variables() -> None:
    """Test error is raised when name of input variable is wrong."""
    template = "This is a {foo} test."
    input_variables = ["bar"]
    with pytest.raises(ValueError):
        PromptTemplate(
            input_variables=input_variables, template=template, validate_template=True
        )
    assert PromptTemplate(
        input_variables=input_variables, template=template
    ).input_variables == ["foo"]


def test_prompt_from_examples_valid() -> None:
    """Test prompt can be successfully constructed from examples."""
    template = """Test Prompt:

Question: who are you?
Answer: foo

Question: what are you?
Answer: bar

Question: {question}
Answer:"""
    input_variables = ["question"]
    example_separator = "\n\n"
    prefix = """Test Prompt:"""
    suffix = """Question: {question}\nAnswer:"""
    examples = [
        """Question: who are you?\nAnswer: foo""",
        """Question: what are you?\nAnswer: bar""",
    ]
    prompt_from_examples = PromptTemplate.from_examples(
        examples,
        suffix,
        input_variables,
        example_separator=example_separator,
        prefix=prefix,
    )
    prompt_from_template = PromptTemplate(
        input_variables=input_variables, template=template
    )
    assert prompt_from_examples.template == prompt_from_template.template
    assert prompt_from_examples.input_variables == prompt_from_template.input_variables


def test_prompt_invalid_template_format() -> None:
    """Test initializing a prompt with invalid template format."""
    template = "This is a {foo} test."
    input_variables = ["foo"]
    with pytest.raises(ValueError):
        PromptTemplate(
            input_variables=input_variables,
            template=template,
            template_format="bar",  # type: ignore[arg-type]
        )


def test_prompt_from_file() -> None:
    """Test prompt can be successfully constructed from a file."""
    template_file = "tests/unit_tests/data/prompt_file.txt"
    input_variables = ["question"]
    prompt = PromptTemplate.from_file(template_file, input_variables)
    assert prompt.template == "Question: {question}\nAnswer:"


def test_prompt_from_file_with_partial_variables() -> None:
    """Test prompt can be successfully constructed from a file
    with partial variables."""
    # given
    template = "This is a {foo} test {bar}."
    partial_variables = {"bar": "baz"}
    # when
    with mock.patch("builtins.open", mock.mock_open(read_data=template)):
        prompt = PromptTemplate.from_file(
            "mock_file_name", partial_variables=partial_variables
        )
    # then
    expected_prompt = PromptTemplate(
        template=template,
        input_variables=["foo"],
        partial_variables=partial_variables,
    )
    assert prompt == expected_prompt


def test_partial_init_string() -> None:
    """Test prompt can be initialized with partial variables."""
    template = "This is a {foo} test."
    prompt = PromptTemplate(
        input_variables=[], template=template, partial_variables={"foo": 1}
    )
    assert prompt.template == template
    assert prompt.input_variables == []
    result = prompt.format()
    assert result == "This is a 1 test."


def test_partial_init_func() -> None:
    """Test prompt can be initialized with partial variables."""
    template = "This is a {foo} test."
    prompt = PromptTemplate(
        input_variables=[], template=template, partial_variables={"foo": lambda: 2}
    )
    assert prompt.template == template
    assert prompt.input_variables == []
    result = prompt.format()
    assert result == "This is a 2 test."


def test_partial() -> None:
    """Test prompt can be partialed."""
    template = "This is a {foo} test."
    prompt = PromptTemplate(input_variables=["foo"], template=template)
    assert prompt.template == template
    assert prompt.input_variables == ["foo"]
    new_prompt = prompt.partial(foo="3")
    new_result = new_prompt.format()
    assert new_result == "This is a 3 test."
    result = prompt.format(foo="foo")
    assert result == "This is a foo test."


def test_nested_prompt_template_as_partial() -> None:
    """Test prompt with PromptTemplate as partial variable."""
    template_nested = "{bar}"
    prompt_nested = PromptTemplate(input_variables=["bar"], template=template_nested)

    template = "This is a {foo} test."
    prompt = PromptTemplate(input_variables=["foo"], template=template)
    assert prompt.template == template
    assert prompt.input_variables == ["foo"]

    new_prompt = prompt.partial(foo=prompt_nested)
    assert new_prompt.input_variables == []
    assert new_prompt.partial_variables["foo"].input_variables == ["bar"]
    assert new_prompt.partial_variables["foo"].partial_variables == {}
    result = new_prompt.format(bar="bar")
    assert result == "This is a bar test."

    new_prompt = prompt.partial(foo=prompt_nested, bar="bar")
    assert new_prompt.input_variables == []
    assert new_prompt.partial_variables["foo"].input_variables == []
    assert new_prompt.partial_variables["foo"].partial_variables == {"bar": "bar"}
    result = new_prompt.format()
    assert result == "This is a bar test."


def test_nested_prompt_template_with_shared_variable() -> None:
    """Test prompt with PromptTemplate as partial variable, sharing another variable."""
    template_nested = "{bar}"
    prompt_nested = PromptTemplate(
        input_variables=["bar", "foo"], template=template_nested
    )

    template = "This is a {foo} {bar} test."
    prompt = PromptTemplate(input_variables=["foo", "bar"], template=template)
    assert prompt.template == template
    assert prompt.input_variables == ["bar", "foo"]

    new_prompt = prompt.partial(foo=prompt_nested)
    assert new_prompt.input_variables == ["bar"]
    assert new_prompt.partial_variables["foo"].input_variables == ["bar"]
    assert new_prompt.partial_variables["foo"].partial_variables == {}
    result = new_prompt.format(bar="bar")
    assert result == "This is a bar bar test."

    new_prompt = prompt.partial(foo=prompt_nested, bar="bar")
    assert new_prompt.input_variables == []
    assert new_prompt.partial_variables["foo"].input_variables == []
    assert new_prompt.partial_variables["foo"].partial_variables == {"bar": "bar"}
    result = new_prompt.format()
    assert result == "This is a bar bar test."


@pytest.mark.requires("jinja2")
def test_prompt_from_jinja2_template() -> None:
    """Test prompts can be constructed from a jinja2 template."""
    # Empty input variable.
    template = """Hello there
There is no variable here {
Will it get confused{ }?
    """
    prompt = PromptTemplate.from_template(template, template_format="jinja2")
    expected_prompt = PromptTemplate(
        template=template, input_variables=[], template_format="jinja2"
    )
    assert prompt == expected_prompt


@pytest.mark.requires("jinja2")
def test_basic_sandboxing_with_jinja2() -> None:
    """Test basic sandboxing with jinja2."""
    import jinja2

    template = " {{''.__class__.__bases__[0] }} "  # malicious code
    prompt = PromptTemplate.from_template(template, template_format="jinja2")
    with pytest.raises(jinja2.exceptions.SecurityError):
        assert prompt.format() == []


@pytest.mark.requires("jinja2")
def test_prompt_from_jinja2_template_multiple_inputs() -> None:
    """Test with multiple input variables."""
    # Multiple input variables.
    template = """\
Hello world

Your variable: {{ foo }}

{# This will not get rendered #}

{% if bar %}
You just set bar boolean variable to true
{% endif %}

{% for i in foo_list %}
{{ i }}
{% endfor %}
"""
    prompt = PromptTemplate.from_template(template, template_format="jinja2")
    expected_prompt = PromptTemplate(
        template=template,
        input_variables=["bar", "foo", "foo_list"],
        template_format="jinja2",
    )

    assert prompt == expected_prompt


@pytest.mark.requires("jinja2")
def test_prompt_from_jinja2_template_multiple_inputs_with_repeats() -> None:
    """Test with multiple input variables and repeats."""
    template = """\
Hello world

Your variable: {{ foo }}

{# This will not get rendered #}

{% if bar %}
You just set bar boolean variable to true
{% endif %}

{% for i in foo_list %}
{{ i }}
{% endfor %}

{% if bar %}
Your variable again: {{ foo }}
{% endif %}
"""
    prompt = PromptTemplate.from_template(template, template_format="jinja2")
    expected_prompt = PromptTemplate(
        template=template,
        input_variables=["bar", "foo", "foo_list"],
        template_format="jinja2",
    )
    assert prompt == expected_prompt


@pytest.mark.requires("jinja2")
def test_prompt_jinja2_missing_input_variables() -> None:
    """Test error is raised when input variables are not provided."""
    template = "This is a {{ foo }} test."
    input_variables: list = []
    with pytest.warns(UserWarning):
        PromptTemplate(
            input_variables=input_variables,
            template=template,
            template_format="jinja2",
            validate_template=True,
        )
    assert PromptTemplate(
        input_variables=input_variables, template=template, template_format="jinja2"
    ).input_variables == ["foo"]


@pytest.mark.requires("jinja2")
def test_prompt_jinja2_extra_input_variables() -> None:
    """Test warning is raised when there are too many input variables."""
    template = "This is a {{ foo }} test."
    input_variables = ["foo", "bar"]
    with pytest.warns(UserWarning):
        PromptTemplate(
            input_variables=input_variables,
            template=template,
            template_format="jinja2",
            validate_template=True,
        )
    assert PromptTemplate(
        input_variables=input_variables, template=template, template_format="jinja2"
    ).input_variables == ["foo"]


@pytest.mark.requires("jinja2")
def test_prompt_jinja2_wrong_input_variables() -> None:
    """Test warning is raised when name of input variable is wrong."""
    template = "This is a {{ foo }} test."
    input_variables = ["bar"]
    with pytest.warns(UserWarning):
        PromptTemplate(
            input_variables=input_variables,
            template=template,
            template_format="jinja2",
            validate_template=True,
        )
    assert PromptTemplate(
        input_variables=input_variables, template=template, template_format="jinja2"
    ).input_variables == ["foo"]


def test_prompt_invoke_with_metadata() -> None:
    """Test prompt can be invoked with metadata."""
    template = "This is a {foo} test."
    prompt = PromptTemplate(
        input_variables=["foo"],
        template=template,
        metadata={"version": "1"},
        tags=["tag1", "tag2"],
    )
    tracer = RunCollectorCallbackHandler()
    result = prompt.invoke(
        {"foo": "bar"}, {"metadata": {"foo": "bar"}, "callbacks": [tracer]}
    )
    assert result.to_string() == "This is a bar test."
    assert len(tracer.traced_runs) == 1
    assert tracer.traced_runs[0].extra["metadata"] == {"version": "1", "foo": "bar"}  # type: ignore
    assert tracer.traced_runs[0].tags == ["tag1", "tag2"]  # type: ignore


async def test_prompt_ainvoke_with_metadata() -> None:
    """Test prompt can be invoked with metadata."""
    template = "This is a {foo} test."
    prompt = PromptTemplate(
        input_variables=["foo"],
        template=template,
        metadata={"version": "1"},
        tags=["tag1", "tag2"],
    )
    tracer = RunCollectorCallbackHandler()
    result = await prompt.ainvoke(
        {"foo": "bar"}, {"metadata": {"foo": "bar"}, "callbacks": [tracer]}
    )
    assert result.to_string() == "This is a bar test."
    assert len(tracer.traced_runs) == 1
    assert tracer.traced_runs[0].extra["metadata"] == {"version": "1", "foo": "bar"}  # type: ignore
    assert tracer.traced_runs[0].tags == ["tag1", "tag2"]  # type: ignore


@pytest.mark.parametrize(
    "value, expected",
    [
        ("0", "0"),
        (0, "0"),
        (0.0, "0.0"),
        (False, "False"),
        ("", ""),
        (
            None,
            {
                "mustache": "",
                "f-string": "None",
            },
        ),
        (
            [],
            {
                "mustache": "",
                "f-string": "[]",
            },
        ),
        (
            {},
            {
                "mustache": "",
                "f-string": "{}",
            },
        ),
    ],
)
@pytest.mark.parametrize("template_format", ["f-string", "mustache"])
def test_prompt_falsy_vars(
    template_format: PromptTemplateFormat,
    value: Any,
    expected: Union[str, dict[str, str]],
) -> None:
    # each line is value, f-string, mustache
    if template_format == "f-string":
        template = "{my_var}"
    elif template_format == "mustache":
        template = "{{my_var}}"
    else:
        msg = f"Invalid template format: {template_format}"
        raise ValueError(msg)

    prompt = PromptTemplate.from_template(template, template_format=template_format)

    result = prompt.invoke({"my_var": value})

    expected_output = (
        expected if not isinstance(expected, dict) else expected[template_format]
    )
    assert result.to_string() == expected_output


def test_prompt_missing_vars_error() -> None:
    prompt = PromptTemplate.from_template("This is a {foo} {goingtobemissing} test.")
    with pytest.raises(KeyError) as e:
        prompt.invoke({"foo": "bar"})

    # Check that the error message contains the missing variable
    assert "{'goingtobemissing'}" in str(e.value.args[0])

    # Check helper text has right number of braces
    assert "'{{goingtobemissing}}'" in str(e.value.args[0])


def test_prompt_with_template_variable_name_fstring() -> None:
    template = "This is a {template} test."
    prompt = PromptTemplate.from_template(template, template_format="f-string")
    assert prompt.invoke({"template": "bar"}).to_string() == "This is a bar test."


def test_prompt_with_template_variable_name_mustache() -> None:
    template = "This is a {{template}} test."
    prompt = PromptTemplate.from_template(template, template_format="mustache")
    assert prompt.invoke({"template": "bar"}).to_string() == "This is a bar test."


@pytest.mark.requires("jinja2")
def test_prompt_with_template_variable_name_jinja2() -> None:
    template = "This is a {{template}} test."
    prompt = PromptTemplate.from_template(template, template_format="jinja2")
    assert prompt.invoke({"template": "bar"}).to_string() == "This is a bar test."
