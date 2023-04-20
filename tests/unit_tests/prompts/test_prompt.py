"""Test functionality related to prompts."""
import pytest

from langchain.prompts.prompt import PromptTemplate


def test_prompt_valid() -> None:
    """Test prompts can be constructed."""
    template = "This is a {foo} test."
    input_variables = ["foo"]
    prompt = PromptTemplate(input_variables=input_variables, template=template)
    assert prompt.template == template
    assert prompt.input_variables == input_variables


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


def test_prompt_missing_input_variables() -> None:
    """Test error is raised when input variables are not provided."""
    template = "This is a {foo} test."
    input_variables: list = []
    with pytest.raises(ValueError):
        PromptTemplate(input_variables=input_variables, template=template)


def test_prompt_extra_input_variables() -> None:
    """Test error is raised when there are too many input variables."""
    template = "This is a {foo} test."
    input_variables = ["foo", "bar"]
    with pytest.raises(ValueError):
        PromptTemplate(input_variables=input_variables, template=template)


def test_prompt_wrong_input_variables() -> None:
    """Test error is raised when name of input variable is wrong."""
    template = "This is a {foo} test."
    input_variables = ["bar"]
    with pytest.raises(ValueError):
        PromptTemplate(input_variables=input_variables, template=template)


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
            input_variables=input_variables, template=template, template_format="bar"
        )


def test_prompt_from_file() -> None:
    """Test prompt can be successfully constructed from a file."""
    template_file = "tests/unit_tests/data/prompt_file.txt"
    input_variables = ["question"]
    prompt = PromptTemplate.from_file(template_file, input_variables)
    assert prompt.template == "Question: {question}\nAnswer:"


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

    # Multiple input variables with repeats.
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


def test_prompt_jinja2_missing_input_variables() -> None:
    """Test error is raised when input variables are not provided."""
    template = "This is a {{ foo }} test."
    input_variables: list = []
    with pytest.raises(ValueError):
        PromptTemplate(
            input_variables=input_variables, template=template, template_format="jinja2"
        )


def test_prompt_jinja2_extra_input_variables() -> None:
    """Test error is raised when there are too many input variables."""
    template = "This is a {{ foo }} test."
    input_variables = ["foo", "bar"]
    with pytest.raises(ValueError):
        PromptTemplate(
            input_variables=input_variables, template=template, template_format="jinja2"
        )


def test_prompt_jinja2_wrong_input_variables() -> None:
    """Test error is raised when name of input variable is wrong."""
    template = "This is a {{ foo }} test."
    input_variables = ["bar"]
    with pytest.raises(ValueError):
        PromptTemplate(
            input_variables=input_variables, template=template, template_format="jinja2"
        )
