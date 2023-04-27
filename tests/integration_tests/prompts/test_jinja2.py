"""Test functionality related to prompts."""
from typing import Dict, List, Tuple

import pytest

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate


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


def test_prompt_jinja2_wrong_input_variables() -> None:
    """Test error is raised when name of input variable is wrong."""
    template = "This is a {{ foo }} test."
    input_variables = ["bar"]
    with pytest.raises(ValueError):
        PromptTemplate(
            input_variables=input_variables, template=template, template_format="jinja2"
        )


@pytest.fixture()
def example_jinja2_prompt() -> Tuple[PromptTemplate, List[Dict[str, str]]]:
    example_template = "{{ word }}: {{ antonym }}"

    examples = [
        {"word": "happy", "antonym": "sad"},
        {"word": "tall", "antonym": "short"},
    ]

    return (
        PromptTemplate(
            input_variables=["word", "antonym"],
            template=example_template,
            template_format="jinja2",
        ),
        examples,
    )


def test_prompt_jinja2_missing_input_variables(
    example_jinja2_prompt: Tuple[PromptTemplate, List[Dict[str, str]]]
) -> None:
    """Test error is raised when input variables are not provided."""
    prefix = "Starting with {{ foo }}"
    suffix = "Ending with {{ bar }}"

    # Test when missing in suffix
    with pytest.raises(ValueError):
        FewShotPromptTemplate(
            input_variables=[],
            suffix=suffix,
            examples=example_jinja2_prompt[1],
            example_prompt=example_jinja2_prompt[0],
            template_format="jinja2",
        )

    # Test when missing in prefix
    with pytest.raises(ValueError):
        FewShotPromptTemplate(
            input_variables=["bar"],
            suffix=suffix,
            prefix=prefix,
            examples=example_jinja2_prompt[1],
            example_prompt=example_jinja2_prompt[0],
            template_format="jinja2",
        )


def test_prompt_jinja2_extra_input_variables(
    example_jinja2_prompt: Tuple[PromptTemplate, List[Dict[str, str]]]
) -> None:
    """Test error is raised when there are too many input variables."""
    prefix = "Starting with {{ foo }}"
    suffix = "Ending with {{ bar }}"
    with pytest.raises(ValueError):
        FewShotPromptTemplate(
            input_variables=["bar", "foo", "extra", "thing"],
            suffix=suffix,
            prefix=prefix,
            examples=example_jinja2_prompt[1],
            example_prompt=example_jinja2_prompt[0],
            template_format="jinja2",
        )


def test_prompt_jinja2_functionality(
    example_jinja2_prompt: Tuple[PromptTemplate, List[Dict[str, str]]]
) -> None:
    prefix = "Starting with {{ foo }}"
    suffix = "Ending with {{ bar }}"

    prompt = FewShotPromptTemplate(
        input_variables=["foo", "bar"],
        suffix=suffix,
        prefix=prefix,
        examples=example_jinja2_prompt[1],
        example_prompt=example_jinja2_prompt[0],
        template_format="jinja2",
    )
    output = prompt.format(foo="hello", bar="bye")
    expected_output = (
        "Starting with hello\n\n" "happy: sad\n\n" "tall: short\n\n" "Ending with bye"
    )

    assert output == expected_output
