"""Test few shot prompt template."""
from typing import Dict, List, Tuple

import pytest

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["question", "answer"], template="{question}: {answer}"
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


def test_suffix_only() -> None:
    """Test prompt works with just a suffix."""
    suffix = "This is a {foo} test."
    input_variables = ["foo"]
    prompt = FewShotPromptTemplate(
        input_variables=input_variables,
        suffix=suffix,
        examples=[],
        example_prompt=EXAMPLE_PROMPT,
    )
    output = prompt.format(foo="bar")
    expected_output = "This is a bar test."
    assert output == expected_output


def test_prompt_missing_input_variables() -> None:
    """Test error is raised when input variables are not provided."""
    # Test when missing in suffix
    template = "This is a {foo} test."
    with pytest.raises(ValueError):
        FewShotPromptTemplate(
            input_variables=[],
            suffix=template,
            examples=[],
            example_prompt=EXAMPLE_PROMPT,
        )

    # Test when missing in prefix
    template = "This is a {foo} test."
    with pytest.raises(ValueError):
        FewShotPromptTemplate(
            input_variables=[],
            suffix="foo",
            examples=[],
            prefix=template,
            example_prompt=EXAMPLE_PROMPT,
        )


def test_prompt_extra_input_variables() -> None:
    """Test error is raised when there are too many input variables."""
    template = "This is a {foo} test."
    input_variables = ["foo", "bar"]
    with pytest.raises(ValueError):
        FewShotPromptTemplate(
            input_variables=input_variables,
            suffix=template,
            examples=[],
            example_prompt=EXAMPLE_PROMPT,
        )


def test_few_shot_functionality() -> None:
    """Test that few shot works with examples."""
    prefix = "This is a test about {content}."
    suffix = "Now you try to talk about {new_content}."
    examples = [
        {"question": "foo", "answer": "bar"},
        {"question": "baz", "answer": "foo"},
    ]
    prompt = FewShotPromptTemplate(
        suffix=suffix,
        prefix=prefix,
        input_variables=["content", "new_content"],
        examples=examples,
        example_prompt=EXAMPLE_PROMPT,
        example_separator="\n",
    )
    output = prompt.format(content="animals", new_content="party")
    expected_output = (
        "This is a test about animals.\n"
        "foo: bar\n"
        "baz: foo\n"
        "Now you try to talk about party."
    )
    assert output == expected_output


def test_partial_init_string() -> None:
    """Test prompt can be initialized with partial variables."""
    prefix = "This is a test about {content}."
    suffix = "Now you try to talk about {new_content}."
    examples = [
        {"question": "foo", "answer": "bar"},
        {"question": "baz", "answer": "foo"},
    ]
    prompt = FewShotPromptTemplate(
        suffix=suffix,
        prefix=prefix,
        input_variables=["new_content"],
        partial_variables={"content": "animals"},
        examples=examples,
        example_prompt=EXAMPLE_PROMPT,
        example_separator="\n",
    )
    output = prompt.format(new_content="party")
    expected_output = (
        "This is a test about animals.\n"
        "foo: bar\n"
        "baz: foo\n"
        "Now you try to talk about party."
    )
    assert output == expected_output


def test_partial_init_func() -> None:
    """Test prompt can be initialized with partial variables."""
    prefix = "This is a test about {content}."
    suffix = "Now you try to talk about {new_content}."
    examples = [
        {"question": "foo", "answer": "bar"},
        {"question": "baz", "answer": "foo"},
    ]
    prompt = FewShotPromptTemplate(
        suffix=suffix,
        prefix=prefix,
        input_variables=["new_content"],
        partial_variables={"content": lambda: "animals"},
        examples=examples,
        example_prompt=EXAMPLE_PROMPT,
        example_separator="\n",
    )
    output = prompt.format(new_content="party")
    expected_output = (
        "This is a test about animals.\n"
        "foo: bar\n"
        "baz: foo\n"
        "Now you try to talk about party."
    )
    assert output == expected_output


def test_partial() -> None:
    """Test prompt can be partialed."""
    prefix = "This is a test about {content}."
    suffix = "Now you try to talk about {new_content}."
    examples = [
        {"question": "foo", "answer": "bar"},
        {"question": "baz", "answer": "foo"},
    ]
    prompt = FewShotPromptTemplate(
        suffix=suffix,
        prefix=prefix,
        input_variables=["content", "new_content"],
        examples=examples,
        example_prompt=EXAMPLE_PROMPT,
        example_separator="\n",
    )
    new_prompt = prompt.partial(content="foo")
    new_output = new_prompt.format(new_content="party")
    expected_output = (
        "This is a test about foo.\n"
        "foo: bar\n"
        "baz: foo\n"
        "Now you try to talk about party."
    )
    assert new_output == expected_output
    output = prompt.format(new_content="party", content="bar")
    expected_output = (
        "This is a test about bar.\n"
        "foo: bar\n"
        "baz: foo\n"
        "Now you try to talk about party."
    )
    assert output == expected_output


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
