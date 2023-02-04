"""Test few shot prompt template."""
import pytest

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["question", "answer"], template="{question}: {answer}"
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
