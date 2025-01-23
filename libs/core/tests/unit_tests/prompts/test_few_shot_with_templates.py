"""Test few shot prompt template."""

import re

import pytest

from langchain_core.prompts.few_shot_with_templates import FewShotPromptWithTemplates
from langchain_core.prompts.prompt import PromptTemplate

EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["question", "answer"], template="{question}: {answer}"
)


async def test_prompttemplate_prefix_suffix() -> None:
    """Test that few shot works when prefix and suffix are PromptTemplates."""
    prefix = PromptTemplate(
        input_variables=["content"], template="This is a test about {content}."
    )
    suffix = PromptTemplate(
        input_variables=["new_content"],
        template="Now you try to talk about {new_content}.",
    )

    examples = [
        {"question": "foo", "answer": "bar"},
        {"question": "baz", "answer": "foo"},
    ]
    prompt = FewShotPromptWithTemplates(
        suffix=suffix,
        prefix=prefix,
        input_variables=["content", "new_content"],
        examples=examples,
        example_prompt=EXAMPLE_PROMPT,
        example_separator="\n",
    )
    expected_output = (
        "This is a test about animals.\n"
        "foo: bar\n"
        "baz: foo\n"
        "Now you try to talk about party."
    )
    output = prompt.format(content="animals", new_content="party")
    assert output == expected_output
    output = await prompt.aformat(content="animals", new_content="party")
    assert output == expected_output


def test_prompttemplate_validation() -> None:
    """Test that few shot works when prefix and suffix are PromptTemplates."""
    prefix = PromptTemplate(
        input_variables=["content"], template="This is a test about {content}."
    )
    suffix = PromptTemplate(
        input_variables=["new_content"],
        template="Now you try to talk about {new_content}.",
    )

    examples = [
        {"question": "foo", "answer": "bar"},
        {"question": "baz", "answer": "foo"},
    ]
    with pytest.raises(
        ValueError,
        match=re.escape("Got input_variables=[], but based on prefix/suffix expected"),
    ):
        FewShotPromptWithTemplates(
            suffix=suffix,
            prefix=prefix,
            input_variables=[],
            examples=examples,
            example_prompt=EXAMPLE_PROMPT,
            example_separator="\n",
            validate_template=True,
        )
    assert FewShotPromptWithTemplates(
        suffix=suffix,
        prefix=prefix,
        input_variables=[],
        examples=examples,
        example_prompt=EXAMPLE_PROMPT,
        example_separator="\n",
    ).input_variables == ["content", "new_content"]
