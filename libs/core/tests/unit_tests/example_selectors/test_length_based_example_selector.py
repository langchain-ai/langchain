"""Test functionality related to length based selector."""

import pytest

from langchain_core.example_selectors import (
    LengthBasedExampleSelector,
)
from langchain_core.prompts import PromptTemplate

EXAMPLES = [
    {"question": "Question: who are you?\nAnswer: foo"},
    {"question": "Question: who are you?\nAnswer: foo"},
]


@pytest.fixture
def selector() -> LengthBasedExampleSelector:
    """Get length based selector to use in tests."""
    prompts = PromptTemplate(input_variables=["question"], template="{question}")
    return LengthBasedExampleSelector(
        examples=EXAMPLES,
        example_prompt=prompts,
        max_length=30,
    )


def test_selector_valid(selector: LengthBasedExampleSelector) -> None:
    """Test LengthBasedExampleSelector can select examples.."""
    short_question = "Short question?"
    output = selector.select_examples({"question": short_question})
    assert output == EXAMPLES


def test_selector_add_example(selector: LengthBasedExampleSelector) -> None:
    """Test LengthBasedExampleSelector can add an example."""
    new_example = {"question": "Question: what are you?\nAnswer: bar"}
    selector.add_example(new_example)
    short_question = "Short question?"
    output = selector.select_examples({"question": short_question})
    assert output == [*EXAMPLES, new_example]


def test_selector_trims_one_example(selector: LengthBasedExampleSelector) -> None:
    """Test LengthBasedExampleSelector can trim one example."""
    long_question = """I am writing a really long question,
    this probably is going to affect the example right?"""
    output = selector.select_examples({"question": long_question})
    assert output == EXAMPLES[:1]


def test_selector_trims_all_examples(
    selector: LengthBasedExampleSelector,
) -> None:
    """Test LengthBasedExampleSelector can trim all examples."""
    longest_question = """This question is super super super,
    super super super super super super super super super super super,
    super super super super long, this will affect the example right?"""
    output = selector.select_examples({"question": longest_question})
    assert output == []


# edge case
def test_selector_empty_example(
    selector: LengthBasedExampleSelector,
) -> None:
    """Test Empty Example result empty."""
    empty_list: list[dict] = []
    empty_selector = LengthBasedExampleSelector(
        examples=empty_list,
        example_prompt=selector.example_prompt,
        max_length=30,
    )
    output = empty_selector.select_examples({"question":"empty question"})
    assert output == []
