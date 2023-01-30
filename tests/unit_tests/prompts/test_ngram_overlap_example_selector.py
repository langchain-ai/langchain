"""Test functionality related to ngram overlap based selector."""

import pytest

from langchain.prompts.example_selector.ngram_overlap import NGramOverlapExampleSelector
from langchain.prompts.prompt import PromptTemplate

EXAMPLES = [
    {"input": "See Spot run.", "output": "foo1"},
    {"input": "My dog barks.", "output": "foo2"},
    {"input": "Spot can run.", "output": "foo3"},
]


@pytest.fixture
def selector() -> NGramOverlapExampleSelector:
    """Get ngram overlap based selector to use in tests."""
    prompts = PromptTemplate(
        input_variables=["input", "output"], template="Input: {input}\nOutput: {output}"
    )
    selector = NGramOverlapExampleSelector(
        examples=EXAMPLES,
        example_prompt=prompts,
    )
    return selector


def test_selector_valid(selector: NGramOverlapExampleSelector) -> None:
    """Test NGramOverlapExampleSelector can select examples."""
    sentence = "Spot can run."
    output = selector.select_examples({"input": sentence})
    assert output == [EXAMPLES[2], EXAMPLES[0]]


def test_selector_add_example(selector: NGramOverlapExampleSelector) -> None:
    """Test NGramOverlapExampleSelector can add an example."""
    new_example = {"input": "Spot plays fetch.", "output": "foo4"}
    selector.add_example(new_example)
    sentence = "Spot can run."
    output = selector.select_examples({"input": sentence})
    assert output == [EXAMPLES[2], EXAMPLES[0]] + [new_example]
