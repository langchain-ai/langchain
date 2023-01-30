"""Test functionality related to ngram overlap based selector."""

import pytest

from langchain.prompts.example_selector.ngram_overlap import (
    NGramOverlapExampleSelector,
    ngram_overlap_score,
)
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
    assert output == [EXAMPLES[2], EXAMPLES[0], EXAMPLES[1]]


def test_selector_add_example(selector: NGramOverlapExampleSelector) -> None:
    """Test NGramOverlapExampleSelector can add an example."""
    new_example = {"input": "Spot plays fetch.", "output": "foo4"}
    selector.add_example(new_example)
    sentence = "Spot can run."
    output = selector.select_examples({"input": sentence})
    assert output == [EXAMPLES[2], EXAMPLES[0]] + [new_example] + [EXAMPLES[1]]


def test_selector_threshold_zero(selector: NGramOverlapExampleSelector) -> None:
    """Tests NGramOverlapExampleSelector threshold set to 0.0."""
    selector.threshold = 0.0
    sentence = "Spot can run."
    output = selector.select_examples({"input": sentence})
    assert output == [EXAMPLES[2], EXAMPLES[0]]


def test_selector_threshold_more_than_one(
    selector: NGramOverlapExampleSelector,
) -> None:
    """Tests NGramOverlapExampleSelector threshold greater than 1.0."""
    selector.threshold = 1.0 + 1e-9
    sentence = "Spot can run."
    output = selector.select_examples({"input": sentence})
    assert output == []


def test_ngram_overlap_score(selector: NGramOverlapExampleSelector) -> None:
    """Tests that ngram_overlap_score returns correct values."""
    selector.threshold = 1.0 + 1e-9
    none = ngram_overlap_score(["Spot can run."], ["My dog barks."])
    some = ngram_overlap_score(["Spot can run."], ["See Spot run."])
    complete = ngram_overlap_score(["Spot can run."], ["Spot can run."])

    check = [abs(none - 0.0) < 1e-9, 0.0 < some < 1.0, abs(complete - 1.0) < 1e-9]
    assert check == [True, True, True]
