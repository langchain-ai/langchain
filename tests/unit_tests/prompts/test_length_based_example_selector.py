"""Test functionality related to dynamic prompts."""
import pytest

from langchain.prompts.example_selector.length_based import LengthBasedExampleSelector
from langchain.prompts.prompt import PromptTemplate

EXAMPLES = [
    {"question": "Question: who are you?\nAnswer: foo"},
    {"question": "Question: who are you?\nAnswer: foo"},
]


@pytest.fixture
def selector() -> LengthBasedExampleSelector:
    """Get length based selector to use in tests."""
    prompts = PromptTemplate(input_variables=["question"], template="{question}")
    selector = LengthBasedExampleSelector(
        examples=EXAMPLES,
        example_prompt=prompts,
        max_length=25,
    )
    return selector


def test_dynamic_prompt_valid(selector: LengthBasedExampleSelector) -> None:
    """Test dynamic prompt can be successfully constructed from examples."""
    short_question = "Short question?"
    output = selector.select_examples({"question": short_question})
    assert output == EXAMPLES


def test_dynamic_prompt_trims_one_example(selector: LengthBasedExampleSelector) -> None:
    """Test dynamic prompt can trim one example."""
    long_question = """I am writing a really long question,
    this probably is going to affect the example right?"""
    output = selector.select_examples({"question": long_question})
    assert output == EXAMPLES[:1]


def test_dynamic_prompt_trims_all_examples(
    selector: LengthBasedExampleSelector,
) -> None:
    """Test dynamic prompt can trim all examples."""
    longest_question = """This question is super super super,
    super super super super super super super super super super super,
    super super super super long, this will affect the example right?"""
    output = selector.select_examples({"question": longest_question})
    assert output == []
