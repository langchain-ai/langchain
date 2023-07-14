# flake8: noqa E501

"""Test LLMCheckerChain functionality."""

import pytest

from langchain.chains.llm_checker.base import LLMCheckerChain
from langchain.chains.llm_checker.prompt import (
    _CHECK_ASSERTIONS_TEMPLATE,
    _CREATE_DRAFT_ANSWER_TEMPLATE,
    _LIST_ASSERTIONS_TEMPLATE,
    _REVISED_ANSWER_TEMPLATE,
)
from tests.unit_tests.llms.fake_llm import FakeLLM


@pytest.fixture
def fake_llm_checker_chain() -> LLMCheckerChain:
    """Fake LLMCheckerChain for testing."""
    queries = {
        _CREATE_DRAFT_ANSWER_TEMPLATE.format(
            question="Which mammal lays the biggest eggs?"
        ): "I don't know which mammal layers the biggest eggs.",
        _LIST_ASSERTIONS_TEMPLATE.format(
            statement="I don't know which mammal layers the biggest eggs.",
        ): "1) I know that mammals lay eggs.\n2) I know that birds lay eggs.\n3) I know that birds are mammals.",
        _CHECK_ASSERTIONS_TEMPLATE.format(
            assertions="1) I know that mammals lay eggs.\n2) I know that birds lay eggs.\n3) I know that birds are mammals.",
        ): "1) I know that mammals lay eggs. TRUE\n2) I know that birds lay eggs. TRUE\n3) I know that birds are mammals. TRUE",
        _REVISED_ANSWER_TEMPLATE.format(
            checked_assertions="1) I know that mammals lay eggs. TRUE\n2) I know that birds lay eggs. TRUE\n3) I know that birds are mammals. TRUE",
            question="Which mammal lays the biggest eggs?",
        ): "I still don't know.",
    }
    fake_llm = FakeLLM(queries=queries)
    return LLMCheckerChain.from_llm(fake_llm, input_key="q", output_key="a")


def test_simple_question(fake_llm_checker_chain: LLMCheckerChain) -> None:
    """Test simple question that should not need python."""
    question = "Which mammal lays the biggest eggs?"
    output = fake_llm_checker_chain.run(question)
    assert output == "I still don't know."
