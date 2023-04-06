# flake8: noqa E501

"""Test LLMReturnFormatCheckerChain functionality."""

import pytest

from langchain.chains.llm_return_format_checker.base import LLMReturnFormatCheckerChain
from langchain.chains.llm_return_format_checker.prompt import (
    _CREATE_DRAFT_ACTION_TEMPLATE_,
    _CHECK_ACTION_VALIDITY_TEMPLATE_,
    _CHECK_ACTION_FORMAT_TEMPLATE_,
    _CHECK_FORMAT_VALIDITY_TEMPLATE_,
)
from tests.unit_tests.llms.fake_llm import FakeLLM


@pytest.fixture
def fake_llm_return_format_checker_chain() -> LLMReturnFormatCheckerChain:
    """Fake LLMCheckerChain for testing."""
    queries = {
        _CREATE_DRAFT_ACTION_TEMPLATE_.format(
        situation="Poker Time!",
        valid_actions = "FOLD!",
        call_to_action = "DO NOT LOSE YOUR MONEY!",
        ): "I FOLD!",
        _CHECK_ACTION_VALIDITY_TEMPLATE_.format(
        situation="Poker Time!",
        initial_action="I FOLD!",
        valid_actions="FOLD!",
        ): "I FOLD!",
        _CHECK_ACTION_FORMAT_TEMPLATE_.format(
        action_format="FOLD!",
        validated_action="I FOLD!",
        ): "I FOLD!",
        _CHECK_FORMAT_VALIDITY_TEMPLATE_.format(
            action_format="FOLD!",
            initial_format_validated_action="I FOLD!",
        ): "FOLD!.",
    }
    fake_llm = FakeLLM(queries=queries)
    return LLMReturnFormatCheckerChain(
        llm=fake_llm, 
    )


def test_simple_question(fake_llm_return_format_checker_chain: LLMReturnFormatCheckerChain) -> None:
    """Test simple question that should not need python."""
    situation = "Poker Time!"
    valid_actions = "FOLD!"
    call_to_action = "DO NOT LOSE YOUR MONEY!"
    action_format = "FOLD!"
    output = fake_llm_return_format_checker_chain.run({
        'situation': situation,
        'valid_actions': valid_actions,
        'call_to_action': call_to_action,
        'action_format': action_format
        })
    assert output == "FOLD!."