"""Test building the Zapier tool, not running it."""
import pytest

from langchain.tools.zapier.prompt import BASE_ZAPIER_TOOL_PROMPT
from langchain.tools.zapier.tool import ZapierNLARunAction
from langchain.utilities.zapier import ZapierNLAWrapper


def test_default_base_prompt() -> None:
    """Test that the default prompt is being inserted."""
    tool = ZapierNLARunAction(
        action_id="test",
        zapier_description="test",
        params_schema={"test": "test"},
        api_wrapper=ZapierNLAWrapper(zapier_nla_api_key="test"),
    )

    # Test that the base prompt was successfully assigned to the default prompt
    assert tool.base_prompt == BASE_ZAPIER_TOOL_PROMPT
    assert tool.description == BASE_ZAPIER_TOOL_PROMPT.format(
        zapier_description="test",
        params=str(list({"test": "test"}.keys())),
    )


def test_custom_base_prompt() -> None:
    """Test that a custom prompt is being inserted."""
    base_prompt = "Test. {zapier_description} and {params}."
    tool = ZapierNLARunAction(
        action_id="test",
        zapier_description="test",
        params_schema={"test": "test"},
        base_prompt=base_prompt,
        api_wrapper=ZapierNLAWrapper(zapier_nla_api_key="test"),
    )

    # Test that the base prompt was successfully assigned to the default prompt
    assert tool.base_prompt == base_prompt
    assert tool.description == "Test. test and ['test']."


def test_custom_base_prompt_fail() -> None:
    """Test validating an invalid custom prompt."""
    base_prompt = "Test. {zapier_description}."
    with pytest.raises(ValueError):
        ZapierNLARunAction(
            action_id="test",
            zapier_description="test",
            params={"test": "test"},
            base_prompt=base_prompt,
            api_wrapper=ZapierNLAWrapper(zapier_nla_api_key="test"),
        )
