"""Test building the Zapier tool, not running it."""
from unittest.mock import MagicMock, patch

import pytest
import requests

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


def test_wrapper_fails_no_api_key_or_access_token_initialization() -> None:
    """Test Wrapper requires either an API Key or OAuth Access Token."""
    with pytest.raises(ValueError):
        ZapierNLAWrapper()


def test_wrapper_api_key_initialization() -> None:
    """Test Wrapper initializes with an API Key."""
    ZapierNLAWrapper(zapier_nla_api_key="test")


def test_wrapper_access_token_initialization() -> None:
    """Test Wrapper initializes with an API Key."""
    ZapierNLAWrapper(zapier_nla_oauth_access_token="test")


def test_list_raises_401_invalid_api_key() -> None:
    """Test that a valid error is raised when the API Key is invalid."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = requests.HTTPError(
        "401 Client Error: Unauthorized for url: https://nla.zapier.com/api/v1/exposed/"
    )
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response

    with patch("requests.Session", return_value=mock_session):
        wrapper = ZapierNLAWrapper(zapier_nla_api_key="test")

        with pytest.raises(requests.HTTPError) as err:
            wrapper.list()

        assert str(err.value).startswith(
            "An unauthorized response occurred. Check that your api key is correct. "
            "Err:"
        )


def test_list_raises_401_invalid_access_token() -> None:
    """Test that a valid error is raised when the API Key is invalid."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = requests.HTTPError(
        "401 Client Error: Unauthorized for url: https://nla.zapier.com/api/v1/exposed/"
    )
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response

    with patch("requests.Session", return_value=mock_session):
        wrapper = ZapierNLAWrapper(zapier_nla_oauth_access_token="test")

        with pytest.raises(requests.HTTPError) as err:
            wrapper.list()

        assert str(err.value).startswith(
            "An unauthorized response occurred. Check that your access token is "
            "correct and doesn't need to be refreshed. Err:"
        )


def test_list_raises_other_error() -> None:
    """Test that a valid error is raised when an unknown HTTP Error occurs."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.HTTPError(
        "404 Client Error: Not found for url"
    )
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response

    with patch("requests.Session", return_value=mock_session):
        wrapper = ZapierNLAWrapper(zapier_nla_oauth_access_token="test")

        with pytest.raises(requests.HTTPError) as err:
            wrapper.list()

        assert str(err.value) == "404 Client Error: Not found for url"
