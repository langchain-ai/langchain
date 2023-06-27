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


def test_format_headers_api_key() -> None:
    """Test that the action headers is being created correctly."""
    tool = ZapierNLARunAction(
        action_id="test",
        zapier_description="test",
        params_schema={"test": "test"},
        api_wrapper=ZapierNLAWrapper(zapier_nla_api_key="test"),
    )
    headers = tool.api_wrapper._format_headers()
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept"] == "application/json"
    assert headers["X-API-Key"] == "test"


def test_format_headers_access_token() -> None:
    """Test that the action headers is being created correctly."""
    tool = ZapierNLARunAction(
        action_id="test",
        zapier_description="test",
        params_schema={"test": "test"},
        api_wrapper=ZapierNLAWrapper(zapier_nla_oauth_access_token="test"),
    )
    headers = tool.api_wrapper._format_headers()
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept"] == "application/json"
    assert headers["Authorization"] == "Bearer test"


def test_create_action_payload() -> None:
    """Test that the action payload is being created correctly."""
    tool = ZapierNLARunAction(
        action_id="test",
        zapier_description="test",
        params_schema={"test": "test"},
        api_wrapper=ZapierNLAWrapper(zapier_nla_api_key="test"),
    )

    payload = tool.api_wrapper._create_action_payload("some instructions")
    assert payload["instructions"] == "some instructions"
    assert payload.get("preview_only") is None


def test_create_action_payload_preview() -> None:
    """Test that the action payload with preview is being created correctly."""
    tool = ZapierNLARunAction(
        action_id="test",
        zapier_description="test",
        params_schema={"test": "test"},
        api_wrapper=ZapierNLAWrapper(zapier_nla_api_key="test"),
    )

    payload = tool.api_wrapper._create_action_payload(
        "some instructions",
        preview_only=True,
    )
    assert payload["instructions"] == "some instructions"
    assert payload["preview_only"] is True


def test_create_action_payload_with_params() -> None:
    """Test that the action payload with params is being created correctly."""
    tool = ZapierNLARunAction(
        action_id="test",
        zapier_description="test",
        params_schema={"test": "test"},
        api_wrapper=ZapierNLAWrapper(zapier_nla_api_key="test"),
    )

    payload = tool.api_wrapper._create_action_payload(
        "some instructions",
        {"test": "test"},
        preview_only=True,
    )
    assert payload["instructions"] == "some instructions"
    assert payload["preview_only"] is True
    assert payload["test"] == "test"


@pytest.mark.asyncio
async def test_apreview(mocker) -> None:  # type: ignore[no-untyped-def]
    """Test that the action payload with params is being created correctly."""
    tool = ZapierNLARunAction(
        action_id="test",
        zapier_description="test",
        params_schema={"test": "test"},
        api_wrapper=ZapierNLAWrapper(
            zapier_nla_api_key="test",
            zapier_nla_api_base="http://localhost:8080/v1/",
        ),
    )
    mockObj = mocker.patch.object(ZapierNLAWrapper, "_arequest")
    await tool.api_wrapper.apreview(
        "random_action_id",
        "some instructions",
        {"test": "test"},
    )
    mockObj.assert_called_once_with(
        "POST",
        "http://localhost:8080/v1/exposed/random_action_id/execute/",
        json={
            "instructions": "some instructions",
            "preview_only": True,
            "test": "test",
        },
    )


@pytest.mark.asyncio
async def test_arun(mocker) -> None:  # type: ignore[no-untyped-def]
    """Test that the action payload with params is being created correctly."""
    tool = ZapierNLARunAction(
        action_id="test",
        zapier_description="test",
        params_schema={"test": "test"},
        api_wrapper=ZapierNLAWrapper(
            zapier_nla_api_key="test",
            zapier_nla_api_base="http://localhost:8080/v1/",
        ),
    )
    mockObj = mocker.patch.object(ZapierNLAWrapper, "_arequest")
    await tool.api_wrapper.arun(
        "random_action_id",
        "some instructions",
        {"test": "test"},
    )
    mockObj.assert_called_once_with(
        "POST",
        "http://localhost:8080/v1/exposed/random_action_id/execute/",
        json={"instructions": "some instructions", "test": "test"},
    )


@pytest.mark.asyncio
async def test_alist(mocker) -> None:  # type: ignore[no-untyped-def]
    """Test that the action payload with params is being created correctly."""
    tool = ZapierNLARunAction(
        action_id="test",
        zapier_description="test",
        params_schema={"test": "test"},
        api_wrapper=ZapierNLAWrapper(
            zapier_nla_api_key="test",
            zapier_nla_api_base="http://localhost:8080/v1/",
        ),
    )
    mockObj = mocker.patch.object(ZapierNLAWrapper, "_arequest")
    await tool.api_wrapper.alist()
    mockObj.assert_called_once_with(
        "GET",
        "http://localhost:8080/v1/exposed/",
    )
