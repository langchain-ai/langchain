"""Test ChatSnowflakeCortex."""

import sys
from typing import Any, Generator
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from langchain_community.chat_models.snowflake import (
    ChatSnowflakeCortex,
    _convert_message_to_dict,
)


@pytest.fixture(autouse=True)
def mock_snowflake_import(monkeypatch: Any) -> MagicMock:
    mock_snowflake = MagicMock()

    monkeypatch.setitem(sys.modules, "snowflake", mock_snowflake)
    monkeypatch.setitem(sys.modules, "snowflake.snowpark", mock_snowflake.snowpark)

    return mock_snowflake


@pytest.fixture
def set_env_vars(request: Any) -> Generator[None, None, None]:
    env_vars = (
        request.param.get("temp_env_vars", {}) if hasattr(request, "param") else {}
    )

    with patch.dict("os.environ", env_vars, clear=True):
        yield


def test_messages_to_prompt_dict_with_valid_messages() -> None:
    messages = [
        SystemMessage(content="System Prompt"),
        HumanMessage(content="User message #1"),
        AIMessage(content="AI message #1"),
        HumanMessage(content="User message #2"),
        AIMessage(content="AI message #2"),
    ]
    result = [_convert_message_to_dict(m) for m in messages]
    expected = [
        {"role": "system", "content": "System Prompt"},
        {"role": "user", "content": "User message #1"},
        {"role": "assistant", "content": "AI message #1"},
        {"role": "user", "content": "User message #2"},
        {"role": "assistant", "content": "AI message #2"},
    ]
    assert result == expected


def test_chat_snowflake_cortex_model(
    mock_snowflake_import: MagicMock,
) -> None:
    """Test ChatSnowflakeCortex handles model_name."""
    mock_session = Mock()
    mock_snowflake_import.return_value = mock_session
    chat = ChatSnowflakeCortex(model="example_model", sp_session=mock_session)
    assert chat.model == "example_model"


@pytest.mark.parametrize(
    "set_env_vars, expected_error_message",
    [
        ({"temp_env_vars": {}}, "Did not find snowflake_username"),
        (
            {"temp_env_vars": {"SNOWFLAKE_USERNAME": "user"}},
            "Did not find snowflake_account",
        ),
        (
            {
                "temp_env_vars": {
                    "SNOWFLAKE_USERNAME": "user",
                    "SNOWFLAKE_ACCOUNT": "account",
                }
            },
            "Either `snowflake_password` or `snowflake_key_file` should be provided",
        ),
        (
            {
                "temp_env_vars": {
                    "SNOWFLAKE_USERNAME": "user",
                    "SNOWFLAKE_ACCOUNT": "account",
                    "SNOWFLAKE_PASSWORD": "password",
                    "SNOWFLAKE_KEY_FILE": "key_file",
                }
            },
            "Either `snowflake_password` or `snowflake_key_file` should be provided",
        ),
        (
            {
                "temp_env_vars": {
                    "SNOWFLAKE_USERNAME": "user",
                    "SNOWFLAKE_ACCOUNT": "account",
                    "SNOWFLAKE_KEY_FILE": "key_file",
                }
            },
            None,
        ),
        (
            {
                "temp_env_vars": {
                    "SNOWFLAKE_USERNAME": "user",
                    "SNOWFLAKE_ACCOUNT": "account",
                    "SNOWFLAKE_PASSWORD": "password",
                }
            },
            None,
        ),
    ],
    indirect=["set_env_vars"],
)
def test_validate_environment(
    set_env_vars: Any,
    expected_error_message: str,
) -> None:
    if expected_error_message:
        with pytest.raises(ValidationError, match=expected_error_message):
            chat = ChatSnowflakeCortex(model="example_model")
    else:
        with patch(
            "snowflake.snowpark.Session.SessionBuilder.create", return_value=MagicMock()
        ):
            chat = ChatSnowflakeCortex(model="example_model")
            assert chat.model == "example_model"
            assert chat.sp_session is not None


def test_session_handled_outside_not_closed() -> None:
    sp_session_mock = MagicMock()
    with patch.object(
        ChatSnowflakeCortex, "__del__", wraps=ChatSnowflakeCortex.__del__
    ) as mock_del:
        obj = ChatSnowflakeCortex(sp_session=sp_session_mock)

        del obj
        mock_del.assert_called_once()
        sp_session_mock.close.assert_not_called()


@pytest.mark.parametrize(
    "set_env_vars",
    [
        {
            "temp_env_vars": {
                "SNOWFLAKE_USERNAME": "user",
                "SNOWFLAKE_ACCOUNT": "account",
                "SNOWFLAKE_PASSWORD": "password",
            }
        }
    ],
    indirect=["set_env_vars"],
)
def test_del_called_without_session_provided(set_env_vars: Any) -> None:
    obj = ChatSnowflakeCortex()
    with patch.object(
        ChatSnowflakeCortex, "__del__", wraps=ChatSnowflakeCortex.__del__
    ) as mock_del:
        del obj
        mock_del.assert_called_once()
