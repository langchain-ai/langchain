import json
import re
import time
from unittest import mock
from urllib.parse import parse_qs, urlparse

from azure.core.credentials import AccessToken

from langchain_azure_dynamic_sessions import SessionsPythonREPLTool
from langchain_azure_dynamic_sessions.tools.sessions import (
    _access_token_provider_factory,
)

POOL_MANAGEMENT_ENDPOINT = "https://westus2.dynamicsessions.io/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/sessions-rg/sessionPools/my-pool"


def test_default_access_token_provider_returns_token() -> None:
    access_token_provider = _access_token_provider_factory()
    with mock.patch(
        "azure.identity.DefaultAzureCredential.get_token"
    ) as mock_get_token:
        mock_get_token.return_value = AccessToken("token_value", 0)
        access_token = access_token_provider()
        assert access_token == "token_value"


def test_default_access_token_provider_returns_cached_token() -> None:
    access_token_provider = _access_token_provider_factory()
    with mock.patch(
        "azure.identity.DefaultAzureCredential.get_token"
    ) as mock_get_token:
        mock_get_token.return_value = AccessToken(
            "token_value", int(time.time() + 1000)
        )
        access_token = access_token_provider()
        assert access_token == "token_value"
        assert mock_get_token.call_count == 1

        mock_get_token.return_value = AccessToken(
            "new_token_value", int(time.time() + 1000)
        )
        access_token = access_token_provider()
        assert access_token == "token_value"
        assert mock_get_token.call_count == 1


def test_default_access_token_provider_refreshes_expiring_token() -> None:
    access_token_provider = _access_token_provider_factory()
    with mock.patch(
        "azure.identity.DefaultAzureCredential.get_token"
    ) as mock_get_token:
        mock_get_token.return_value = AccessToken("token_value", int(time.time() - 1))
        access_token = access_token_provider()
        assert access_token == "token_value"
        assert mock_get_token.call_count == 1

        mock_get_token.return_value = AccessToken(
            "new_token_value", int(time.time() + 1000)
        )
        access_token = access_token_provider()
        assert access_token == "new_token_value"
        assert mock_get_token.call_count == 2


@mock.patch("requests.post")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_code_execution_calls_api(
    mock_get_token: mock.MagicMock, mock_post: mock.MagicMock
) -> None:
    tool = SessionsPythonREPLTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
    mock_post.return_value.json.return_value = {
        "$id": "1",
        "properties": {
            "$id": "2",
            "status": "Success",
            "stdout": "hello world\n",
            "stderr": "",
            "result": "",
            "executionTimeInMilliseconds": 33,
        },
    }
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))

    result = tool.run("print('hello world')")

    assert json.loads(result) == {
        "result": "",
        "stdout": "hello world\n",
        "stderr": "",
    }

    api_url = f"{POOL_MANAGEMENT_ENDPOINT}/code/execute"
    headers = {
        "Authorization": "Bearer token_value",
        "Content-Type": "application/json",
        "User-Agent": mock.ANY,
    }
    body = {
        "properties": {
            "codeInputType": "inline",
            "executionType": "synchronous",
            "code": "print('hello world')",
        }
    }
    mock_post.assert_called_once_with(mock.ANY, headers=headers, json=body)

    called_headers = mock_post.call_args.kwargs["headers"]
    assert re.match(
        r"^langchain-azure-dynamic-sessions/\d+\.\d+\.\d+.* \(Language=Python\)",
        called_headers["User-Agent"],
    )

    called_api_url = mock_post.call_args.args[0]
    assert called_api_url.startswith(api_url)


@mock.patch("requests.post")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_uses_specified_session_id(
    mock_get_token: mock.MagicMock, mock_post: mock.MagicMock
) -> None:
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        session_id="00000000-0000-0000-0000-000000000003",
    )
    mock_post.return_value.json.return_value = {
        "$id": "1",
        "properties": {
            "$id": "2",
            "status": "Success",
            "stdout": "",
            "stderr": "",
            "result": "2",
            "executionTimeInMilliseconds": 33,
        },
    }
    mock_get_token.return_value = AccessToken("token_value", int(time.time() + 1000))
    tool.run("1 + 1")
    call_url = mock_post.call_args.args[0]
    parsed_url = urlparse(call_url)
    call_identifier = parse_qs(parsed_url.query)["identifier"][0]
    assert call_identifier == "00000000-0000-0000-0000-000000000003"


def test_sanitizes_input() -> None:
    tool = SessionsPythonREPLTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
    with mock.patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "$id": "1",
            "properties": {
                "$id": "2",
                "status": "Success",
                "stdout": "",
                "stderr": "",
                "result": "",
                "executionTimeInMilliseconds": 33,
            },
        }
        tool.run("```python\nprint('hello world')\n```")
        body = mock_post.call_args.kwargs["json"]
        assert body["properties"]["code"] == "print('hello world')"


def test_does_not_sanitize_input() -> None:
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT, sanitize_input=False
    )
    with mock.patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "$id": "1",
            "properties": {
                "$id": "2",
                "status": "Success",
                "stdout": "",
                "stderr": "",
                "result": "",
                "executionTimeInMilliseconds": 33,
            },
        }
        tool.run("```python\nprint('hello world')\n```")
        body = mock_post.call_args.kwargs["json"]
        assert body["properties"]["code"] == "```python\nprint('hello world')\n```"


def test_uses_custom_access_token_provider() -> None:
    def custom_access_token_provider() -> str:
        return "custom_token"

    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        access_token_provider=custom_access_token_provider,
    )

    with mock.patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "$id": "1",
            "properties": {
                "$id": "2",
                "status": "Success",
                "stdout": "",
                "stderr": "",
                "result": "",
                "executionTimeInMilliseconds": 33,
            },
        }
        tool.run("print('hello world')")
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer custom_token"
