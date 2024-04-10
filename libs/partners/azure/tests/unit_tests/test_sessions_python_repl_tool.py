import time
from unittest import mock
import uuid
from azure.core.credentials import AccessToken
from langchain_azure import SessionsPythonREPLTool
from langchain_azure.tools.sessions import _access_token_provider_factory


POOL_MANAGEMENT_ENDPOINT = "https://westus2.acasessions.io/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/sessions-rg/sessionPools/my-pool/"


def test_default_access_token_provider_returns_token():
    access_token_provider = _access_token_provider_factory()
    with mock.patch("azure.identity.DefaultAzureCredential.get_token") as mock_get_token:
        mock_get_token.return_value = AccessToken("token_value", 0)
        access_token = access_token_provider()
        assert access_token == "token_value"


def test_default_access_token_provider_returns_cached_token():
    access_token_provider = _access_token_provider_factory()
    with mock.patch("azure.identity.DefaultAzureCredential.get_token") as mock_get_token:
        mock_get_token.return_value = AccessToken("token_value", time.time() + 1000)
        access_token = access_token_provider()
        assert access_token == "token_value"
        assert mock_get_token.call_count == 1

        mock_get_token.return_value = AccessToken("new_token_value", time.time() + 1000)
        access_token = access_token_provider()
        assert access_token == "token_value"
        assert mock_get_token.call_count == 1


def test_default_access_token_provider_refreshes_expiring_token():
    access_token_provider = _access_token_provider_factory()
    with mock.patch("azure.identity.DefaultAzureCredential.get_token") as mock_get_token:
        mock_get_token.return_value = AccessToken("token_value", time.time() - 1)
        access_token = access_token_provider()
        assert access_token == "token_value"
        assert mock_get_token.call_count == 1

        mock_get_token.return_value = AccessToken("new_token_value", time.time() + 1000)
        access_token = access_token_provider()
        assert access_token == "new_token_value"
        assert mock_get_token.call_count == 2


@mock.patch("requests.post")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_code_execution_calls_api(mock_get_token, mock_post: mock.MagicMock):
    tool = SessionsPythonREPLTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
    mock_post.return_value.json.return_value = {"result": "", "stdout": "hello world\n", "stderr": ""}
    mock_get_token.return_value = AccessToken("token_value", time.time() + 1000)
    
    result = tool.run("print('hello world')")
    
    assert result == "result:\n\n\nstdout:\nhello world\n\n\nstderr:\n"
    
    api_url = f"{POOL_MANAGEMENT_ENDPOINT}python/execute"
    headers = {
        "Authorization": f"Bearer token_value",
        "Content-Type": "application/json",
    }
    body = {
        "properties": {
            "identifier": mock.ANY,
            "codeInputType": "inline",
            "executionType": "synchronous",
            "pythonCode": "print('hello world')",
        }
    }
    mock_post.assert_called_once_with(api_url, headers=headers, json=body)


@mock.patch("requests.post")
@mock.patch("azure.identity.DefaultAzureCredential.get_token")
def test_uses_specified_session_id(mock_get_token, mock_post: mock.MagicMock):
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        session_id="00000000-0000-0000-0000-000000000003",
    )
    mock_post.return_value.json.return_value = {"result": "2", "stdout": "", "stderr": ""}
    mock_get_token.return_value = AccessToken("token_value", time.time() + 1000)
    tool.run("1 + 1")
    body = mock_post.call_args.kwargs["json"]
    assert body["properties"]["identifier"] == "00000000-0000-0000-0000-000000000003"


def test_sanitizes_input():
    tool = SessionsPythonREPLTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)
    with mock.patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"result": "", "stdout": "", "stderr": ""}
        tool.run("```python\nprint('hello world')\n```")
        body = mock_post.call_args.kwargs["json"]
        assert body["properties"]["pythonCode"] == "print('hello world')"


def test_does_not_sanitize_input():
    tool = SessionsPythonREPLTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT, sanitize_input=False)
    with mock.patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"result": "", "stdout": "", "stderr": ""}
        tool.run("```python\nprint('hello world')\n```")
        body = mock_post.call_args.kwargs["json"]
        assert body["properties"]["pythonCode"] == "```python\nprint('hello world')\n```"


def test_uses_custom_access_token_provider():
    def custom_access_token_provider():
        return "custom_token"
    
    tool = SessionsPythonREPLTool(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT,
        access_token_provider=custom_access_token_provider,
    )

    with mock.patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"result": "", "stdout": "", "stderr": ""}
        tool.run("print('hello world')")
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer custom_token"