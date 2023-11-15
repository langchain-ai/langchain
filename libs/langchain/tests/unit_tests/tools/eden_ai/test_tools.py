from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from langchain.tools.edenai import EdenAiTextModerationTool

tool = EdenAiTextModerationTool(
    providers=["openai"], language="en", edenai_api_key="fake_key"
)


@pytest.fixture
def mock_post() -> Generator:
    with patch("langchain.tools.edenai.edenai_base_tool.requests.post") as mock:
        yield mock


def test_provider_not_available(mock_post: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "error": {
                "message": """Amazon has returned an error: 
                An error occurred (TextSizeLimitExceededException) 
                when calling the DetectTargetedSentiment 
                operation: Input text size exceeds limit. 
                Max length of request text allowed is 5000 bytes 
                while in this request the text size is 47380 bytes""",
                "type": "ProviderInvalidInputTextLengthError",
            },
            "status": "fail",
            "provider": "amazon",
            "provider_status_code": 400,
            "cost": 0.0,
        }
    ]
    mock_post.return_value = mock_response

    with pytest.raises(ValueError):
        tool._run("some query")


def test_unexpected_response(mock_post: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "status": "success",
        }
    ]
    mock_post.return_value = mock_response
    with pytest.raises(RuntimeError):
        tool._run("some query")


def test_incomplete_response(mock_post: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "status": "success",
            "provider": "microsoft",
            "nsfw_likelihood": 5,
            "cost": 0.001,
            "label": ["sexually explicit", "sexually suggestive", "offensive"],
        }
    ]

    mock_post.return_value = mock_response
    with pytest.raises(RuntimeError):
        tool._run("some query")


def test_invalid_payload(mock_post: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {}
    mock_post.return_value = mock_response

    with pytest.raises(ValueError):
        tool._run("some query")


def test_parse_response_format(mock_post: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "status": "success",
            "provider": "microsoft",
            "nsfw_likelihood": 5,
            "cost": 0.001,
            "label": ["offensive", "hate_speech"],
            "likelihood": [4, 5],
        }
    ]
    mock_post.return_value = mock_response

    result = tool("some query")

    assert result == 'nsfw_likelihood: 5\n"offensive": 4\n"hate_speech": 5'
