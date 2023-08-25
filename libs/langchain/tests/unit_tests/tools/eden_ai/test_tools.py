from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from langchain.tools.edenai import EdenAiTextModerationTool

tool = EdenAiTextModerationTool(providers=["openai"], language="en")


@pytest.fixture
def mock_post() -> Generator:
    with patch("langchain.tools.edenai.edenai_base_tool.requests.post") as mock:
        yield mock


def test_provider_not_available(mock_post: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"status": "fail", "openai": {"error": {"message": "Provider not available"}}}
    ]
    mock_post.return_value = mock_response

    with pytest.raises(ValueError):
        tool._run("some query")


def test_unexpected_response(mock_post: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"status": "success", "openai": {"data": "unexpected data"}}
    ]
    mock_post.return_value = mock_response
    with pytest.raises(RuntimeError):
        tool._run("some query")


def test_parsing_response(mock_post: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "status": "success",
            "openai": {
                "nsfw_likelihood": 3,
                "label": "az",
                "likelihood": [4, 5],
            },
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
