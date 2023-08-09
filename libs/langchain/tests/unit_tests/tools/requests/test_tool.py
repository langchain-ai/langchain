import asyncio
from typing import Any, Dict

import pytest

from langchain.tools.requests.tool import (
    RequestsDeleteTool,
    RequestsGetTool,
    RequestsPatchTool,
    RequestsPostTool,
    RequestsPutTool,
    _parse_input,
)
from langchain.utilities.requests import TextRequestsWrapper


class _MockTextRequestsWrapper(TextRequestsWrapper):
    @staticmethod
    def get(url: str, **kwargs: Any) -> str:
        return "get_response"

    @staticmethod
    async def aget(url: str, **kwargs: Any) -> str:
        return "aget_response"

    @staticmethod
    def post(url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        return f"post {str(data)}"

    @staticmethod
    async def apost(url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        return f"apost {str(data)}"

    @staticmethod
    def patch(url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        return f"patch {str(data)}"

    @staticmethod
    async def apatch(url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        return f"apatch {str(data)}"

    @staticmethod
    def put(url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        return f"put {str(data)}"

    @staticmethod
    async def aput(url: str, data: Dict[str, Any], **kwargs: Any) -> str:
        return f"aput {str(data)}"

    @staticmethod
    def delete(url: str, **kwargs: Any) -> str:
        return "delete_response"

    @staticmethod
    async def adelete(url: str, **kwargs: Any) -> str:
        return "adelete_response"


@pytest.fixture
def mock_requests_wrapper() -> TextRequestsWrapper:
    return _MockTextRequestsWrapper()


def test_parse_input() -> None:
    input_text = '{"url": "https://example.com", "data": {"key": "value"}}'
    expected_output = {"url": "https://example.com", "data": {"key": "value"}}
    assert _parse_input(input_text) == expected_output


def test_requests_get_tool(mock_requests_wrapper: TextRequestsWrapper) -> None:
    tool = RequestsGetTool(requests_wrapper=mock_requests_wrapper)
    assert tool.run("https://example.com") == "get_response"
    assert asyncio.run(tool.arun("https://example.com")) == "aget_response"


def test_requests_post_tool(mock_requests_wrapper: TextRequestsWrapper) -> None:
    tool = RequestsPostTool(requests_wrapper=mock_requests_wrapper)
    input_text = '{"url": "https://example.com", "data": {"key": "value"}}'
    assert tool.run(input_text) == "post {'key': 'value'}"
    assert asyncio.run(tool.arun(input_text)) == "apost {'key': 'value'}"


def test_requests_patch_tool(mock_requests_wrapper: TextRequestsWrapper) -> None:
    tool = RequestsPatchTool(requests_wrapper=mock_requests_wrapper)
    input_text = '{"url": "https://example.com", "data": {"key": "value"}}'
    assert tool.run(input_text) == "patch {'key': 'value'}"
    assert asyncio.run(tool.arun(input_text)) == "apatch {'key': 'value'}"


def test_requests_put_tool(mock_requests_wrapper: TextRequestsWrapper) -> None:
    tool = RequestsPutTool(requests_wrapper=mock_requests_wrapper)
    input_text = '{"url": "https://example.com", "data": {"key": "value"}}'
    assert tool.run(input_text) == "put {'key': 'value'}"
    assert asyncio.run(tool.arun(input_text)) == "aput {'key': 'value'}"


def test_requests_delete_tool(mock_requests_wrapper: TextRequestsWrapper) -> None:
    tool = RequestsDeleteTool(requests_wrapper=mock_requests_wrapper)
    assert tool.run("https://example.com") == "delete_response"
    assert asyncio.run(tool.arun("https://example.com")) == "adelete_response"
