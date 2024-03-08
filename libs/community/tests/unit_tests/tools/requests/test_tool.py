import asyncio
import json
from typing import Any, Dict

import pytest

from langchain_community.tools.requests.tool import (
    RequestsDeleteTool,
    RequestsGetTool,
    RequestsPatchTool,
    RequestsPostTool,
    RequestsPutTool,
    _parse_input,
)
from langchain_community.utilities.requests import (
    JsonRequestsWrapper,
    TextRequestsWrapper,
)


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
    tool = RequestsGetTool(
        requests_wrapper=mock_requests_wrapper, allow_dangerous_requests=True
    )
    assert tool.run("https://example.com") == "get_response"
    assert asyncio.run(tool.arun("https://example.com")) == "aget_response"


def test_requests_post_tool(mock_requests_wrapper: TextRequestsWrapper) -> None:
    tool = RequestsPostTool(
        requests_wrapper=mock_requests_wrapper, allow_dangerous_requests=True
    )
    input_text = '{"url": "https://example.com", "data": {"key": "value"}}'
    assert tool.run(input_text) == "post {'key': 'value'}"
    assert asyncio.run(tool.arun(input_text)) == "apost {'key': 'value'}"


def test_requests_patch_tool(mock_requests_wrapper: TextRequestsWrapper) -> None:
    tool = RequestsPatchTool(
        requests_wrapper=mock_requests_wrapper, allow_dangerous_requests=True
    )
    input_text = '{"url": "https://example.com", "data": {"key": "value"}}'
    assert tool.run(input_text) == "patch {'key': 'value'}"
    assert asyncio.run(tool.arun(input_text)) == "apatch {'key': 'value'}"


def test_requests_put_tool(mock_requests_wrapper: TextRequestsWrapper) -> None:
    tool = RequestsPutTool(
        requests_wrapper=mock_requests_wrapper, allow_dangerous_requests=True
    )
    input_text = '{"url": "https://example.com", "data": {"key": "value"}}'
    assert tool.run(input_text) == "put {'key': 'value'}"
    assert asyncio.run(tool.arun(input_text)) == "aput {'key': 'value'}"


def test_requests_delete_tool(mock_requests_wrapper: TextRequestsWrapper) -> None:
    tool = RequestsDeleteTool(
        requests_wrapper=mock_requests_wrapper, allow_dangerous_requests=True
    )
    assert tool.run("https://example.com") == "delete_response"
    assert asyncio.run(tool.arun("https://example.com")) == "adelete_response"


class _MockJsonRequestsWrapper(JsonRequestsWrapper):
    @staticmethod
    def get(url: str, **kwargs: Any) -> Dict[str, Any]:
        return {"response": "get_response"}

    @staticmethod
    async def aget(url: str, **kwargs: Any) -> Dict[str, Any]:
        return {"response": "aget_response"}

    @staticmethod
    def post(url: str, data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return {"response": f"post {json.dumps(data)}"}

    @staticmethod
    async def apost(url: str, data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return {"response": f"apost {json.dumps(data)}"}

    @staticmethod
    def patch(url: str, data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return {"response": f"patch {json.dumps(data)}"}

    @staticmethod
    async def apatch(url: str, data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return {"response": f"apatch {json.dumps(data)}"}

    @staticmethod
    def put(url: str, data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return {"response": f"put {json.dumps(data)}"}

    @staticmethod
    async def aput(url: str, data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return {"response": f"aput {json.dumps(data)}"}

    @staticmethod
    def delete(url: str, **kwargs: Any) -> Dict[str, Any]:
        return {"response": "delete_response"}

    @staticmethod
    async def adelete(url: str, **kwargs: Any) -> Dict[str, Any]:
        return {"response": "adelete_response"}


@pytest.fixture
def mock_json_requests_wrapper() -> JsonRequestsWrapper:
    return _MockJsonRequestsWrapper()


def test_requests_get_tool_json(
    mock_json_requests_wrapper: JsonRequestsWrapper,
) -> None:
    tool = RequestsGetTool(
        requests_wrapper=mock_json_requests_wrapper, allow_dangerous_requests=True
    )
    assert tool.run("https://example.com") == {"response": "get_response"}
    assert asyncio.run(tool.arun("https://example.com")) == {
        "response": "aget_response"
    }


def test_requests_post_tool_json(
    mock_json_requests_wrapper: JsonRequestsWrapper,
) -> None:
    tool = RequestsPostTool(
        requests_wrapper=mock_json_requests_wrapper, allow_dangerous_requests=True
    )
    input_text = '{"url": "https://example.com", "data": {"key": "value"}}'
    assert tool.run(input_text) == {"response": 'post {"key": "value"}'}
    assert asyncio.run(tool.arun(input_text)) == {"response": 'apost {"key": "value"}'}


def test_requests_patch_tool_json(
    mock_json_requests_wrapper: JsonRequestsWrapper,
) -> None:
    tool = RequestsPatchTool(
        requests_wrapper=mock_json_requests_wrapper, allow_dangerous_requests=True
    )
    input_text = '{"url": "https://example.com", "data": {"key": "value"}}'
    assert tool.run(input_text) == {"response": 'patch {"key": "value"}'}
    assert asyncio.run(tool.arun(input_text)) == {"response": 'apatch {"key": "value"}'}


def test_requests_put_tool_json(
    mock_json_requests_wrapper: JsonRequestsWrapper,
) -> None:
    tool = RequestsPutTool(
        requests_wrapper=mock_json_requests_wrapper, allow_dangerous_requests=True
    )
    input_text = '{"url": "https://example.com", "data": {"key": "value"}}'
    assert tool.run(input_text) == {"response": 'put {"key": "value"}'}
    assert asyncio.run(tool.arun(input_text)) == {"response": 'aput {"key": "value"}'}


def test_requests_delete_tool_json(
    mock_json_requests_wrapper: JsonRequestsWrapper,
) -> None:
    tool = RequestsDeleteTool(
        requests_wrapper=mock_json_requests_wrapper, allow_dangerous_requests=True
    )
    assert tool.run("https://example.com") == {"response": "delete_response"}
    assert asyncio.run(tool.arun("https://example.com")) == {
        "response": "adelete_response"
    }
