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
    def get(url: str, headers: Dict[str, str] = {}, **kwargs: Any) -> str:
        return f"get_response {headers}"

    @staticmethod
    async def aget(url: str, headers: Dict[str, str] = {}, **kwargs: Any) -> str:
        return f"aget_response {headers}"

    @staticmethod
    def post(
        url: str, data: Dict[str, Any], headers: Dict[str, str] = {}, **kwargs: Any
    ) -> str:
        return f"post {str(data)} {headers}"

    @staticmethod
    async def apost(
        url: str, data: Dict[str, Any], headers: Dict[str, str] = {}, **kwargs: Any
    ) -> str:
        return f"apost {str(data)} {headers}"

    @staticmethod
    def patch(
        url: str, data: Dict[str, Any], headers: Dict[str, str] = {}, **kwargs: Any
    ) -> str:
        return f"patch {str(data)} {headers}"

    @staticmethod
    async def apatch(
        url: str, data: Dict[str, Any], headers: Dict[str, str] = {}, **kwargs: Any
    ) -> str:
        return f"apatch {str(data)} {headers}"

    @staticmethod
    def put(
        url: str, data: Dict[str, Any], headers: Dict[str, str] = {}, **kwargs: Any
    ) -> str:
        return f"put {str(data)} {headers}"

    @staticmethod
    async def aput(
        url: str, data: Dict[str, Any], headers: Dict[str, str] = {}, **kwargs: Any
    ) -> str:
        return f"aput {str(data)} {headers}"

    @staticmethod
    def delete(url: str, headers: Dict[str, str] = {}, **kwargs: Any) -> str:
        return f"delete_response {headers}"

    @staticmethod
    async def adelete(url: str, headers: Dict[str, str] = {}, **kwargs: Any) -> str:
        return f"adelete_response {headers}"


@pytest.fixture
def mock_requests_wrapper() -> TextRequestsWrapper:
    return _MockTextRequestsWrapper()


def test_parse_input() -> None:
    input_text = (
        '{"url": "https://example.com", '
        '"data": {"key": "value"}, '
        '"headers": {"Key": "Value"}}'
    )
    expected_output = {
        "url": "https://example.com",
        "data": {"key": "value"},
        "headers": {"Key": "Value"},
    }
    assert _parse_input(input_text) == expected_output


def test_requests_get_tool(mock_requests_wrapper: TextRequestsWrapper) -> None:
    tool = RequestsGetTool(
        requests_wrapper=mock_requests_wrapper, allow_dangerous_requests=True
    )
    input_text = '{"url": "https://example.com"}'
    assert tool.run(input_text) == "get_response {}"
    assert asyncio.run(tool.arun(input_text)) == "aget_response {}"


def test_requests_get_tool_with_headers(
    mock_requests_wrapper: TextRequestsWrapper,
) -> None:
    tool = RequestsGetTool(
        requests_wrapper=mock_requests_wrapper, allow_dangerous_requests=True
    )
    headers = {"Key": "Value"}
    input_text = '{"url": "https://example.com", "headers": {"Key": "Value"}}'
    assert tool.run(input_text) == f"get_response {headers}"
    assert asyncio.run(tool.arun(input_text)) == f"aget_response {headers}"


def test_requests_get_tool_with_headers_merge(
    mock_requests_wrapper: TextRequestsWrapper,
) -> None:
    tool = RequestsGetTool(
        requests_wrapper=mock_requests_wrapper, allow_dangerous_requests=True
    )
    headers = {"Key": "Value"}
    input_text = '{"url": "https://example.com", "headers": {"Key": "Value"}}'
    assert tool.run(input_text) == f"get_response {headers}"
    assert asyncio.run(tool.arun(input_text)) == f"aget_response {headers}"


def test_requests_post_tool(mock_requests_wrapper: TextRequestsWrapper) -> None:
    tool = RequestsPostTool(
        requests_wrapper=mock_requests_wrapper, allow_dangerous_requests=True
    )
    input_text = '{"url": "https://example.com", "data": {"key": "value"}}'
    assert tool.run(input_text) == "post {'key': 'value'} {}"
    assert asyncio.run(tool.arun(input_text)) == "apost {'key': 'value'} {}"


def test_requests_post_tool_with_headers(
    mock_requests_wrapper: TextRequestsWrapper,
) -> None:
    tool = RequestsPostTool(
        requests_wrapper=mock_requests_wrapper, allow_dangerous_requests=True
    )
    headers = {"Key": "Value"}
    input_text = (
        '{"url": "https://example.com", '
        '"data": {"key": "value"}, '
        '"headers": {"Key": "Value"}}'
    )
    assert tool.run(input_text) == f"post {{'key': 'value'}} {headers}"
    assert asyncio.run(tool.arun(input_text)) == f"apost {{'key': 'value'}} {headers}"


def test_requests_patch_tool(mock_requests_wrapper: TextRequestsWrapper) -> None:
    tool = RequestsPatchTool(
        requests_wrapper=mock_requests_wrapper, allow_dangerous_requests=True
    )
    input_text = '{"url": "https://example.com", "data": {"key": "value"}}'
    assert tool.run(input_text) == "patch {'key': 'value'} {}"
    assert asyncio.run(tool.arun(input_text)) == "apatch {'key': 'value'} {}"


def test_requests_patch_tool_with_headers(
    mock_requests_wrapper: TextRequestsWrapper,
) -> None:
    tool = RequestsPatchTool(
        requests_wrapper=mock_requests_wrapper, allow_dangerous_requests=True
    )
    headers = {"Key": "Value"}
    input_text = (
        '{"url": "https://example.com", '
        '"data": {"key": "value"}, '
        '"headers": {"Key": "Value"}}'
    )
    assert tool.run(input_text) == f"patch {{'key': 'value'}} {headers}"
    assert asyncio.run(tool.arun(input_text)) == f"apatch {{'key': 'value'}} {headers}"


def test_requests_put_tool(mock_requests_wrapper: TextRequestsWrapper) -> None:
    tool = RequestsPutTool(
        requests_wrapper=mock_requests_wrapper, allow_dangerous_requests=True
    )
    input_text = '{"url": "https://example.com", "data": {"key": "value"}}'
    assert tool.run(input_text) == "put {'key': 'value'} {}"
    assert asyncio.run(tool.arun(input_text)) == "aput {'key': 'value'} {}"


def test_requests_put_tool_with_headers(
    mock_requests_wrapper: TextRequestsWrapper,
) -> None:
    tool = RequestsPutTool(
        requests_wrapper=mock_requests_wrapper, allow_dangerous_requests=True
    )
    headers = {"Key": "Value"}
    input_text = (
        '{"url": "https://example.com", '
        '"data": {"key": "value"}, '
        '"headers": {"Key": "Value"}}'
    )
    assert tool.run(input_text) == f"put {{'key': 'value'}} {headers}"
    assert asyncio.run(tool.arun(input_text)) == f"aput {{'key': 'value'}} {headers}"


def test_requests_delete_tool(mock_requests_wrapper: TextRequestsWrapper) -> None:
    tool = RequestsDeleteTool(
        requests_wrapper=mock_requests_wrapper, allow_dangerous_requests=True
    )
    input_text = '{"url": "https://example.com"}'
    assert tool.run(input_text) == "delete_response {}"
    assert asyncio.run(tool.arun(input_text)) == "adelete_response {}"


def test_requests_delete_tool_with_headers(
    mock_requests_wrapper: TextRequestsWrapper,
) -> None:
    tool = RequestsDeleteTool(
        requests_wrapper=mock_requests_wrapper, allow_dangerous_requests=True
    )
    headers = {"Key": "Value"}
    input_text = '{"url": "https://example.com", "headers": {"Key": "Value"}}'
    assert tool.run(input_text) == f"delete_response {headers}"
    assert asyncio.run(tool.arun(input_text)) == f"adelete_response {headers}"


class _MockJsonRequestsWrapper(JsonRequestsWrapper):
    @staticmethod
    def get(url: str, headers: Dict[str, str] = {}, **kwargs: Any) -> Dict[str, Any]:
        return {"response": f"get_response {headers}"}

    @staticmethod
    async def aget(
        url: str, headers: Dict[str, str] = {}, **kwargs: Any
    ) -> Dict[str, Any]:
        return {"response": f"aget_response {headers}"}

    @staticmethod
    def post(
        url: str, data: Dict[str, Any], headers: Dict[str, str] = {}, **kwargs: Any
    ) -> Dict[str, Any]:
        return {"response": f"post {json.dumps(data)} {headers}"}

    @staticmethod
    async def apost(
        url: str, data: Dict[str, Any], headers: Dict[str, str] = {}, **kwargs: Any
    ) -> Dict[str, Any]:
        return {"response": f"apost {json.dumps(data)} {headers}"}

    @staticmethod
    def patch(
        url: str, data: Dict[str, Any], headers: Dict[str, str] = {}, **kwargs: Any
    ) -> Dict[str, Any]:
        return {"response": f"patch {json.dumps(data)} {headers}"}

    @staticmethod
    async def apatch(
        url: str, data: Dict[str, Any], headers: Dict[str, str] = {}, **kwargs: Any
    ) -> Dict[str, Any]:
        return {"response": f"apatch {json.dumps(data)} {headers}"}

    @staticmethod
    def put(
        url: str, data: Dict[str, Any], headers: Dict[str, str] = {}, **kwargs: Any
    ) -> Dict[str, Any]:
        return {"response": f"put {json.dumps(data)} {headers}"}

    @staticmethod
    async def aput(
        url: str, data: Dict[str, Any], headers: Dict[str, str] = {}, **kwargs: Any
    ) -> Dict[str, Any]:
        return {"response": f"aput {json.dumps(data)} {headers}"}

    @staticmethod
    def delete(url: str, headers: Dict[str, str] = {}, **kwargs: Any) -> Dict[str, Any]:
        return {"response": f"delete_response {headers}"}

    @staticmethod
    async def adelete(
        url: str, headers: Dict[str, str] = {}, **kwargs: Any
    ) -> Dict[str, Any]:
        return {"response": f"adelete_response {headers}"}


@pytest.fixture
def mock_json_requests_wrapper() -> JsonRequestsWrapper:
    return _MockJsonRequestsWrapper()


def test_requests_get_tool_json(
    mock_json_requests_wrapper: JsonRequestsWrapper,
) -> None:
    tool = RequestsGetTool(
        requests_wrapper=mock_json_requests_wrapper, allow_dangerous_requests=True
    )
    input_text = '{"url": "https://example.com"}'
    assert tool.run(input_text) == {"response": "get_response {}"}
    assert asyncio.run(tool.arun(input_text)) == {"response": "aget_response {}"}


def test_requests_get_tool_json_with_headers(
    mock_json_requests_wrapper: JsonRequestsWrapper,
) -> None:
    tool = RequestsGetTool(
        requests_wrapper=mock_json_requests_wrapper, allow_dangerous_requests=True
    )
    headers = {"Key": "Value"}
    input_text = '{"url": "https://example.com", "headers": {"Key": "Value"}}'
    assert tool.run(input_text) == {"response": f"get_response {headers}"}
    assert asyncio.run(tool.arun(input_text)) == {
        "response": f"aget_response {headers}"
    }


def test_requests_post_tool_json(
    mock_json_requests_wrapper: JsonRequestsWrapper,
) -> None:
    tool = RequestsPostTool(
        requests_wrapper=mock_json_requests_wrapper, allow_dangerous_requests=True
    )
    input_text = '{"url": "https://example.com", "data": {"key": "value"}}'
    assert tool.run(input_text) == {"response": 'post {"key": "value"} {}'}
    assert asyncio.run(tool.arun(input_text)) == {
        "response": 'apost {"key": "value"} {}'
    }


def test_requests_post_tool_json_with_headers(
    mock_json_requests_wrapper: JsonRequestsWrapper,
) -> None:
    tool = RequestsPostTool(
        requests_wrapper=mock_json_requests_wrapper, allow_dangerous_requests=True
    )
    headers = {"Key": "Value"}
    input_text = (
        '{"url": "https://example.com", '
        '"data": {"key": "value"}, '
        '"headers": {"Key": "Value"}}'
    )
    assert tool.run(input_text) == {"response": f'post {{"key": "value"}} {headers}'}
    assert asyncio.run(tool.arun(input_text)) == {
        "response": f'apost {{"key": "value"}} {headers}'
    }


def test_requests_patch_tool_json(
    mock_json_requests_wrapper: JsonRequestsWrapper,
) -> None:
    tool = RequestsPatchTool(
        requests_wrapper=mock_json_requests_wrapper, allow_dangerous_requests=True
    )
    input_text = '{"url": "https://example.com", "data": {"key": "value"}}'
    assert tool.run(input_text) == {"response": 'patch {"key": "value"} {}'}
    assert asyncio.run(tool.arun(input_text)) == {
        "response": 'apatch {"key": "value"} {}'
    }


def test_requests_patch_tool_json_with_headers(
    mock_json_requests_wrapper: JsonRequestsWrapper,
) -> None:
    tool = RequestsPatchTool(
        requests_wrapper=mock_json_requests_wrapper, allow_dangerous_requests=True
    )
    headers = {"Key": "Value"}
    input_text = (
        '{"url": "https://example.com", '
        '"data": {"key": "value"}, '
        '"headers": {"Key": "Value"}}'
    )
    assert tool.run(input_text) == {"response": f'patch {{"key": "value"}} {headers}'}
    assert asyncio.run(tool.arun(input_text)) == {
        "response": f'apatch {{"key": "value"}} {headers}'
    }


def test_requests_put_tool_json(
    mock_json_requests_wrapper: JsonRequestsWrapper,
) -> None:
    tool = RequestsPutTool(
        requests_wrapper=mock_json_requests_wrapper, allow_dangerous_requests=True
    )
    input_text = '{"url": "https://example.com", "data": {"key": "value"}}'
    assert tool.run(input_text) == {"response": 'put {"key": "value"} {}'}
    assert asyncio.run(tool.arun(input_text)) == {
        "response": 'aput {"key": "value"} {}'
    }


def test_requests_put_tool_json_with_headers(
    mock_json_requests_wrapper: JsonRequestsWrapper,
) -> None:
    tool = RequestsPutTool(
        requests_wrapper=mock_json_requests_wrapper, allow_dangerous_requests=True
    )
    headers = {"Key": "Value"}
    input_text = (
        '{"url": "https://example.com", '
        '"data": {"key": "value"}, '
        '"headers": {"Key": "Value"}}'
    )
    assert tool.run(input_text) == {"response": f'put {{"key": "value"}} {headers}'}
    assert asyncio.run(tool.arun(input_text)) == {
        "response": f'aput {{"key": "value"}} {headers}'
    }


def test_requests_delete_tool_json(
    mock_json_requests_wrapper: JsonRequestsWrapper,
) -> None:
    tool = RequestsDeleteTool(
        requests_wrapper=mock_json_requests_wrapper, allow_dangerous_requests=True
    )
    input_text = '{"url": "https://example.com"}'
    assert tool.run(input_text) == {"response": "delete_response {}"}
    assert asyncio.run(tool.arun(input_text)) == {"response": "adelete_response {}"}


def test_requests_delete_tool_json_with_headers(
    mock_json_requests_wrapper: JsonRequestsWrapper,
) -> None:
    tool = RequestsDeleteTool(
        requests_wrapper=mock_json_requests_wrapper, allow_dangerous_requests=True
    )
    headers = {"Key": "Value"}
    input_text = '{"url": "https://example.com", "headers": {"Key": "Value"}}'
    assert tool.run(input_text) == {"response": f"delete_response {headers}"}
    assert asyncio.run(tool.arun(input_text)) == {
        "response": f"adelete_response {headers}"
    }
