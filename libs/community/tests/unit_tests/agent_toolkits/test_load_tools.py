from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools.requests.tool import (
    RequestsDeleteTool,
    RequestsGetTool,
    RequestsPatchTool,
    RequestsPostTool,
    RequestsPutTool,
)


def test_load_request_tools() -> None:
    request_tools = load_tools(["requests_all"], allow_dangerous_tools=True)
    assert len(request_tools) == 5
    assert any(isinstance(tool, RequestsDeleteTool) for tool in request_tools)
    assert any(isinstance(tool, RequestsGetTool) for tool in request_tools)
    assert any(isinstance(tool, RequestsPatchTool) for tool in request_tools)
    assert any(isinstance(tool, RequestsPostTool) for tool in request_tools)
    assert any(isinstance(tool, RequestsPutTool) for tool in request_tools)
