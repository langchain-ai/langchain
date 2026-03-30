from langchain_openai.chat_models._compat import _consolidate_calls


def _call(tool_name: str, call_id: str = "call_1") -> dict:
    return {
        "type": "server_tool_call",
        "id": call_id,
        "name": tool_name,
        "args": {},
    }


def _result(call_id: str = "call_1", status: str = "success", output=None) -> dict:
    return {
        "type": "server_tool_result",
        "tool_call_id": call_id,
        "status": status,
        "output": output or [],
        "extras": {},
    }


def test_consolidate_calls_unknown_tool_does_not_crash() -> None:
    chunks = [_call("unknown_tool"), _result()]
    assert list(_consolidate_calls(chunks)) == []


def test_consolidate_calls_web_search_still_yields_result() -> None:
    chunks = [_call("web_search"), _result()]
    consolidated = list(_consolidate_calls(chunks))
    assert len(consolidated) == 1
    assert consolidated[0]["type"] == "web_search_call"


def test_consolidate_calls_known_and_unknown_tools_is_deterministic() -> None:
    chunks = [
        _call("unknown_tool", "call_unknown"),
        _result("call_unknown"),
        _call("web_search", "call_web"),
        _result("call_web"),
    ]
    consolidated = list(_consolidate_calls(chunks))
    assert len(consolidated) == 1
    assert consolidated[0]["type"] == "web_search_call"
