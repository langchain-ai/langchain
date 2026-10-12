"""Unit test for ToolMessage None coercion."""

from langchain_core.messages import ToolMessage


def test_tool_message_none_content_coercion() -> None:
    # content explicitly None
    m_none = ToolMessage(content=None, tool_call_id="call_1")
    assert m_none.content == ""

    # content omitted
    m_omitted = ToolMessage(tool_call_id="call_1")
    assert m_omitted.content == ""

    # numeric 0 content preserved
    m_zero = ToolMessage(content=0, tool_call_id="call_1")  # type: ignore[arg-type]
    assert m_zero.content == "0"

    # normal string preserved
    m_str = ToolMessage(content="hello", tool_call_id="call_1")
    assert m_str.content == "hello"

    # list of contents preserved
    m_list = ToolMessage(content=["a", "b"], tool_call_id="call_1")
    assert m_list.content == ["a", "b"]
