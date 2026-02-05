from langchain_core.messages import ToolMessageChunk


def test_tool_message_chunk_add_none_content() -> None:
    chunk1 = ToolMessageChunk(
        tool_call_id="call_123",
        content=None,
        artifact=None,
        additional_kwargs={},
        response_metadata={},
        id="msg1",
    )

    chunk2 = ToolMessageChunk(
        tool_call_id="call_123",
        content=None,
        artifact=None,
        additional_kwargs={},
        response_metadata={},
        id="msg1",
    )

    result = chunk1 + chunk2
    assert result.content == ""

    chunk3 = ToolMessageChunk(tool_call_id="call_123", content="hello")
    result2 = chunk1 + chunk3
    assert result2.content == "hello"

    result3 = chunk3 + chunk1
    assert result3.content == "hello"


def test_tool_message_chunk_add_list_none() -> None:
    chunk1 = ToolMessageChunk(tool_call_id="call_123", content=None)
    chunk2 = ToolMessageChunk(
        tool_call_id="call_123", content=[{"type": "text", "text": "hi"}]
    )

    result = chunk1 + chunk2
    assert result.content == [{"type": "text", "text": "hi"}]

    result2 = chunk2 + chunk1
    assert result2.content == [{"type": "text", "text": "hi"}]
