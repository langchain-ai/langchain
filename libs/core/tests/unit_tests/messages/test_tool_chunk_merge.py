from langchain_core.messages import ToolMessageChunk

def test_tool_message_chunk_add_none_content():
    """Test merging ToolMessageChunks with None content."""
    chunk1 = ToolMessageChunk(
        tool_call_id="call_123",
        content=None,
        id="msg1"
    )
    chunk2 = ToolMessageChunk(
        tool_call_id="call_123",
        content=None,
        id="msg1"
    )
    merged = chunk1 + chunk2
    assert merged.content is None

def test_tool_message_chunk_add_none_and_str_content():
    """Test merging ToolMessageChunks with None and string content."""
    chunk1 = ToolMessageChunk(
        tool_call_id="call_123",
        content=None,
        id="msg1"
    )
    chunk2 = ToolMessageChunk(
        tool_call_id="call_123",
        content="hello",
        id="msg1"
    )
    
    # None + Str
    merged1 = chunk1 + chunk2
    assert merged1.content == "hello"
    
    # Str + None
    merged2 = chunk2 + chunk1
    assert merged2.content == "hello"

def test_tool_message_chunk_add_str_content():
    """Test merging ToolMessageChunks with string content (ensure no regression)."""
    chunk1 = ToolMessageChunk(
        tool_call_id="call_123",
        content="hello",
        id="msg1"
    )
    chunk2 = ToolMessageChunk(
        tool_call_id="call_123",
        content=" world",
        id="msg1"
    )
    merged = chunk1 + chunk2
    assert merged.content == "hello world"
