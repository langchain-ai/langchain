from unittest.mock import MagicMock
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_core.messages import AIMessageChunk

def test_stream_usage_false_preserves_response_metadata():
    """Verify stream_usage=False preserves stop_reason and model_name while omitting usage_metadata (#39713)."""
    chat = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key="mock-key")
    
    # Mock message_start event
    mock_start = MagicMock()
    mock_start.type = "message_start"
    mock_start.message.model = "claude-3-5-sonnet-20241022"
    
    chunk_start, _ = chat._make_message_chunk_from_anthropic_event(
        mock_start, stream_usage=False, coerce_content_to_string=False
    )
    assert chunk_start is not None
    assert chunk_start.response_metadata.get("model_name") == "claude-3-5-sonnet-20241022"
    assert chunk_start.usage_metadata is None

    # Mock message_delta event
    mock_delta = MagicMock()
    mock_delta.type = "message_delta"
    mock_delta.delta.stop_reason = "max_tokens"
    mock_delta.delta.stop_sequence = None
    mock_delta.delta.container = None
    mock_delta.usage.input_tokens = 10
    mock_delta.usage.output_tokens = 20

    chunk_delta, _ = chat._make_message_chunk_from_anthropic_event(
        mock_delta, stream_usage=False, coerce_content_to_string=False
    )
    assert chunk_delta is not None
    assert chunk_delta.response_metadata.get("stop_reason") == "max_tokens"
    assert chunk_delta.usage_metadata is None
