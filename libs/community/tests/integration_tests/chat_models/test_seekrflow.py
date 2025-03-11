import pytest
from unittest.mock import MagicMock, AsyncMock
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models.seekrflow import ChatSeekrFlow



@pytest.fixture
def mock_seekr_client():
    """Fixture to create and return a mock SeekrFlow client."""
    # We’ll mock its chat.completions.create method
    mock_client = MagicMock()
    mock_client.chat.completions.create = MagicMock()
    return mock_client


def test_invoke_sync(mock_seekr_client):
    """Test the synchronous invoke method."""
    # Setup: Create a mock response object
    mock_seekr_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello from SeekrFlow!"))]
    )

    # Instantiate ChatSeekrFlow
    llm = ChatSeekrFlow(client=mock_seekr_client, model_name="test-model")

    # Prepare test messages
    messages = [HumanMessage(content="Say something!")]

    # Act: Call invoke
    response = llm.invoke(messages)

    # Assert: Verify the client was called correctly and check response
    mock_seekr_client.chat.completions.create.assert_called_once()
    assert response.content == "Hello from SeekrFlow!"


def test_invoke_with_system_message(mock_seekr_client):
    """Test invoke behavior when a system message is provided."""
    mock_seekr_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="System + User Response"))]
    )
    llm = ChatSeekrFlow(client=mock_seekr_client, model_name="test-model")

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="How's the weather?"),
    ]

    response = llm.invoke(messages)
    assert "System + User Response" in response.content


def test_stop_tokens(mock_seekr_client):
    """Test that stop tokens cause the response to be truncated."""
    mock_seekr_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="All good. STOP Extra text"))]
    )
    llm = ChatSeekrFlow(client=mock_seekr_client, model_name="test-model")

    # Provide a stop token: "STOP"
    messages = [HumanMessage(content="Give me an update")]
    response = llm.invoke(messages, stop=["STOP"])

    assert response.content == "All good. "


@pytest.mark.parametrize("streaming_enabled", [True, False])
def test_stream_sync(mock_seekr_client, streaming_enabled):
    """Test the synchronous streaming method (if implemented)."""
    # Simulate streaming chunks
    # e.g., each chunk has 'choices[0].delta.content'
    mock_stream_response = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="Part1 "))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="Part2 "))]),
    ]
    mock_seekr_client.chat.completions.create.return_value = mock_stream_response

    llm = ChatSeekrFlow(
        client=mock_seekr_client, model_name="test-model", streaming=streaming_enabled
    )

    messages = [HumanMessage(content="Stream to me!")]

    if not streaming_enabled:
        # Expect an error if streaming=False and we call .stream()
        with pytest.raises(ValueError):
            list(llm.stream(messages))
    else:
        # If streaming=True, we can iterate over chunks
        chunks = list(llm.stream(messages))
        # We expect 2 partial outputs
        assert len(chunks) == 2
        assert chunks[0].content == "Part1 "
        assert chunks[1].content == "Part2 "
