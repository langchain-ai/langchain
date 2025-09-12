"""Test raw_response functionality in ChatOpenAI integration."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_openai import ChatOpenAI


@pytest.mark.scheduled
def test_chat_openai_raw_response_generate() -> None:
    """Test that raw_response is included when requested in generate mode."""
    # Test with include_raw_response=True
    chat = ChatOpenAI(
        model="gpt-4o-mini", max_completion_tokens=50, include_raw_response=True
    )
    message = HumanMessage(content="Say hello")
    response = chat.invoke([message])

    assert isinstance(response, AIMessage)
    assert response.raw_response is not None
    assert isinstance(response.raw_response, dict)
    # Should have OpenAI response structure
    assert "id" in response.raw_response
    assert "object" in response.raw_response
    assert "model" in response.raw_response

    # Test with include_raw_response=False (default)
    chat_no_raw = ChatOpenAI(model="gpt-4o-mini", max_completion_tokens=50)
    response_no_raw = chat_no_raw.invoke([message])

    assert isinstance(response_no_raw, AIMessage)
    assert response_no_raw.raw_response is None


@pytest.mark.scheduled
def test_chat_openai_raw_response_per_invocation() -> None:
    """Test that raw_response can be enabled per invocation."""
    chat = ChatOpenAI(
        model="gpt-4o-mini", max_completion_tokens=50, include_raw_response=False
    )
    message = HumanMessage(content="Say hello")

    # Test per-invocation override
    response_with_raw = chat.invoke([message], include_raw_response=True)
    assert isinstance(response_with_raw, AIMessage)
    assert response_with_raw.raw_response is not None

    # Test normal invocation (should be None)
    response_without_raw = chat.invoke([message])
    assert isinstance(response_without_raw, AIMessage)
    assert response_without_raw.raw_response is None


@pytest.mark.scheduled
def test_chat_openai_raw_response_streaming() -> None:
    """Test that raw_response is included in streaming chunks when requested."""
    chat = ChatOpenAI(
        model="gpt-4o-mini", max_completion_tokens=50, include_raw_response=True
    )
    message = HumanMessage(content="Count to 3")

    chunks = []
    for chunk in chat.stream([message]):
        chunks.append(chunk)

    assert len(chunks) > 0
    # Check that chunks have raw_response
    for chunk in chunks:
        assert hasattr(chunk, "raw_response")
        # Raw response should be present for each chunk when enabled
        if chunk.raw_response is not None:
            assert isinstance(chunk.raw_response, (dict, object))


@pytest.mark.scheduled
@pytest.mark.parametrize("use_responses_api", [False, True])
def test_chat_openai_raw_response_both_apis(use_responses_api: bool) -> None:
    """Test raw_response works with both Chat Completions and Responses API."""
    chat = ChatOpenAI(
        model="gpt-4o-mini",
        max_completion_tokens=50,
        include_raw_response=True,
        use_responses_api=use_responses_api,
    )
    message = HumanMessage(content="Say hello briefly")
    response = chat.invoke([message])

    assert isinstance(response, AIMessage)
    assert response.raw_response is not None
    # Both APIs should return some form of raw response
    assert isinstance(response.raw_response, (dict, object))


def test_raw_response_not_serialized() -> None:
    """Test that raw_response is not included in serialization."""
    from langchain_core.load import dumps, loads

    # Create an AIMessage with raw_response
    mock_raw_response = {"sensitive": "data", "api_key": "secret"}
    message = AIMessage(content="Hello", raw_response=mock_raw_response)

    # Serialize and deserialize
    serialized = dumps(message)
    deserialized = loads(serialized)

    # raw_response should not be in serialized form
    assert "raw_response" not in serialized
    # Deserialized message should not have raw_response
    assert deserialized.raw_response is None
    # But other fields should be preserved
    assert deserialized.content == "Hello"
