"""Unit tests for metadata.user_id handling in ChatAnthropic."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from langchain_anthropic import ChatAnthropic

MODEL_NAME = "claude-3-5-sonnet-20241022"


def test_metadata_user_id_passed_to_api() -> None:
    """Test that metadata.user_id from config is passed to Anthropic API."""
    llm = ChatAnthropic(model=MODEL_NAME, anthropic_api_key="test-key")  # type: ignore[call-arg]

    with patch.object(llm, "_create") as mock_create:
        mock_create.return_value = MagicMock(
            content=[{"type": "text", "text": "test response"}],
            usage=MagicMock(input_tokens=10, output_tokens=5),
            id="test-id",
            model=MODEL_NAME,
            role="assistant",
            stop_reason="end_turn",
            stop_sequence=None,
            type="message",
        )

        messages = [HumanMessage(content="test")]
        config = {"metadata": {"user_id": "test-user-123"}}

        llm.invoke(messages, config=config)

        # Verify that _create was called with metadata containing user_id
        mock_create.assert_called_once()
        payload = mock_create.call_args[0][0]
        assert "metadata" in payload
        assert payload["metadata"]["user_id"] == "test-user-123"


def test_metadata_user_id_not_passed_when_absent() -> None:
    """Test that metadata is not added when user_id is not in config."""
    llm = ChatAnthropic(model=MODEL_NAME, anthropic_api_key="test-key")  # type: ignore[call-arg]

    with patch.object(llm, "_create") as mock_create:
        mock_create.return_value = MagicMock(
            content=[{"type": "text", "text": "test response"}],
            usage=MagicMock(input_tokens=10, output_tokens=5),
            id="test-id",
            model=MODEL_NAME,
            role="assistant",
            stop_reason="end_turn",
            stop_sequence=None,
            type="message",
        )

        messages = [HumanMessage(content="test")]
        config = {"metadata": {"other_field": "value"}}

        llm.invoke(messages, config=config)

        # Verify that _create was called without metadata
        mock_create.assert_called_once()
        payload = mock_create.call_args[0][0]
        assert "metadata" not in payload


def test_metadata_user_id_with_no_config() -> None:
    """Test that metadata is not added when config is not provided."""
    llm = ChatAnthropic(model=MODEL_NAME, anthropic_api_key="test-key")  # type: ignore[call-arg]

    with patch.object(llm, "_create") as mock_create:
        mock_create.return_value = MagicMock(
            content=[{"type": "text", "text": "test response"}],
            usage=MagicMock(input_tokens=10, output_tokens=5),
            id="test-id",
            model=MODEL_NAME,
            role="assistant",
            stop_reason="end_turn",
            stop_sequence=None,
            type="message",
        )

        messages = [HumanMessage(content="test")]

        llm.invoke(messages)

        # Verify that _create was called without metadata
        mock_create.assert_called_once()
        payload = mock_create.call_args[0][0]
        assert "metadata" not in payload


@pytest.mark.asyncio
async def test_async_metadata_user_id_passed_to_api() -> None:
    """Test that metadata.user_id from config is passed to Anthropic API in async mode."""
    llm = ChatAnthropic(model=MODEL_NAME, anthropic_api_key="test-key")  # type: ignore[call-arg]

    with patch.object(llm, "_acreate") as mock_acreate:
        mock_acreate.return_value = MagicMock(
            content=[{"type": "text", "text": "test response"}],
            usage=MagicMock(input_tokens=10, output_tokens=5),
            id="test-id",
            model=MODEL_NAME,
            role="assistant",
            stop_reason="end_turn",
            stop_sequence=None,
            type="message",
        )

        messages = [HumanMessage(content="test")]
        config = {"metadata": {"user_id": "test-user-456"}}

        await llm.ainvoke(messages, config=config)

        # Verify that _acreate was called with metadata containing user_id
        mock_acreate.assert_called_once()
        payload = mock_acreate.call_args[0][0]
        assert "metadata" in payload
        assert payload["metadata"]["user_id"] == "test-user-456"


def test_metadata_user_id_with_other_metadata_fields() -> None:
    """Test that user_id is extracted even when other metadata fields are present."""
    llm = ChatAnthropic(model=MODEL_NAME, anthropic_api_key="test-key")  # type: ignore[call-arg]

    with patch.object(llm, "_create") as mock_create:
        mock_create.return_value = MagicMock(
            content=[{"type": "text", "text": "test response"}],
            usage=MagicMock(input_tokens=10, output_tokens=5),
            id="test-id",
            model=MODEL_NAME,
            role="assistant",
            stop_reason="end_turn",
            stop_sequence=None,
            type="message",
        )

        messages = [HumanMessage(content="test")]
        config = {
            "metadata": {
                "user_id": "config-user",
                "session_id": "session-123",
                "other_field": "value",
            }
        }

        llm.invoke(messages, config=config)

        # Verify that only user_id is extracted to the API metadata
        mock_create.assert_called_once()
        payload = mock_create.call_args[0][0]
        assert "metadata" in payload
        assert payload["metadata"]["user_id"] == "config-user"
        # Other fields should not be in the API metadata
        assert "session_id" not in payload["metadata"]
        assert "other_field" not in payload["metadata"]



def test_metadata_user_id_in_payload_generation() -> None:
    """Test that metadata.user_id is correctly added to payload when passed via kwargs."""
    llm = ChatAnthropic(model=MODEL_NAME, anthropic_api_key="test-key")  # type: ignore[call-arg]

    messages = [HumanMessage(content="test")]

    # Test that _get_request_payload correctly handles metadata in kwargs
    payload = llm._get_request_payload(
        messages,
        metadata={"user_id": "direct-user-123"}
    )

    assert "metadata" in payload
    assert payload["metadata"]["user_id"] == "direct-user-123"
