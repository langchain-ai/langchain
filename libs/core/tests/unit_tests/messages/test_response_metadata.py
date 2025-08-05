"""Unit tests for ResponseMetadata TypedDict."""

from langchain_core.v1.messages import AIMessage, AIMessageChunk, ResponseMetadata


class TestResponseMetadata:
    """Test the ResponseMetadata TypedDict functionality."""

    def test_response_metadata_basic_fields(self) -> None:
        """Test ResponseMetadata with basic required fields."""
        metadata: ResponseMetadata = {
            "model_provider": "openai",
            "model_name": "gpt-4",
        }

        assert metadata.get("model_provider") == "openai"
        assert metadata.get("model_name") == "gpt-4"

    def test_response_metadata_is_optional(self) -> None:
        """Test that ResponseMetadata fields are optional due to total=False."""
        # Should be able to create empty ResponseMetadata
        metadata: ResponseMetadata = {}
        assert metadata == {}

        # Should be able to create with just one field
        metadata_partial: ResponseMetadata = {"model_provider": "anthropic"}
        assert metadata_partial.get("model_provider") == "anthropic"
        assert "model_name" not in metadata_partial

    def test_response_metadata_supports_extra_fields(self) -> None:
        """Test that ResponseMetadata supports provider-specific extra fields."""
        metadata: ResponseMetadata = {
            "model_provider": "openai",
            "model_name": "gpt-4-turbo",
            # Extra fields should be allowed
            "system_fingerprint": "fp_12345",
            "logprobs": None,
            "finish_reason": "stop",
            "request_id": "req_abc123",
        }

        assert metadata.get("model_provider") == "openai"
        assert metadata.get("model_name") == "gpt-4-turbo"
        assert metadata.get("system_fingerprint") == "fp_12345"
        assert metadata.get("logprobs") is None
        assert metadata.get("finish_reason") == "stop"
        assert metadata.get("request_id") == "req_abc123"

    def test_response_metadata_various_data_types(self) -> None:
        """Test that ResponseMetadata can store various data types in extra fields."""
        metadata: ResponseMetadata = {
            "model_provider": "anthropic",
            "model_name": "claude-3-sonnet",
            "string_field": "test_value",
            "int_field": 42,
            "float_field": 3.14,
            "bool_field": True,
            "none_field": None,
            "list_field": [1, 2, 3, "test"],
            "dict_field": {"nested": {"deeply": "nested_value"}},
        }

        assert metadata.get("string_field") == "test_value"
        assert metadata.get("int_field") == 42
        assert metadata.get("float_field") == 3.14
        assert metadata.get("bool_field") is True
        assert metadata.get("none_field") is None

        list_field = metadata.get("list_field")
        assert isinstance(list_field, list)
        assert list_field == [1, 2, 3, "test"]

        dict_field = metadata.get("dict_field")
        assert isinstance(dict_field, dict)
        nested = dict_field.get("nested")
        assert isinstance(nested, dict)
        assert nested.get("deeply") == "nested_value"

    def test_response_metadata_can_be_modified(self) -> None:
        """Test that ResponseMetadata can be modified after creation."""
        metadata: ResponseMetadata = {
            "model_provider": "openai",
            "model_name": "gpt-3.5-turbo",
        }

        # Modify existing fields
        metadata["model_name"] = "gpt-4"
        assert metadata.get("model_name") == "gpt-4"

        # Add new fields
        metadata["request_id"] = "req_12345"
        assert metadata.get("request_id") == "req_12345"

        # Modify nested structures
        metadata["headers"] = {"x-request-id": "abc123"}
        metadata["headers"]["x-rate-limit"] = "100"  # type: ignore[typeddict-item]

        headers = metadata.get("headers")
        assert isinstance(headers, dict)
        assert headers.get("x-request-id") == "abc123"
        assert headers.get("x-rate-limit") == "100"

    def test_response_metadata_provider_specific_examples(self) -> None:
        """Test ResponseMetadata with realistic provider-specific examples."""
        # OpenAI-style metadata
        openai_metadata: ResponseMetadata = {
            "model_provider": "openai",
            "model_name": "gpt-4-turbo-2024-04-09",
            "system_fingerprint": "fp_abc123",
            "created": 1234567890,
            "logprobs": None,
            "finish_reason": "stop",
        }

        assert openai_metadata.get("model_provider") == "openai"
        assert openai_metadata.get("system_fingerprint") == "fp_abc123"

        # Anthropic-style metadata
        anthropic_metadata: ResponseMetadata = {
            "model_provider": "anthropic",
            "model_name": "claude-3-sonnet-20240229",
            "stop_reason": "end_turn",
            "stop_sequence": None,
        }

        assert anthropic_metadata.get("model_provider") == "anthropic"
        assert anthropic_metadata.get("stop_reason") == "end_turn"

        # Custom provider metadata
        custom_metadata: ResponseMetadata = {
            "model_provider": "custom_llm_service",
            "model_name": "custom-model-v1",
            "service_tier": "premium",
            "rate_limit_info": {
                "requests_remaining": 100,
                "reset_time": "2024-01-01T00:00:00Z",
            },
            "response_time_ms": 1250,
        }

        assert custom_metadata.get("service_tier") == "premium"
        rate_limit = custom_metadata.get("rate_limit_info")
        assert isinstance(rate_limit, dict)
        assert rate_limit.get("requests_remaining") == 100


class TestResponseMetadataWithAIMessages:
    """Test ResponseMetadata integration with AI message classes."""

    def test_ai_message_with_response_metadata(self) -> None:
        """Test AIMessage with ResponseMetadata."""
        metadata: ResponseMetadata = {
            "model_provider": "openai",
            "model_name": "gpt-4",
            "system_fingerprint": "fp_xyz789",
        }

        message = AIMessage(content="Hello, world!", response_metadata=metadata)

        assert message.response_metadata == metadata
        assert message.response_metadata.get("model_provider") == "openai"
        assert message.response_metadata.get("model_name") == "gpt-4"
        assert message.response_metadata.get("system_fingerprint") == "fp_xyz789"

    def test_ai_message_chunk_with_response_metadata(self) -> None:
        """Test AIMessageChunk with ResponseMetadata."""
        metadata: ResponseMetadata = {
            "model_provider": "anthropic",
            "model_name": "claude-3-sonnet",
            "stream_id": "stream_12345",
        }

        chunk = AIMessageChunk(content="Hello", response_metadata=metadata)

        assert chunk.response_metadata == metadata
        assert chunk.response_metadata.get("stream_id") == "stream_12345"

    def test_ai_message_default_empty_response_metadata(self) -> None:
        """Test that AIMessage creates empty ResponseMetadata by default."""
        message = AIMessage(content="Test message")

        # Should have empty dict as default
        assert message.response_metadata == {}
        assert isinstance(message.response_metadata, dict)

    def test_ai_message_chunk_default_empty_response_metadata(self) -> None:
        """Test that AIMessageChunk creates empty ResponseMetadata by default."""
        chunk = AIMessageChunk(content="Test chunk")

        # Should have empty dict as default
        assert chunk.response_metadata == {}
        assert isinstance(chunk.response_metadata, dict)

    def test_response_metadata_merging_in_chunks(self) -> None:
        """Test that ResponseMetadata is properly merged when adding AIMessageChunks."""
        metadata1: ResponseMetadata = {
            "model_provider": "openai",
            "model_name": "gpt-4",
            "request_id": "req_123",
            "system_fingerprint": "fp_abc",
        }

        metadata2: ResponseMetadata = {
            "stream_chunk": 1,
            "finish_reason": "length",
        }

        chunk1 = AIMessageChunk(content="Hello ", response_metadata=metadata1)
        chunk2 = AIMessageChunk(content="world!", response_metadata=metadata2)

        merged = chunk1 + chunk2

        # Should have merged response_metadata
        assert merged.response_metadata.get("model_provider") == "openai"
        assert merged.response_metadata.get("model_name") == "gpt-4"
        assert merged.response_metadata.get("request_id") == "req_123"
        assert merged.response_metadata.get("stream_chunk") == 1
        assert merged.response_metadata.get("system_fingerprint") == "fp_abc"
        assert merged.response_metadata.get("finish_reason") == "length"

    def test_response_metadata_modification_after_message_creation(self) -> None:
        """Test that ResponseMetadata can be modified after message creation."""
        message = AIMessage(
            content="Initial message",
            response_metadata={"model_provider": "openai", "model_name": "gpt-3.5"},
        )

        # Modify existing field
        message.response_metadata["model_name"] = "gpt-4"
        assert message.response_metadata.get("model_name") == "gpt-4"

        # Add new field
        message.response_metadata["finish_reason"] = "stop"
        assert message.response_metadata.get("finish_reason") == "stop"

    def test_response_metadata_with_none_values(self) -> None:
        """Test ResponseMetadata handling of None values."""
        metadata: ResponseMetadata = {
            "model_provider": "openai",
            "model_name": "gpt-4",
            "system_fingerprint": None,
            "logprobs": None,
        }

        message = AIMessage(content="Test", response_metadata=metadata)

        assert message.response_metadata.get("system_fingerprint") is None
        assert message.response_metadata.get("logprobs") is None
        assert "system_fingerprint" in message.response_metadata
        assert "logprobs" in message.response_metadata


class TestResponseMetadataEdgeCases:
    """Test edge cases and error conditions for ResponseMetadata."""

    def test_response_metadata_with_complex_nested_structures(self) -> None:
        """Test ResponseMetadata with deeply nested and complex structures."""
        metadata: ResponseMetadata = {
            "model_provider": "custom",
            "model_name": "complex-model",
            "complex_data": {
                "level1": {
                    "level2": {
                        "level3": {
                            "deeply_nested": "value",
                            "array": [
                                {"item": 1, "metadata": {"nested": True}},
                                {"item": 2, "metadata": {"nested": False}},
                            ],
                        }
                    }
                }
            },
        }

        complex_data = metadata.get("complex_data")
        assert isinstance(complex_data, dict)
        level1 = complex_data.get("level1")
        assert isinstance(level1, dict)
        level2 = level1.get("level2")
        assert isinstance(level2, dict)
        level3 = level2.get("level3")
        assert isinstance(level3, dict)

        assert level3.get("deeply_nested") == "value"
        array = level3.get("array")
        assert isinstance(array, list)
        assert len(array) == 2
        assert array[0]["item"] == 1
        assert array[0]["metadata"]["nested"] is True

    def test_response_metadata_large_data(self) -> None:
        """Test ResponseMetadata with large amounts of data."""
        # Create metadata with many fields
        large_metadata: ResponseMetadata = {
            "model_provider": "test_provider",
            "model_name": "test_model",
        }

        # Add 100 extra fields
        for i in range(100):
            large_metadata[f"field_{i}"] = f"value_{i}"  # type: ignore[literal-required]

        message = AIMessage(content="Test", response_metadata=large_metadata)

        # Verify all fields are accessible
        assert message.response_metadata.get("model_provider") == "test_provider"
        for i in range(100):
            assert message.response_metadata.get(f"field_{i}") == f"value_{i}"

    def test_response_metadata_empty_vs_none(self) -> None:
        """Test the difference between empty ResponseMetadata and None."""
        # Message with empty metadata
        message_empty = AIMessage(content="Test", response_metadata={})
        assert message_empty.response_metadata == {}
        assert isinstance(message_empty.response_metadata, dict)

        # Message with None metadata (should become empty dict)
        message_none = AIMessage(content="Test", response_metadata=None)
        assert message_none.response_metadata == {}
        assert isinstance(message_none.response_metadata, dict)

        # Default message (no metadata specified)
        message_default = AIMessage(content="Test")
        assert message_default.response_metadata == {}
        assert isinstance(message_default.response_metadata, dict)

    def test_response_metadata_preserves_original_dict_type(self) -> None:
        """Test that ResponseMetadata preserves the original dict when passed."""
        original_dict: ResponseMetadata = {
            "model_provider": "openai",
            "model_name": "gpt-4",
            "custom_field": "custom_value",
        }

        message = AIMessage(content="Test", response_metadata=original_dict)

        # Should be the same dict object
        assert message.response_metadata is original_dict

        # Modifications to the message's response_metadata should affect original
        message.response_metadata["new_field"] = "new_value"
        assert original_dict.get("new_field") == "new_value"
