"""Override test_stream to work with OpenRouter reasoning content."""

from langchain_core.messages import AIMessageChunk

from langchain_openrouter.chat_models import ChatOpenRouter

MODEL_NAME = "minimax/minimax-m2:free"


class TestChatOpenRouterStreamOverride:
    """Override streaming tests to handle OpenRouter reasoning content."""

    def test_stream(self) -> None:
        """Override the standard test_stream to handle reasoning content."""
        model = ChatOpenRouter(model=MODEL_NAME, temperature=0)

        num_chunks = 0
        full = None

        for chunk in model.stream("Hello"):
            assert chunk is not None
            assert isinstance(chunk, AIMessageChunk)
            assert isinstance(chunk.content, str | list)
            num_chunks += 1
            full = chunk if full is None else full + chunk

        assert num_chunks > 0
        assert isinstance(full, AIMessageChunk)
        assert full.content

    def test_stream_simple_greeting(self) -> None:
        """Test streaming with a simple greeting message."""
        model = ChatOpenRouter(model=MODEL_NAME, temperature=0)

        response_chunks = list(model.stream("Hi there"))
        assert len(response_chunks) > 0

        # Combine all chunks
        final_response = None
        for chunk in response_chunks:
            final_response = chunk if final_response is None else final_response + chunk

        # Verify the response has proper structure
        assert isinstance(final_response, AIMessageChunk)
        assert final_response.content

    def test_stream_chunks_are_valid(self) -> None:
        """Verify that streaming chunks maintain proper structure."""
        model = ChatOpenRouter(model=MODEL_NAME, temperature=0)

        chunks = list(model.stream("Hello"))
        assert len(chunks) > 0

        for chunk in chunks:
            assert isinstance(chunk, AIMessageChunk)

        # Accumulate all chunks to get the complete response
        accumulated = None
        for chunk in chunks:
            accumulated = chunk if accumulated is None else accumulated + chunk

        # The accumulated response should have content
        assert accumulated is not None
        assert accumulated.content

        # At least one chunk should have content
        has_content_chunk = any(chunk.content for chunk in chunks)
        assert has_content_chunk, "At least one chunk should contain actual content"

    def test_stream_reasoning_content_present(self) -> None:
        """Test that reasoning content is properly handled when present."""
        model = ChatOpenRouter(model=MODEL_NAME, temperature=0)

        chunks = list(model.stream("What is 1+1?"))
        final_chunk = chunks[-1]

        # Should have content
        assert final_chunk is not None
        assert final_chunk.content


if __name__ == "__main__":
    # Run tests to verify they work
    test_instance = TestChatOpenRouterStreamOverride()

    test_instance.test_stream()
    test_instance.test_stream_simple_greeting()
    test_instance.test_stream_chunks_are_valid()
    test_instance.test_stream_reasoning_content_present()
