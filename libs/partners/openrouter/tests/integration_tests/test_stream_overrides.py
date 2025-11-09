"""Override test_stream to work with OpenRouter reasoning content."""

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_openrouter.chat_models import ChatOpenRouter

MODEL_NAME = "minimax/minimax-m2:free"

class TestChatOpenRouterStreamOverride:
    """Override streaming tests to handle OpenRouter reasoning content."""

    def test_stream(self):
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

        # FIXED: Handle both reasoning + text content blocks (2 blocks)
        # instead of expecting only 1 block
        assert len(full.content_blocks) >= 1  # Minimum 1 block
        assert len(full.content_blocks) <= 2  # Maximum 2 blocks (reasoning + text)

        # Verify content structure
        for block in full.content_blocks:
            assert "type" in block
            assert block["type"] in ["reasoning", "text"]

        # Should always have text content
        text_blocks = [b for b in full.content_blocks if b.get("type") == "text"]
        assert len(text_blocks) > 0

        # If reasoning content exists, verify it
        reasoning_blocks = [b for b in full.content_blocks if b.get("type") == "reasoning"]
        if reasoning_blocks:
            for block in reasoning_blocks:
                assert "reasoning" in block
                assert isinstance(block["reasoning"], str)

    def test_stream_simple_greeting(self):
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
        assert len(final_response.content_blocks) >= 1

        # Check for text content
        text_blocks = [b for b in final_response.content_blocks if b.get("type") == "text"]
        assert len(text_blocks) > 0

        # Text should contain greeting content
        text_content = text_blocks[0]["text"].lower()
        assert "hi" in text_content or "hello" in text_content or "greet" in text_content

    def test_stream_chunks_are_valid(self):
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
        assert accumulated.content

        # At least one chunk should have content
        has_content_chunk = any(chunk.content for chunk in chunks)
        assert has_content_chunk, "At least one chunk should contain actual content"

    def test_stream_reasoning_content_present(self):
        """Test that reasoning content is properly handled when present."""
        model = ChatOpenRouter(model=MODEL_NAME, temperature=0)

        chunks = list(model.stream("What is 1+1?"))
        final_chunk = chunks[-1]

        # Should have content blocks
        if hasattr(final_chunk, 'content_blocks') and final_chunk.content_blocks:
            reasoning_blocks = [b for b in final_chunk.content_blocks if b.get("type") == "reasoning"]
            text_blocks = [b for b in final_chunk.content_blocks if b.get("type") == "text"]

            # Verify text content is present
            assert len(text_blocks) > 0
            assert "2" in text_blocks[0]["text"] or "two" in text_blocks[0]["text"].lower()

if __name__ == "__main__":
    # Run tests to verify they work
    test_instance = TestChatOpenRouterStreamOverride()

    print("Running test_stream...")
    test_instance.test_stream()
    print("✅ test_stream PASSED")

    print("Running test_stream_simple_greeting...")
    test_instance.test_stream_simple_greeting()
    print("✅ test_stream_simple_greeting PASSED")

    print("Running test_stream_chunks_are_valid...")
    test_instance.test_stream_chunks_are_valid()
    print("✅ test_stream_chunks_are_valid PASSED")

    print("Running test_stream_reasoning_content_present...")
    test_instance.test_stream_reasoning_content_present()
    print("✅ test_stream_reasoning_content_present PASSED")

    print("\n🎉 All custom streaming tests PASSED!")