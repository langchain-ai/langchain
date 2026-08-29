from copy import deepcopy

from langchain_core.messages import AIMessage, AIMessageChunk

GROUNDING_METADATA = {
    "grounding_chunks": [{"web": {"uri": "https://example.com", "title": "Example"}}],
    "grounding_supports": [
        {
            "segment": {"start_index": 0, "end_index": 5},
            "grounding_chunk_indices": [0],
        }
    ],
}


def test_google_genai_content_blocks_does_not_mutate_ai_message_content() -> None:
    message = AIMessage(
        content=[{"type": "text", "text": "hello"}],
        response_metadata={
            "model_provider": "google_genai",
            "grounding_metadata": GROUNDING_METADATA,
        },
    )
    original = deepcopy(message.content)

    _ = message.content_blocks

    assert message.content == original


def test_google_genai_content_blocks_does_not_mutate_ai_message_chunk_content() -> None:
    message = AIMessageChunk(
        content=[{"type": "text", "text": "hello"}],
        response_metadata={
            "model_provider": "google_genai",
            "grounding_metadata": GROUNDING_METADATA,
        },
    )
    original = deepcopy(message.content)

    _ = message.content_blocks

    assert message.content == original
