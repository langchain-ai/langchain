"""Tests for Google GenAI block translator."""

from langchain_core.messages import AIMessageChunk
from langchain_core.messages.block_translators.google_genai import (
    translate_content_chunk,
    translate_grounding_metadata_to_citations,
)
from langchain_core.messages.tool import tool_call_chunk


def test_translate_grounding_metadata_web() -> None:
    """Test translation of web grounding metadata to citations."""
    grounding_metadata = {
        "grounding_chunks": [
            {
                "web": {
                    "uri": "https://example.com",
                    "title": "Example Site",
                },
                "maps": None,
            }
        ],
        "grounding_supports": [
            {
                "segment": {
                    "start_index": 0,
                    "end_index": 13,
                    "text": "Test response",
                },
                "grounding_chunk_indices": [0],
                "confidence_scores": [],
            }
        ],
        "web_search_queries": ["test query"],
    }

    citations = translate_grounding_metadata_to_citations(grounding_metadata)

    assert len(citations) == 1
    citation = citations[0]
    assert citation["type"] == "citation"
    assert citation.get("url") == "https://example.com"
    assert citation.get("title") == "Example Site"
    assert citation.get("start_index") == 0
    assert citation.get("end_index") == 13
    assert citation.get("cited_text") == "Test response"

    extras = citation.get("extras", {})["google_ai_metadata"]
    assert extras["web_search_queries"] == ["test query"]
    assert extras["grounding_chunk_index"] == 0
    assert "place_id" not in extras


def test_translate_grounding_metadata_maps() -> None:
    """Test translation of maps grounding metadata to citations."""
    grounding_metadata = {
        "grounding_chunks": [
            {
                "web": None,
                "maps": {
                    "uri": "https://maps.google.com/?cid=13100894621228039586",
                    "title": "Heaven on 7th Marketplace",
                    "placeId": "places/ChIJ0-zA1vBZwokRon0fGj-6z7U",
                },
            }
        ],
        "grounding_supports": [
            {
                "segment": {
                    "start_index": 0,
                    "end_index": 25,
                    "text": "Great Italian restaurant",
                },
                "grounding_chunk_indices": [0],
                "confidence_scores": [0.95],
            }
        ],
        "web_search_queries": [],
    }

    citations = translate_grounding_metadata_to_citations(grounding_metadata)

    assert len(citations) == 1
    citation = citations[0]
    assert citation["type"] == "citation"
    assert citation.get("url") == "https://maps.google.com/?cid=13100894621228039586"
    assert citation.get("title") == "Heaven on 7th Marketplace"
    assert citation.get("start_index") == 0
    assert citation.get("end_index") == 25
    assert citation.get("cited_text") == "Great Italian restaurant"

    extras = citation.get("extras", {})["google_ai_metadata"]
    assert extras["web_search_queries"] == []
    assert extras["grounding_chunk_index"] == 0
    assert extras["confidence_scores"] == [0.95]
    assert extras["place_id"] == "places/ChIJ0-zA1vBZwokRon0fGj-6z7U"


def test_translate_grounding_metadata_none() -> None:
    """Test translation when both web and maps are None."""
    grounding_metadata = {
        "grounding_chunks": [
            {
                "web": None,
                "maps": None,
            }
        ],
        "grounding_supports": [
            {
                "segment": {
                    "start_index": 0,
                    "end_index": 10,
                    "text": "test text",
                },
                "grounding_chunk_indices": [0],
                "confidence_scores": [],
            }
        ],
        "web_search_queries": [],
    }

    citations = translate_grounding_metadata_to_citations(grounding_metadata)

    # Should still create citation but without url/title fields when None
    assert len(citations) == 1
    citation = citations[0]
    assert citation["type"] == "citation"
    # url and title are omitted when None
    assert "url" not in citation
    assert "title" not in citation
    assert citation.get("start_index") == 0
    assert citation.get("end_index") == 10
    assert citation.get("cited_text") == "test text"


def test_translate_grounding_metadata_confidence_scores_none() -> None:
    """Test translation when confidence_scores is None (API returns this)."""
    grounding_metadata = {
        "grounding_chunks": [
            {
                "web": None,
                "maps": {
                    "uri": "https://maps.google.com/?cid=123",
                    "title": "Test Restaurant",
                    "placeId": "places/ChIJ123",
                },
            }
        ],
        "grounding_supports": [
            {
                "segment": {
                    "start_index": 0,
                    "end_index": 10,
                    "text": "test text",
                },
                "grounding_chunk_indices": [0],
                "confidence_scores": None,  # API returns None, not []
            }
        ],
        "web_search_queries": ["test query"],
    }

    citations = translate_grounding_metadata_to_citations(grounding_metadata)

    assert len(citations) == 1
    extras = citations[0].get("extras", {})["google_ai_metadata"]
    # Should convert None to empty list
    assert extras["confidence_scores"] == []
    assert isinstance(extras["confidence_scores"], list)


def test_translate_grounding_metadata_multiple_chunks() -> None:
    """Test translation with multiple grounding chunks."""
    grounding_metadata = {
        "grounding_chunks": [
            {
                "web": {
                    "uri": "https://example1.com",
                    "title": "Example 1",
                },
                "maps": None,
            },
            {
                "web": None,
                "maps": {
                    "uri": "https://maps.google.com/?cid=123",
                    "title": "Place 1",
                    "placeId": "places/123",
                },
            },
        ],
        "grounding_supports": [
            {
                "segment": {
                    "start_index": 0,
                    "end_index": 10,
                    "text": "First part",
                },
                "grounding_chunk_indices": [0, 1],
                "confidence_scores": [],
            }
        ],
        "web_search_queries": [],
    }

    citations = translate_grounding_metadata_to_citations(grounding_metadata)

    # Should create two citations, one for each chunk
    assert len(citations) == 2

    # First citation from web chunk
    assert citations[0].get("url") == "https://example1.com"
    assert citations[0].get("title") == "Example 1"
    assert "place_id" not in citations[0].get("extras", {})["google_ai_metadata"]

    # Second citation from maps chunk
    assert citations[1].get("url") == "https://maps.google.com/?cid=123"
    assert citations[1].get("title") == "Place 1"
    assert (
        citations[1].get("extras", {})["google_ai_metadata"]["place_id"] == "places/123"
    )


def test_translate_content_chunk_intermediate_streaming() -> None:
    """Intermediate chunks should have `tool_call_chunk` in `content_blocks`."""
    chunk = AIMessageChunk(
        content=[],
        tool_call_chunks=[
            tool_call_chunk(name="my_tool", args='{"arg": "value"}', id="123", index=0)
        ],
        response_metadata={"model_provider": "google_genai"},
        # No chunk_position set (intermediate chunk)
    )

    blocks = translate_content_chunk(chunk)
    tool_blocks = [b for b in blocks if b.get("type") == "tool_call_chunk"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0].get("name") == "my_tool"
    assert tool_blocks[0].get("args") == '{"arg": "value"}'
    assert tool_blocks[0].get("index") == 0


def test_translate_content_chunk_final_chunk() -> None:
    """Final chunks should have `tool_call` in `content_blocks`."""
    chunk = AIMessageChunk(
        content=[],
        tool_call_chunks=[
            tool_call_chunk(name="my_tool", args='{"arg": "value"}', id="123")
        ],
        response_metadata={"model_provider": "google_genai"},
        chunk_position="last",  # Final chunk
    )

    blocks = translate_content_chunk(chunk)
    tool_blocks = [b for b in blocks if b.get("type") == "tool_call"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0].get("name") == "my_tool"


def test_translate_content_chunk_multiple_tool_calls() -> None:
    """Test intermediate chunk with multiple `tool_call_chunks`."""
    chunk = AIMessageChunk(
        content=[],
        tool_call_chunks=[
            tool_call_chunk(name="tool_a", args='{"a": 1}', id="1", index=0),
            tool_call_chunk(name="tool_b", args='{"b": 2}', id="2", index=1),
        ],
        response_metadata={"model_provider": "google_genai"},
    )

    blocks = translate_content_chunk(chunk)
    tool_blocks = [b for b in blocks if b.get("type") == "tool_call_chunk"]
    assert len(tool_blocks) == 2
    assert tool_blocks[0].get("name") == "tool_a"
    assert tool_blocks[0].get("index") == 0
    assert tool_blocks[1].get("name") == "tool_b"
    assert tool_blocks[1].get("index") == 1


def test_translate_content_chunk_with_text_and_tool_call() -> None:
    """Test intermediate chunk with both text `content` and `tool_call_chunks`."""
    chunk = AIMessageChunk(
        content=[{"type": "text", "text": "Let me call a tool."}],
        tool_call_chunks=[
            tool_call_chunk(name="my_tool", args='{"arg": "value"}', id="123", index=0)
        ],
        response_metadata={"model_provider": "google_genai"},
    )

    blocks = translate_content_chunk(chunk)

    text_blocks = [b for b in blocks if b.get("type") == "text"]
    tool_chunk_blocks = [b for b in blocks if b.get("type") == "tool_call_chunk"]

    assert len(text_blocks) == 1
    assert text_blocks[0].get("text") == "Let me call a tool."
    assert len(tool_chunk_blocks) == 1
    assert tool_chunk_blocks[0].get("name") == "my_tool"
