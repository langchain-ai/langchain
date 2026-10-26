"""Tests for Google GenAI block translator."""

from copy import deepcopy
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.block_translators.google_genai import (
    translate_grounding_metadata_to_citations,
)
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk


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


def _image_block(data: str, index: int) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{data}"},
        "index": index,
    }


def _genai_chunk(**kwargs: Any) -> AIMessageChunk:
    return AIMessageChunk(
        response_metadata={"model_provider": "google_genai"}, **kwargs
    )


def test_content_blocks_preserve_index_for_images() -> None:
    """Streaming index must survive the `image_url` -> `image` conversion.

    Consumers that key blocks by index (the `v3` event stream) merge blocks that
    share one, so dropping the index collapses distinct images into one.
    """
    message = _genai_chunk(
        content=[_image_block("QUFBQQ==", 0), _image_block("QkJCQg==", 1)]
    )

    blocks = message.content_blocks

    assert [block["type"] for block in blocks] == ["image", "image"]
    assert [block.get("index") for block in blocks] == [0, 1]


def test_content_blocks_preserve_index_for_parallel_tool_calls() -> None:
    """Tool calls rebuilt from `tool_calls` must recover their chunk index."""
    message = _genai_chunk(
        content=[],
        tool_call_chunks=[
            create_tool_call_chunk(name="f1", args='{"a": 1}', id="id-1", index=0),
            create_tool_call_chunk(name="f2", args='{"b": 2}', id="id-2", index=1),
        ],
    )

    blocks = message.content_blocks

    assert [block["type"] for block in blocks] == ["tool_call", "tool_call"]
    assert [block.get("id") for block in blocks] == ["id-1", "id-2"]
    assert [block.get("index") for block in blocks] == ["lc_tc_0", "lc_tc_1"]


def test_content_blocks_tool_call_index_cannot_collide_with_content() -> None:
    """Tool call indices must not land in the content block index namespace.

    Tool call indices count from zero independently of content block indices, so
    an unprefixed copy lets an image at index 0 and the first tool call share a
    wire block -- the image starts the block and the tool call finishes it,
    silently dropping the image. A per-message translator cannot dedupe against
    indices used by earlier chunks, so the namespaces must be disjoint.
    """
    image_chunk = _genai_chunk(content=[_image_block("QUFBQQ==", 0)])
    tool_chunk = _genai_chunk(
        content=[],
        tool_call_chunks=[
            create_tool_call_chunk(name="f1", args='{"a": 1}', id="id-1", index=0)
        ],
    )

    image_index = image_chunk.content_blocks[0]["index"]
    tool_index = tool_chunk.content_blocks[0]["index"]

    assert image_index == 0
    assert tool_index != image_index


def test_content_blocks_omit_index_when_source_has_none() -> None:
    """Blocks without a source index must not gain a spurious one."""
    message = _genai_chunk(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,QUFBQQ=="},
            }
        ],
        tool_call_chunks=[
            create_tool_call_chunk(name="f1", args='{"a": 1}', id="id-1", index=None)
        ],
    )

    for block in message.content_blocks:
        assert "index" not in block


def test_content_blocks_index_propagation_is_one_to_one() -> None:
    """Each source item must map to exactly one indexed block.

    Index propagation copies an item's index onto the block it produced. If an
    item ever expanded into several blocks, copying one index onto all of them
    would recreate the very collision this propagation prevents, so indices must
    never be duplicated across blocks.
    """
    message = _genai_chunk(
        content=[
            {"type": "thinking", "thinking": "hmm", "index": 0},
            _image_block("QUFBQQ==", 1),
            {"type": "text", "text": "hi", "index": 2},
        ]
    )

    indices = [block["index"] for block in message.content_blocks if "index" in block]

    assert indices == [0, 1, 2]
    assert len(indices) == len(set(indices))


_GROUNDING_METADATA = {
    "grounding_chunks": [{"web": {"uri": "https://example.com", "title": "Example"}}],
    "grounding_supports": [
        {
            "segment": {"start_index": 0, "end_index": 5},
            "grounding_chunk_indices": [0],
        }
    ],
}


def test_grounding_citations_do_not_mutate_content() -> None:
    """Attaching grounding citations must leave `message.content` unchanged."""
    message = AIMessage(
        content=[{"type": "text", "text": "hello"}],
        response_metadata={
            "model_provider": "google_genai",
            "grounding_metadata": _GROUNDING_METADATA,
        },
    )
    original_content = deepcopy(message.content)

    blocks = message.content_blocks

    assert message.content == original_content
    assert "annotations" in blocks[0]
