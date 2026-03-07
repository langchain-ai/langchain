from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseTextConfig,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from openai.types.responses.response import Response
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import Summary
from openai.types.responses.response_reasoning_summary_part_added_event import (
    Part as PartAdded,
)
from openai.types.responses.response_reasoning_summary_part_done_event import (
    Part as PartDone,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)
from openai.types.shared.reasoning import Reasoning
from openai.types.shared.response_format_text import ResponseFormatText

from langchain_openai import ChatOpenAI
from tests.unit_tests.chat_models.test_base import MockSyncContextManager

responses_stream = [
    ResponseCreatedEvent(
        response=Response(
            id="resp_123",
            created_at=1749734255.0,
            error=None,
            incomplete_details=None,
            instructions=None,
            metadata={},
            model="o4-mini-2025-04-16",
            object="response",
            output=[],
            parallel_tool_calls=True,
            temperature=1.0,
            tool_choice="auto",
            tools=[],
            top_p=1.0,
            background=False,
            max_output_tokens=None,
            previous_response_id=None,
            reasoning=Reasoning(
                effort="medium", generate_summary=None, summary="detailed"
            ),
            service_tier="auto",
            status="in_progress",
            text=ResponseTextConfig(format=ResponseFormatText(type="text")),
            truncation="disabled",
            usage=None,
            user=None,
        ),
        sequence_number=0,
        type="response.created",
    ),
    ResponseInProgressEvent(
        response=Response(
            id="resp_123",
            created_at=1749734255.0,
            error=None,
            incomplete_details=None,
            instructions=None,
            metadata={},
            model="o4-mini-2025-04-16",
            object="response",
            output=[],
            parallel_tool_calls=True,
            temperature=1.0,
            tool_choice="auto",
            tools=[],
            top_p=1.0,
            background=False,
            max_output_tokens=None,
            previous_response_id=None,
            reasoning=Reasoning(
                effort="medium", generate_summary=None, summary="detailed"
            ),
            service_tier="auto",
            status="in_progress",
            text=ResponseTextConfig(format=ResponseFormatText(type="text")),
            truncation="disabled",
            usage=None,
            user=None,
        ),
        sequence_number=1,
        type="response.in_progress",
    ),
    ResponseOutputItemAddedEvent(
        item=ResponseReasoningItem(
            id="rs_123",
            summary=[],
            type="reasoning",
            encrypted_content=None,
            status=None,
        ),
        output_index=0,
        sequence_number=2,
        type="response.output_item.added",
    ),
    ResponseReasoningSummaryPartAddedEvent(
        item_id="rs_123",
        output_index=0,
        part=PartAdded(text="", type="summary_text"),
        sequence_number=3,
        summary_index=0,
        type="response.reasoning_summary_part.added",
    ),
    ResponseReasoningSummaryTextDeltaEvent(
        delta="reasoning block",
        item_id="rs_123",
        output_index=0,
        sequence_number=4,
        summary_index=0,
        type="response.reasoning_summary_text.delta",
    ),
    ResponseReasoningSummaryTextDeltaEvent(
        delta=" one",
        item_id="rs_123",
        output_index=0,
        sequence_number=5,
        summary_index=0,
        type="response.reasoning_summary_text.delta",
    ),
    ResponseReasoningSummaryTextDoneEvent(
        item_id="rs_123",
        output_index=0,
        sequence_number=6,
        summary_index=0,
        text="reasoning block one",
        type="response.reasoning_summary_text.done",
    ),
    ResponseReasoningSummaryPartDoneEvent(
        item_id="rs_123",
        output_index=0,
        part=PartDone(text="reasoning block one", type="summary_text"),
        sequence_number=7,
        summary_index=0,
        type="response.reasoning_summary_part.done",
    ),
    ResponseReasoningSummaryPartAddedEvent(
        item_id="rs_123",
        output_index=0,
        part=PartAdded(text="", type="summary_text"),
        sequence_number=8,
        summary_index=1,
        type="response.reasoning_summary_part.added",
    ),
    ResponseReasoningSummaryTextDeltaEvent(
        delta="another reasoning",
        item_id="rs_123",
        output_index=0,
        sequence_number=9,
        summary_index=1,
        type="response.reasoning_summary_text.delta",
    ),
    ResponseReasoningSummaryTextDeltaEvent(
        delta=" block",
        item_id="rs_123",
        output_index=0,
        sequence_number=10,
        summary_index=1,
        type="response.reasoning_summary_text.delta",
    ),
    ResponseReasoningSummaryTextDoneEvent(
        item_id="rs_123",
        output_index=0,
        sequence_number=11,
        summary_index=1,
        text="another reasoning block",
        type="response.reasoning_summary_text.done",
    ),
    ResponseReasoningSummaryPartDoneEvent(
        item_id="rs_123",
        output_index=0,
        part=PartDone(text="another reasoning block", type="summary_text"),
        sequence_number=12,
        summary_index=1,
        type="response.reasoning_summary_part.done",
    ),
    ResponseOutputItemDoneEvent(
        item=ResponseReasoningItem(
            id="rs_123",
            summary=[
                Summary(text="reasoning block one", type="summary_text"),
                Summary(text="another reasoning block", type="summary_text"),
            ],
            type="reasoning",
            encrypted_content=None,
            status=None,
        ),
        output_index=0,
        sequence_number=13,
        type="response.output_item.done",
    ),
    ResponseOutputItemAddedEvent(
        item=ResponseOutputMessage(
            id="msg_123",
            content=[],
            role="assistant",
            status="in_progress",
            type="message",
        ),
        output_index=1,
        sequence_number=14,
        type="response.output_item.added",
    ),
    ResponseContentPartAddedEvent(
        content_index=0,
        item_id="msg_123",
        output_index=1,
        part=ResponseOutputText(annotations=[], text="", type="output_text"),
        sequence_number=15,
        type="response.content_part.added",
    ),
    ResponseTextDeltaEvent(
        content_index=0,
        delta="text block",
        item_id="msg_123",
        output_index=1,
        sequence_number=16,
        logprobs=[],
        type="response.output_text.delta",
    ),
    ResponseTextDeltaEvent(
        content_index=0,
        delta=" one",
        item_id="msg_123",
        output_index=1,
        sequence_number=17,
        logprobs=[],
        type="response.output_text.delta",
    ),
    ResponseTextDoneEvent(
        content_index=0,
        item_id="msg_123",
        output_index=1,
        sequence_number=18,
        text="text block one",
        logprobs=[],
        type="response.output_text.done",
    ),
    ResponseContentPartDoneEvent(
        content_index=0,
        item_id="msg_123",
        output_index=1,
        part=ResponseOutputText(
            annotations=[], text="text block one", type="output_text"
        ),
        sequence_number=19,
        type="response.content_part.done",
    ),
    ResponseContentPartAddedEvent(
        content_index=1,
        item_id="msg_123",
        output_index=1,
        part=ResponseOutputText(annotations=[], text="", type="output_text"),
        sequence_number=20,
        type="response.content_part.added",
    ),
    ResponseTextDeltaEvent(
        content_index=1,
        delta="another text",
        item_id="msg_123",
        output_index=1,
        sequence_number=21,
        logprobs=[],
        type="response.output_text.delta",
    ),
    ResponseTextDeltaEvent(
        content_index=1,
        delta=" block",
        item_id="msg_123",
        output_index=1,
        sequence_number=22,
        logprobs=[],
        type="response.output_text.delta",
    ),
    ResponseTextDoneEvent(
        content_index=1,
        item_id="msg_123",
        output_index=1,
        sequence_number=23,
        text="another text block",
        logprobs=[],
        type="response.output_text.done",
    ),
    ResponseContentPartDoneEvent(
        content_index=1,
        item_id="msg_123",
        output_index=1,
        part=ResponseOutputText(
            annotations=[], text="another text block", type="output_text"
        ),
        sequence_number=24,
        type="response.content_part.done",
    ),
    ResponseOutputItemDoneEvent(
        item=ResponseOutputMessage(
            id="msg_123",
            content=[
                ResponseOutputText(
                    annotations=[], text="text block one", type="output_text"
                ),
                ResponseOutputText(
                    annotations=[], text="another text block", type="output_text"
                ),
            ],
            role="assistant",
            status="completed",
            type="message",
        ),
        output_index=1,
        sequence_number=25,
        type="response.output_item.done",
    ),
    ResponseOutputItemAddedEvent(
        item=ResponseReasoningItem(
            id="rs_234",
            summary=[],
            type="reasoning",
            encrypted_content="encrypted-content",
            status=None,
        ),
        output_index=2,
        sequence_number=26,
        type="response.output_item.added",
    ),
    ResponseReasoningSummaryPartAddedEvent(
        item_id="rs_234",
        output_index=2,
        part=PartAdded(text="", type="summary_text"),
        sequence_number=27,
        summary_index=0,
        type="response.reasoning_summary_part.added",
    ),
    ResponseReasoningSummaryTextDeltaEvent(
        delta="more reasoning",
        item_id="rs_234",
        output_index=2,
        sequence_number=28,
        summary_index=0,
        type="response.reasoning_summary_text.delta",
    ),
    ResponseReasoningSummaryTextDoneEvent(
        item_id="rs_234",
        output_index=2,
        sequence_number=29,
        summary_index=0,
        text="more reasoning",
        type="response.reasoning_summary_text.done",
    ),
    ResponseReasoningSummaryPartDoneEvent(
        item_id="rs_234",
        output_index=2,
        part=PartDone(text="more reasoning", type="summary_text"),
        sequence_number=30,
        summary_index=0,
        type="response.reasoning_summary_part.done",
    ),
    ResponseReasoningSummaryPartAddedEvent(
        item_id="rs_234",
        output_index=2,
        part=PartAdded(text="", type="summary_text"),
        sequence_number=31,
        summary_index=1,
        type="response.reasoning_summary_part.added",
    ),
    ResponseReasoningSummaryTextDeltaEvent(
        delta="still more reasoning",
        item_id="rs_234",
        output_index=2,
        sequence_number=32,
        summary_index=1,
        type="response.reasoning_summary_text.delta",
    ),
    ResponseReasoningSummaryTextDoneEvent(
        item_id="rs_234",
        output_index=2,
        sequence_number=33,
        summary_index=1,
        text="still more reasoning",
        type="response.reasoning_summary_text.done",
    ),
    ResponseReasoningSummaryPartDoneEvent(
        item_id="rs_234",
        output_index=2,
        part=PartDone(text="still more reasoning", type="summary_text"),
        sequence_number=34,
        summary_index=1,
        type="response.reasoning_summary_part.done",
    ),
    ResponseOutputItemDoneEvent(
        item=ResponseReasoningItem(
            id="rs_234",
            summary=[
                Summary(text="more reasoning", type="summary_text"),
                Summary(text="still more reasoning", type="summary_text"),
            ],
            type="reasoning",
            encrypted_content="encrypted-content",
            status=None,
        ),
        output_index=2,
        sequence_number=35,
        type="response.output_item.done",
    ),
    ResponseOutputItemAddedEvent(
        item=ResponseOutputMessage(
            id="msg_234",
            content=[],
            role="assistant",
            status="in_progress",
            type="message",
        ),
        output_index=3,
        sequence_number=36,
        type="response.output_item.added",
    ),
    ResponseContentPartAddedEvent(
        content_index=0,
        item_id="msg_234",
        output_index=3,
        part=ResponseOutputText(annotations=[], text="", type="output_text"),
        sequence_number=37,
        type="response.content_part.added",
    ),
    ResponseTextDeltaEvent(
        content_index=0,
        delta="more",
        item_id="msg_234",
        output_index=3,
        sequence_number=38,
        logprobs=[],
        type="response.output_text.delta",
    ),
    ResponseTextDoneEvent(
        content_index=0,
        item_id="msg_234",
        output_index=3,
        sequence_number=39,
        text="more",
        logprobs=[],
        type="response.output_text.done",
    ),
    ResponseContentPartDoneEvent(
        content_index=0,
        item_id="msg_234",
        output_index=3,
        part=ResponseOutputText(annotations=[], text="more", type="output_text"),
        sequence_number=40,
        type="response.content_part.done",
    ),
    ResponseContentPartAddedEvent(
        content_index=1,
        item_id="msg_234",
        output_index=3,
        part=ResponseOutputText(annotations=[], text="", type="output_text"),
        sequence_number=41,
        type="response.content_part.added",
    ),
    ResponseTextDeltaEvent(
        content_index=1,
        delta="text",
        item_id="msg_234",
        output_index=3,
        sequence_number=42,
        logprobs=[],
        type="response.output_text.delta",
    ),
    ResponseTextDoneEvent(
        content_index=1,
        item_id="msg_234",
        output_index=3,
        sequence_number=43,
        text="text",
        logprobs=[],
        type="response.output_text.done",
    ),
    ResponseContentPartDoneEvent(
        content_index=1,
        item_id="msg_234",
        output_index=3,
        part=ResponseOutputText(annotations=[], text="text", type="output_text"),
        sequence_number=44,
        type="response.content_part.done",
    ),
    ResponseOutputItemDoneEvent(
        item=ResponseOutputMessage(
            id="msg_234",
            content=[
                ResponseOutputText(annotations=[], text="more", type="output_text"),
                ResponseOutputText(annotations=[], text="text", type="output_text"),
            ],
            role="assistant",
            status="completed",
            type="message",
        ),
        output_index=3,
        sequence_number=45,
        type="response.output_item.done",
    ),
    ResponseCompletedEvent(
        response=Response(
            id="resp_123",
            created_at=1749734255.0,
            error=None,
            incomplete_details=None,
            instructions=None,
            metadata={},
            model="o4-mini-2025-04-16",
            object="response",
            output=[
                ResponseReasoningItem(
                    id="rs_123",
                    summary=[
                        Summary(text="reasoning block one", type="summary_text"),
                        Summary(text="another reasoning block", type="summary_text"),
                    ],
                    type="reasoning",
                    encrypted_content=None,
                    status=None,
                ),
                ResponseOutputMessage(
                    id="msg_123",
                    content=[
                        ResponseOutputText(
                            annotations=[], text="text block one", type="output_text"
                        ),
                        ResponseOutputText(
                            annotations=[],
                            text="another text block",
                            type="output_text",
                        ),
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                ),
                ResponseReasoningItem(
                    id="rs_234",
                    summary=[
                        Summary(text="more reasoning", type="summary_text"),
                        Summary(text="still more reasoning", type="summary_text"),
                    ],
                    type="reasoning",
                    encrypted_content="encrypted-content",
                    status=None,
                ),
                ResponseOutputMessage(
                    id="msg_234",
                    content=[
                        ResponseOutputText(
                            annotations=[], text="more", type="output_text"
                        ),
                        ResponseOutputText(
                            annotations=[], text="text", type="output_text"
                        ),
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                ),
            ],
            parallel_tool_calls=True,
            temperature=1.0,
            tool_choice="auto",
            tools=[],
            top_p=1.0,
            background=False,
            max_output_tokens=None,
            previous_response_id=None,
            reasoning=Reasoning(
                effort="medium", generate_summary=None, summary="detailed"
            ),
            service_tier="default",
            status="completed",
            text=ResponseTextConfig(format=ResponseFormatText(type="text")),
            truncation="disabled",
            usage=ResponseUsage(
                input_tokens=13,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens=71,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=64),
                total_tokens=84,
            ),
            user=None,
        ),
        sequence_number=46,
        type="response.completed",
    ),
]


def _strip_none(obj: Any) -> Any:
    """Recursively strip None values from dictionaries and lists."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj]
    return obj


@pytest.mark.parametrize(
    ("output_version", "expected_content"),
    [
        (
            "responses/v1",
            [
                {
                    "id": "rs_123",
                    "summary": [
                        {
                            "index": 0,
                            "type": "summary_text",
                            "text": "reasoning block one",
                        },
                        {
                            "index": 1,
                            "type": "summary_text",
                            "text": "another reasoning block",
                        },
                    ],
                    "type": "reasoning",
                    "index": 0,
                },
                {"type": "text", "text": "text block one", "index": 1, "id": "msg_123"},
                {
                    "type": "text",
                    "text": "another text block",
                    "index": 2,
                    "id": "msg_123",
                },
                {
                    "id": "rs_234",
                    "summary": [
                        {"index": 0, "type": "summary_text", "text": "more reasoning"},
                        {
                            "index": 1,
                            "type": "summary_text",
                            "text": "still more reasoning",
                        },
                    ],
                    "encrypted_content": "encrypted-content",
                    "type": "reasoning",
                    "index": 3,
                },
                {"type": "text", "text": "more", "index": 4, "id": "msg_234"},
                {"type": "text", "text": "text", "index": 5, "id": "msg_234"},
            ],
        ),
        (
            "v1",
            [
                {
                    "type": "reasoning",
                    "reasoning": "reasoning block one",
                    "id": "rs_123",
                    "index": "lc_rs_305f30",
                },
                {
                    "type": "reasoning",
                    "reasoning": "another reasoning block",
                    "id": "rs_123",
                    "index": "lc_rs_305f31",
                },
                {
                    "type": "text",
                    "text": "text block one",
                    "index": "lc_txt_1",
                    "id": "msg_123",
                },
                {
                    "type": "text",
                    "text": "another text block",
                    "index": "lc_txt_2",
                    "id": "msg_123",
                },
                {
                    "type": "reasoning",
                    "reasoning": "more reasoning",
                    "id": "rs_234",
                    "extras": {"encrypted_content": "encrypted-content"},
                    "index": "lc_rs_335f30",
                },
                {
                    "type": "reasoning",
                    "reasoning": "still more reasoning",
                    "id": "rs_234",
                    "index": "lc_rs_335f31",
                },
                {"type": "text", "text": "more", "index": "lc_txt_4", "id": "msg_234"},
                {"type": "text", "text": "text", "index": "lc_txt_5", "id": "msg_234"},
            ],
        ),
    ],
)
def test_responses_stream(output_version: str, expected_content: list[dict]) -> None:
    llm = ChatOpenAI(
        model="o4-mini", use_responses_api=True, output_version=output_version
    )
    mock_client = MagicMock()

    def mock_create(*args: Any, **kwargs: Any) -> MockSyncContextManager:
        return MockSyncContextManager(responses_stream)

    mock_client.responses.create = mock_create

    full: BaseMessageChunk | None = None
    chunks = []
    with patch.object(llm, "root_client", mock_client):
        for chunk in llm.stream("test"):
            assert isinstance(chunk, AIMessageChunk)
            full = chunk if full is None else full + chunk
            chunks.append(chunk)
    assert isinstance(full, AIMessageChunk)

    assert full.content == expected_content
    assert full.additional_kwargs == {}
    assert full.id == "resp_123"

    # Test reconstruction
    payload = llm._get_request_payload([full])
    completed = [
        item
        for item in responses_stream
        if item.type == "response.completed"  # type: ignore[attr-defined]
    ]
    assert len(completed) == 1
    response = completed[0].response  # type: ignore[attr-defined]

    assert len(response.output) == len(payload["input"])
    for idx, item in enumerate(response.output):
        dumped = _strip_none(item.model_dump())
        _ = dumped.pop("status", None)
        assert dumped == payload["input"][idx]


def test_responses_stream_with_image_generation_multiple_calls() -> None:
    """Test that streaming with image_generation tool works across multiple calls.

    Regression test: image_generation tool should not be mutated between calls,
    which would cause NotImplementedError on subsequent invocations.
    """
    tools: list[dict[str, Any]] = [
        {"type": "image_generation"},
        {"type": "function", "name": "my_tool", "parameters": {}},
    ]
    llm = ChatOpenAI(
        model="gpt-4o",
        use_responses_api=True,
        streaming=True,
    )
    llm_with_tools = llm.bind_tools(tools)

    mock_client = MagicMock()

    def mock_create(*args: Any, **kwargs: Any) -> MockSyncContextManager:
        return MockSyncContextManager(responses_stream)

    mock_client.responses.create = mock_create

    # First call should work
    with patch.object(llm, "root_client", mock_client):
        chunks = list(llm_with_tools.stream("test"))
        assert len(chunks) > 0

    # Second call should also work (would fail before fix due to tool mutation)
    with patch.object(llm, "root_client", mock_client):
        chunks = list(llm_with_tools.stream("test again"))
        assert len(chunks) > 0


def test_responses_stream_completed_chunk_includes_tool_calls() -> None:
    """Test that the response.completed chunk includes tool_calls.

    Regression test for https://github.com/langchain-ai/langchain/issues/34660:
    When streaming with the Responses API and tool calls, the response.completed
    chunk should include tool_call_chunks so that the aggregated message has
    tool_calls populated. Previously, the code only extracted parsed/usage/metadata
    from the completed message but never extracted tool_calls.
    """
    from openai.types.responses import (
        ResponseFunctionCallArgumentsDeltaEvent,
        ResponseFunctionCallArgumentsDoneEvent,
        ResponseFunctionToolCall,
    )

    tool_call_stream = [
        ResponseCreatedEvent(
            response=Response(
                id="resp_tool_123",
                created_at=1749734255.0,
                model="gpt-4o-mini",
                object="response",
                output=[],
                tool_choice="auto",
                tools=[
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                            "required": ["location"],
                        },
                    }
                ],
                parallel_tool_calls=True,
                temperature=1.0,
                top_p=1.0,
                background=False,
                max_output_tokens=None,
                previous_response_id=None,
                reasoning=None,
                service_tier="default",
                status="in_progress",
                text=ResponseTextConfig(format=ResponseFormatText(type="text")),
                truncation="disabled",
                usage=None,
                user=None,
            ),
            sequence_number=0,
            type="response.created",
        ),
        ResponseInProgressEvent(
            response=Response(
                id="resp_tool_123",
                created_at=1749734255.0,
                model="gpt-4o-mini",
                object="response",
                output=[],
                tool_choice="auto",
                tools=[],
                parallel_tool_calls=True,
                temperature=1.0,
                top_p=1.0,
                background=False,
                max_output_tokens=None,
                previous_response_id=None,
                reasoning=None,
                service_tier="default",
                status="in_progress",
                text=ResponseTextConfig(format=ResponseFormatText(type="text")),
                truncation="disabled",
                usage=None,
                user=None,
            ),
            sequence_number=1,
            type="response.in_progress",
        ),
        # Tool call item added
        ResponseOutputItemAddedEvent(
            item=ResponseFunctionToolCall(
                id="fc_item_123",
                call_id="call_abc123",
                name="get_weather",
                arguments="",
                type="function_call",
                status="in_progress",
            ),
            output_index=0,
            sequence_number=2,
            type="response.output_item.added",
        ),
        # Arguments streamed in chunks
        ResponseFunctionCallArgumentsDeltaEvent(
            delta='{"locat',
            item_id="fc_item_123",
            output_index=0,
            sequence_number=3,
            type="response.function_call_arguments.delta",
        ),
        ResponseFunctionCallArgumentsDeltaEvent(
            delta='ion": "San ',
            item_id="fc_item_123",
            output_index=0,
            sequence_number=4,
            type="response.function_call_arguments.delta",
        ),
        ResponseFunctionCallArgumentsDeltaEvent(
            delta='Francisco"}',
            item_id="fc_item_123",
            output_index=0,
            sequence_number=5,
            type="response.function_call_arguments.delta",
        ),
        ResponseFunctionCallArgumentsDoneEvent(
            arguments='{"location": "San Francisco"}',
            item_id="fc_item_123",
            name="get_weather",
            output_index=0,
            sequence_number=6,
            type="response.function_call_arguments.done",
        ),
        # Item done
        ResponseOutputItemDoneEvent(
            item=ResponseFunctionToolCall(
                id="fc_item_123",
                call_id="call_abc123",
                name="get_weather",
                arguments='{"location": "San Francisco"}',
                type="function_call",
                status="completed",
            ),
            output_index=0,
            sequence_number=7,
            type="response.output_item.done",
        ),
        # Response completed
        ResponseCompletedEvent(
            response=Response(
                id="resp_tool_123",
                created_at=1749734255.0,
                model="gpt-4o-mini",
                object="response",
                output=[
                    ResponseFunctionToolCall(
                        id="fc_item_123",
                        call_id="call_abc123",
                        name="get_weather",
                        arguments='{"location": "San Francisco"}',
                        type="function_call",
                        status="completed",
                    ),
                ],
                tool_choice="auto",
                tools=[
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                            "required": ["location"],
                        },
                    }
                ],
                parallel_tool_calls=True,
                temperature=1.0,
                top_p=1.0,
                background=False,
                max_output_tokens=None,
                previous_response_id=None,
                reasoning=None,
                service_tier="default",
                status="completed",
                text=ResponseTextConfig(format=ResponseFormatText(type="text")),
                truncation="disabled",
                usage=ResponseUsage(
                    input_tokens=50,
                    input_tokens_details=InputTokensDetails(cached_tokens=0),
                    output_tokens=20,
                    output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                    total_tokens=70,
                ),
                user=None,
            ),
            sequence_number=8,
            type="response.completed",
        ),
    ]

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key="test-key",
        use_responses_api=True,
        output_version="responses/v1",
    )
    mock_client = MagicMock()

    def mock_create(*args: Any, **kwargs: Any) -> MockSyncContextManager:
        return MockSyncContextManager(tool_call_stream)

    mock_client.responses.create = mock_create

    full: BaseMessageChunk | None = None
    last_chunk: AIMessageChunk | None = None
    with patch.object(llm, "root_client", mock_client):
        for chunk in llm.stream("What's the weather in San Francisco?"):
            assert isinstance(chunk, AIMessageChunk)
            full = chunk if full is None else full + chunk
            last_chunk = chunk

    assert isinstance(full, AIMessageChunk)

    # The last chunk (response.completed) should have tool_call_chunks
    assert last_chunk is not None
    assert last_chunk.chunk_position == "last"
    assert len(last_chunk.tool_call_chunks) > 0, (
        "response.completed chunk should contain tool_call_chunks "
        "(regression: https://github.com/langchain-ai/langchain/issues/34660)"
    )
    assert last_chunk.tool_call_chunks[0]["name"] == "get_weather"

    # The completed chunk by itself should provide full tool_calls
    assert len(last_chunk.tool_calls) > 0
    assert last_chunk.tool_calls[0]["name"] == "get_weather"
    assert last_chunk.tool_calls[0]["args"] == {"location": "San Francisco"}
