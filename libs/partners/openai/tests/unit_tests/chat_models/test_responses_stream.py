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


def _make_base_response(**overrides: Any) -> Response:
    """Build a minimal Response object for test event streams."""
    defaults: dict[str, Any] = {
        "id": "resp_123",
        "created_at": 1749734255.0,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "model": "gpt-4o-2024-08-06",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "background": False,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": Reasoning(effort=None, generate_summary=None, summary=None),
        "service_tier": "auto",
        "status": "in_progress",
        "text": ResponseTextConfig(format=ResponseFormatText(type="text")),
        "truncation": "disabled",
        "usage": None,
        "user": None,
    }
    defaults.update(overrides)
    return Response(**defaults)


# ---------------------------------------------------------------------------
# Server-side tool streaming tests
# ---------------------------------------------------------------------------


def test_server_tool_streaming_web_search() -> None:
    """Verify .added emits a minimal chunk and .done the full result.

    After aggregation the web_search_call content block should have
    non-duplicated string fields and the complete action payload.
    """
    from openai.types.responses.response_function_web_search import (
        ActionSearch,
        ResponseFunctionWebSearch,
    )

    ws_added = ResponseFunctionWebSearch.model_construct(
        id="ws_001",
        type="web_search_call",
        status="in_progress",
    )
    ws_done = ResponseFunctionWebSearch(
        id="ws_001",
        type="web_search_call",
        status="completed",
        action=ActionSearch(type="search", query="latest AI news"),
    )

    stream_events = [
        ResponseCreatedEvent(
            response=_make_base_response(),
            sequence_number=0,
            type="response.created",
        ),
        ResponseInProgressEvent(
            response=_make_base_response(),
            sequence_number=1,
            type="response.in_progress",
        ),
        # --- web search: added (partial) ---
        ResponseOutputItemAddedEvent(
            item=ws_added,
            output_index=0,
            sequence_number=2,
            type="response.output_item.added",
        ),
        # --- web search: done (complete) ---
        ResponseOutputItemDoneEvent(
            item=ws_done,
            output_index=0,
            sequence_number=3,
            type="response.output_item.done",
        ),
        # --- assistant text ---
        ResponseOutputItemAddedEvent(
            item=ResponseOutputMessage(
                id="msg_001",
                content=[],
                role="assistant",
                status="in_progress",
                type="message",
            ),
            output_index=1,
            sequence_number=4,
            type="response.output_item.added",
        ),
        ResponseContentPartAddedEvent(
            content_index=0,
            item_id="msg_001",
            output_index=1,
            part=ResponseOutputText(annotations=[], text="", type="output_text"),
            sequence_number=5,
            type="response.content_part.added",
        ),
        ResponseTextDeltaEvent(
            content_index=0,
            delta="Search results are here",
            item_id="msg_001",
            output_index=1,
            logprobs=[],
            sequence_number=6,
            type="response.output_text.delta",
        ),
        ResponseTextDoneEvent(
            content_index=0,
            item_id="msg_001",
            output_index=1,
            text="Search results are here",
            logprobs=[],
            sequence_number=7,
            type="response.output_text.done",
        ),
        ResponseContentPartDoneEvent(
            content_index=0,
            item_id="msg_001",
            output_index=1,
            part=ResponseOutputText(
                annotations=[], text="Search results are here", type="output_text"
            ),
            sequence_number=8,
            type="response.content_part.done",
        ),
        ResponseOutputItemDoneEvent(
            item=ResponseOutputMessage(
                id="msg_001",
                content=[
                    ResponseOutputText(
                        annotations=[],
                        text="Search results are here",
                        type="output_text",
                    ),
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
            output_index=1,
            sequence_number=9,
            type="response.output_item.done",
        ),
        ResponseCompletedEvent(
            response=_make_base_response(
                status="completed",
                service_tier="default",
                usage=ResponseUsage(
                    input_tokens=10,
                    input_tokens_details=InputTokensDetails(cached_tokens=0),
                    output_tokens=20,
                    output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                    total_tokens=30,
                ),
            ),
            sequence_number=10,
            type="response.completed",
        ),
    ]

    llm = ChatOpenAI(model="gpt-4o", use_responses_api=True)
    mock_client = MagicMock()

    def mock_create(*_args: Any, **_kwargs: Any) -> MockSyncContextManager:
        return MockSyncContextManager(stream_events)

    mock_client.responses.create = mock_create

    full: BaseMessageChunk | None = None
    chunks: list[AIMessageChunk] = []
    with patch.object(llm, "root_client", mock_client):
        for chunk in llm.stream("search the web"):
            assert isinstance(chunk, AIMessageChunk)
            full = chunk if full is None else full + chunk
            chunks.append(chunk)

    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)

    # Find the web_search_call content block
    ws_blocks = [
        b
        for b in full.content
        if isinstance(b, dict) and b.get("type") == "web_search_call"
    ]
    assert len(ws_blocks) == 1
    ws_block = ws_blocks[0]

    # id must NOT be duplicated
    assert ws_block["id"] == "ws_001"

    # status is concatenated from the two chunks (expected aggregation behavior)
    assert "in_progress" in ws_block["status"]
    assert "completed" in ws_block["status"]

    # action must come through from .done
    assert ws_block["action"] == {"type": "search", "query": "latest AI news"}

    # Verify the .added chunk was emitted first with minimal data (no 'action')
    ws_chunks = [
        c
        for c in chunks
        if isinstance(c.content, list)
        and any(
            isinstance(b, dict) and b.get("type") == "web_search_call"
            for b in c.content
        )
    ]
    assert len(ws_chunks) >= 2  # at least added + done
    first_ws = next(
        b
        for b in ws_chunks[0].content
        if isinstance(b, dict) and b.get("type") == "web_search_call"
    )
    assert first_ws["status"] == "in_progress"
    assert "action" not in first_ws  # minimal chunk has no action


def test_server_tool_streaming_code_interpreter_no_field_duplication() -> None:
    """Ensure string fields like container_id are NOT duplicated on aggregation.

    This is the core regression test for the partial-data handling fix.
    Previously, model_dump() on the .added item included container_id,
    which was then concatenated with the same value from .done.
    """
    from openai.types.responses import ResponseCodeInterpreterToolCall

    ci_added = ResponseCodeInterpreterToolCall.model_construct(
        id="ci_001",
        type="code_interpreter_call",
        status="in_progress",
        container_id="ctnr_abc",
    )
    ci_done = ResponseCodeInterpreterToolCall(
        id="ci_001",
        type="code_interpreter_call",
        status="completed",
        container_id="ctnr_abc",
        code="print(42)",
        outputs=[],
    )

    stream_events = [
        ResponseCreatedEvent(
            response=_make_base_response(),
            sequence_number=0,
            type="response.created",
        ),
        ResponseInProgressEvent(
            response=_make_base_response(),
            sequence_number=1,
            type="response.in_progress",
        ),
        ResponseOutputItemAddedEvent(
            item=ci_added,
            output_index=0,
            sequence_number=2,
            type="response.output_item.added",
        ),
        ResponseOutputItemDoneEvent(
            item=ci_done,
            output_index=0,
            sequence_number=3,
            type="response.output_item.done",
        ),
        # Minimal text output to complete the response
        ResponseOutputItemAddedEvent(
            item=ResponseOutputMessage(
                id="msg_001",
                content=[],
                role="assistant",
                status="in_progress",
                type="message",
            ),
            output_index=1,
            sequence_number=4,
            type="response.output_item.added",
        ),
        ResponseContentPartAddedEvent(
            content_index=0,
            item_id="msg_001",
            output_index=1,
            part=ResponseOutputText(annotations=[], text="", type="output_text"),
            sequence_number=5,
            type="response.content_part.added",
        ),
        ResponseTextDeltaEvent(
            content_index=0,
            delta="42",
            item_id="msg_001",
            output_index=1,
            logprobs=[],
            sequence_number=6,
            type="response.output_text.delta",
        ),
        ResponseTextDoneEvent(
            content_index=0,
            item_id="msg_001",
            output_index=1,
            text="42",
            logprobs=[],
            sequence_number=7,
            type="response.output_text.done",
        ),
        ResponseContentPartDoneEvent(
            content_index=0,
            item_id="msg_001",
            output_index=1,
            part=ResponseOutputText(annotations=[], text="42", type="output_text"),
            sequence_number=8,
            type="response.content_part.done",
        ),
        ResponseOutputItemDoneEvent(
            item=ResponseOutputMessage(
                id="msg_001",
                content=[
                    ResponseOutputText(annotations=[], text="42", type="output_text"),
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
            output_index=1,
            sequence_number=9,
            type="response.output_item.done",
        ),
        ResponseCompletedEvent(
            response=_make_base_response(
                status="completed",
                service_tier="default",
                usage=ResponseUsage(
                    input_tokens=10,
                    input_tokens_details=InputTokensDetails(cached_tokens=0),
                    output_tokens=5,
                    output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                    total_tokens=15,
                ),
            ),
            sequence_number=10,
            type="response.completed",
        ),
    ]

    llm = ChatOpenAI(model="gpt-4o", use_responses_api=True)
    mock_client = MagicMock()

    def mock_create(*_args: Any, **_kwargs: Any) -> MockSyncContextManager:
        return MockSyncContextManager(stream_events)

    mock_client.responses.create = mock_create

    full: BaseMessageChunk | None = None
    with patch.object(llm, "root_client", mock_client):
        for chunk in llm.stream("run code"):
            assert isinstance(chunk, AIMessageChunk)
            full = chunk if full is None else full + chunk

    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)

    ci_blocks = [
        b
        for b in full.content
        if isinstance(b, dict) and b.get("type") == "code_interpreter_call"
    ]
    assert len(ci_blocks) == 1
    ci_block = ci_blocks[0]

    # CRITICAL: container_id must be the original value, NOT duplicated
    assert ci_block["container_id"] == "ctnr_abc"

    # id must also not be duplicated
    assert ci_block["id"] == "ci_001"

    # status concatenation is expected
    assert "in_progress" in ci_block["status"]
    assert "completed" in ci_block["status"]

    # Complete data from .done must be present
    assert ci_block["code"] == "print(42)"
    assert ci_block["outputs"] == []


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
