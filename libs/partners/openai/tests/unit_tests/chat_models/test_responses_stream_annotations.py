from typing import Any, Optional
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from openai.types.responses import (
    FileSearchTool,
    FunctionTool,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFileSearchCallCompletedEvent,
    ResponseFileSearchCallInProgressEvent,
    ResponseFileSearchCallSearchingEvent,
    ResponseFileSearchToolCall,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseTextConfig,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from openai.types.responses.response import Response, ResponseUsage
from openai.types.responses.response_output_text import (
    AnnotationFileCitation,
    ResponseOutputText,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)
from openai.types.shared.reasoning import Reasoning
from openai.types.shared.response_format_text import ResponseFormatText

from langchain_openai import ChatOpenAI
from tests.unit_tests.chat_models.test_base import MockSyncContextManager

responses_stream = [
    ResponseCreatedEvent(
        response=Response(
            id="resp_123",
            created_at=1757346128.0,
            error=None,
            incomplete_details=None,
            instructions=None,
            metadata={},
            model="gpt-4.1-mini-2025-04-14",
            object="response",
            output=[],
            parallel_tool_calls=True,
            temperature=1.0,
            tool_choice="auto",
            tools=[
                FileSearchTool(
                    type="file_search",
                    vector_store_ids=["vs_123"],
                    filters=None,
                    max_num_results=20,
                ),
                FunctionTool(
                    name="grade",
                    parameters={
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "type": "object",
                        "additionalProperties": False,
                    },
                    strict=True,
                    type="function",
                    description="Grade a given answer.",
                ),
            ],
            top_p=1.0,
            background=False,
            conversation=None,
            max_output_tokens=None,
            max_tool_calls=None,
            previous_response_id=None,
            prompt=None,
            prompt_cache_key=None,
            reasoning=Reasoning(effort=None, generate_summary=None, summary=None),
            safety_identifier=None,
            service_tier="auto",
            status="in_progress",
            text=ResponseTextConfig(
                format=ResponseFormatText(type="text"), verbosity="medium"
            ),
            top_logprobs=0,
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
            created_at=1757346128.0,
            error=None,
            incomplete_details=None,
            instructions=None,
            metadata={},
            model="gpt-4.1-mini-2025-04-14",
            object="response",
            output=[],
            parallel_tool_calls=True,
            temperature=1.0,
            tool_choice="auto",
            tools=[
                FileSearchTool(
                    type="file_search",
                    vector_store_ids=["vs_123"],
                    filters=None,
                    max_num_results=20,
                ),
                FunctionTool(
                    name="grade",
                    parameters={
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "type": "object",
                        "additionalProperties": False,
                    },
                    strict=True,
                    type="function",
                    description="Grade a given answer.",
                ),
            ],
            top_p=1.0,
            background=False,
            conversation=None,
            max_output_tokens=None,
            max_tool_calls=None,
            previous_response_id=None,
            prompt=None,
            prompt_cache_key=None,
            reasoning=Reasoning(effort=None, generate_summary=None, summary=None),
            safety_identifier=None,
            service_tier="auto",
            status="in_progress",
            text=ResponseTextConfig(
                format=ResponseFormatText(type="text"), verbosity="medium"
            ),
            top_logprobs=0,
            truncation="disabled",
            usage=None,
            user=None,
        ),
        sequence_number=1,
        type="response.in_progress",
    ),
    ResponseOutputItemAddedEvent(
        item=ResponseFileSearchToolCall(
            id="fs_123",
            queries=[],
            status="in_progress",
            type="file_search_call",
            results=None,
        ),
        output_index=0,
        sequence_number=2,
        type="response.output_item.added",
    ),
    ResponseFileSearchCallInProgressEvent(
        item_id="fs_123",
        output_index=0,
        sequence_number=3,
        type="response.file_search_call.in_progress",
    ),
    ResponseFileSearchCallSearchingEvent(
        item_id="fs_123",
        output_index=0,
        sequence_number=4,
        type="response.file_search_call.searching",
    ),
    ResponseFileSearchCallCompletedEvent(
        item_id="fs_123",
        output_index=0,
        sequence_number=5,
        type="response.file_search_call.completed",
    ),
    ResponseOutputItemDoneEvent(
        item=ResponseFileSearchToolCall(
            id="fs_123",
            queries=["query"],
            status="completed",
            type="file_search_call",
            results=None,
        ),
        output_index=0,
        sequence_number=6,
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
        sequence_number=7,
        type="response.output_item.added",
    ),
    ResponseContentPartAddedEvent(
        content_index=0,
        item_id="msg_123",
        output_index=1,
        part=ResponseOutputText(
            annotations=[], text="", type="output_text", logprobs=[]
        ),
        sequence_number=8,
        type="response.content_part.added",
    ),
    ResponseTextDeltaEvent(
        content_index=0,
        delta="The answer",
        item_id="msg_123",
        logprobs=[],
        output_index=1,
        sequence_number=9,
        type="response.output_text.delta",
    ),
    ResponseTextDeltaEvent(
        content_index=0,
        delta=" is...",
        item_id="msg_123",
        logprobs=[],
        output_index=1,
        sequence_number=10,
        type="response.output_text.delta",
    ),
    ResponseOutputTextAnnotationAddedEvent(
        annotation={
            "type": "file_citation",
            "file_id": "file-123",
            "filename": "my_file.pdf",
            "index": 1430,
        },
        annotation_index=0,
        content_index=0,
        item_id="msg_123",
        output_index=1,
        sequence_number=253,
        type="response.output_text.annotation.added",
    ),
    ResponseTextDeltaEvent(
        content_index=0,
        delta=".",
        item_id="msg_123",
        logprobs=[],
        output_index=1,
        sequence_number=254,
        type="response.output_text.delta",
    ),
    ResponseTextDoneEvent(
        content_index=0,
        item_id="msg_123",
        logprobs=[],
        output_index=1,
        sequence_number=255,
        text="The answer is...",
        type="response.output_text.done",
    ),
    ResponseContentPartDoneEvent(
        content_index=0,
        item_id="msg_123",
        output_index=1,
        part=ResponseOutputText(
            annotations=[
                AnnotationFileCitation(
                    file_id="file-123",
                    filename="my_file.pdf",
                    index=1430,
                    type="file_citation",
                )
            ],
            text="The answer is...",
            type="output_text",
            logprobs=[],
        ),
        sequence_number=256,
        type="response.content_part.done",
    ),
    ResponseOutputItemDoneEvent(
        item=ResponseOutputMessage(
            id="msg_123",
            content=[
                ResponseOutputText(
                    annotations=[
                        AnnotationFileCitation(
                            file_id="file-123",
                            filename="my_file.pdf",
                            index=1430,
                            type="file_citation",
                        )
                    ],
                    text="The answer is...",
                    type="output_text",
                )
            ],
            role="assistant",
            status="completed",
            type="message",
        ),
        output_index=1,
        sequence_number=257,
        type="response.output_item.done",
    ),
    ResponseOutputItemAddedEvent(
        item=ResponseFunctionToolCall(
            arguments="",
            call_id="call_123",
            name="grade",
            type="function_call",
            id="fc_123",
            status="in_progress",
        ),
        output_index=2,
        sequence_number=258,
        type="response.output_item.added",
    ),
    ResponseFunctionCallArgumentsDeltaEvent(
        delta='{"',
        item_id="fc_123",
        output_index=2,
        sequence_number=259,
        type="response.function_call_arguments.delta",
    ),
    ResponseFunctionCallArgumentsDeltaEvent(
        delta="answer",
        item_id="fc_123",
        output_index=2,
        sequence_number=260,
        type="response.function_call_arguments.delta",
    ),
    ResponseFunctionCallArgumentsDeltaEvent(
        delta='":"',
        item_id="fc_123",
        output_index=2,
        sequence_number=261,
        type="response.function_call_arguments.delta",
    ),
    ResponseFunctionCallArgumentsDeltaEvent(
        delta='answer"',
        item_id="fc_123",
        output_index=2,
        sequence_number=512,
        type="response.function_call_arguments.delta",
    ),
    ResponseFunctionCallArgumentsDeltaEvent(
        delta="}",
        item_id="fc_123",
        output_index=2,
        sequence_number=514,
        type="response.function_call_arguments.delta",
    ),
    ResponseFunctionCallArgumentsDoneEvent(
        arguments='{"answer":"answer"}',
        item_id="fc_123",
        output_index=2,
        sequence_number=515,
        type="response.function_call_arguments.done",
    ),
    ResponseOutputItemDoneEvent(
        item=ResponseFunctionToolCall(
            arguments='{"answer":"answer"}',
            call_id="call_123",
            name="grade",
            type="function_call",
            id="fc_123",
            status="completed",
        ),
        output_index=2,
        sequence_number=516,
        type="response.output_item.done",
    ),
    ResponseCompletedEvent(
        response=Response(
            id="resp_123",
            created_at=1757346128.0,
            error=None,
            incomplete_details=None,
            instructions=None,
            metadata={},
            model="gpt-4.1-mini-2025-04-14",
            object="response",
            output=[
                ResponseFileSearchToolCall(
                    id="fs_123",
                    queries=["query"],
                    status="completed",
                    type="file_search_call",
                    results=None,
                ),
                ResponseOutputMessage(
                    id="msg_123",
                    content=[
                        ResponseOutputText(
                            annotations=[
                                AnnotationFileCitation(
                                    file_id="file-123",
                                    filename="my_file.pdf",
                                    index=1430,
                                    type="file_citation",
                                )
                            ],
                            text="The answer is....",
                            type="output_text",
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                ),
                ResponseFunctionToolCall(
                    arguments='{"answer":"answer"}',
                    call_id="call_123",
                    name="grade",
                    type="function_call",
                    id="fc_123",
                    status="completed",
                ),
            ],
            parallel_tool_calls=True,
            temperature=1.0,
            tool_choice="auto",
            tools=[
                FileSearchTool(
                    type="file_search",
                    vector_store_ids=["vs_123"],
                    filters=None,
                    max_num_results=20,
                ),
                FunctionTool(
                    name="grade",
                    parameters={
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "type": "object",
                        "additionalProperties": False,
                    },
                    strict=True,
                    type="function",
                    description="Grade a given answer.",
                ),
            ],
            top_p=1.0,
            background=False,
            conversation=None,
            max_output_tokens=None,
            max_tool_calls=None,
            previous_response_id=None,
            prompt=None,
            prompt_cache_key=None,
            reasoning=Reasoning(effort=None, generate_summary=None, summary=None),
            safety_identifier=None,
            service_tier="default",
            status="completed",
            text=ResponseTextConfig(
                format=ResponseFormatText(type="text"), verbosity="medium"
            ),
            top_logprobs=0,
            truncation="disabled",
            usage=ResponseUsage(
                input_tokens=17052,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens=545,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=17597,
            ),
            user=None,
        ),
        sequence_number=517,
        type="response.completed",
    ),
]


def _strip_none(obj: Any) -> Any:
    """Recursively strip None values from dictionaries and lists."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [_strip_none(v) for v in obj]
    else:
        return obj


def test_responses_stream_annotations_fc() -> None:
    llm = ChatOpenAI(model="gpt-4.1-mini", output_version="responses/v1")
    mock_client = MagicMock()

    def mock_create(*args: Any, **kwargs: Any) -> MockSyncContextManager:
        return MockSyncContextManager(responses_stream)

    mock_client.responses.create = mock_create

    full: Optional[BaseMessageChunk] = None
    with patch.object(llm, "root_client", mock_client):
        for chunk in llm.stream("test"):
            assert isinstance(chunk, AIMessageChunk)
            full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)

    expected_content = [
        {
            "id": "fs_123",
            "queries": ["query"],
            "status": "completed",
            "type": "file_search_call",
            "index": 0,
        },
        {
            "type": "text",
            "text": "The answer is....",
            "index": 1,
            "annotations": [
                {
                    "type": "file_citation",
                    "file_id": "file-123",
                    "filename": "my_file.pdf",
                    "index": 1430,
                }
            ],
            "id": "msg_123",
        },
        {
            "type": "function_call",
            "name": "grade",
            "arguments": '{"answer":"answer"}',
            "call_id": "call_123",
            "id": "fc_123",
            "index": 2,
        },
    ]
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
        _ = payload["input"][idx].pop("status", None)
        _ = dumped.pop("logprobs", None)
        assert dumped == payload["input"][idx]
