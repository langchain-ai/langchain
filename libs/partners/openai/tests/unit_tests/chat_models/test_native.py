from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableBinding
from openai.types.responses import Response
from openai.types.responses.compacted_response import CompactedResponse
from pydantic import BaseModel, SecretStr

from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import (
    _construct_lc_result_from_responses_api,
    _construct_responses_api_input,
    _convert_responses_chunk_to_generation_chunk,
)

PATCH = {
    "type": "apply_patch_call",
    "id": "ap_1",
    "call_id": "call_1",
    "status": "completed",
    "operation": {"type": "create_file", "path": "hello.txt", "diff": "+hello"},
}
SHELL = {
    "type": "shell_call",
    "id": "sh_1",
    "call_id": "call_1",
    "status": "completed",
    "action": {"commands": ["pwd"], "timeout_ms": 1000, "max_output_length": 200},
    "environment": {"type": "local"},
}
LOCAL_SHELL = {
    "type": "local_shell_call",
    "id": "lsh_1",
    "call_id": "call_1",
    "status": "completed",
    "action": {"type": "exec", "command": ["pwd"], "env": {}},
}


def response_for(block: dict[str, Any]) -> Response:
    return Response.model_validate(
        {
            "id": "resp_1",
            "created_at": 0,
            "model": "test-model",
            "object": "response",
            "parallel_tool_calls": True,
            "tools": [],
            "tool_choice": "auto",
            "output": [block],
        }
    )


@pytest.mark.parametrize("block", [PATCH, SHELL, LOCAL_SHELL])
@pytest.mark.parametrize("output_version", ["v0", "responses/v1"])
def test_native_calls_roundtrip(block: dict[str, Any], output_version: str) -> None:
    result = _construct_lc_result_from_responses_api(
        response_for(block), output_version=output_version
    )
    message = result.generations[0].message
    assert isinstance(message, AIMessage)
    key = "operation" if block["type"] == "apply_patch_call" else "action"
    assert message.tool_calls == [
        {
            "type": "tool_call",
            "name": block["type"].removesuffix("_call"),
            "args": {key: block[key]},
            "id": "call_1",
        }
    ]
    assert _construct_responses_api_input([message]) == [block]
    standard = message.model_copy(update={"content": message.content_blocks})
    assert _construct_responses_api_input([standard]) == [block]


@pytest.mark.parametrize("block", [PATCH, SHELL, LOCAL_SHELL])
def test_native_streaming(block: dict[str, Any]) -> None:
    item = response_for(block).output[0]
    _, _, _, added = _convert_responses_chunk_to_generation_chunk(
        SimpleNamespace(type="response.output_item.added", output_index=0, item=item),
        -1,
        -1,
        -1,
    )
    if added is not None:
        assert isinstance(added.message, AIMessageChunk)
        assert not added.message.tool_call_chunks
    _, _, _, done = _convert_responses_chunk_to_generation_chunk(
        SimpleNamespace(type="response.output_item.done", output_index=0, item=item),
        -1,
        -1,
        -1,
    )
    assert done is not None
    assert isinstance(done.message, AIMessageChunk)
    assert done.message.tool_calls[0]["id"] == "call_1"
    assert _construct_responses_api_input([done.message]) == [block]


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "incomplete"},
        {"status": "in_progress"},
        {"environment": {"type": "container_reference", "container_id": "cntr_1"}},
        {"caller": {"type": "program", "caller_id": "program_1"}},
    ],
)
def test_non_executable_calls_are_preserved(changes: dict[str, Any]) -> None:
    block = {**SHELL, **changes}
    message = (
        _construct_lc_result_from_responses_api(response_for(block))
        .generations[0]
        .message
    )
    assert isinstance(message, AIMessage)
    assert not message.tool_calls
    assert _construct_responses_api_input([message]) == [block]


@pytest.mark.parametrize(
    "output",
    [
        {"type": "apply_patch_call_output", "status": "failed", "output": "denied"},
        {
            "type": "shell_call_output",
            "output": [
                {
                    "stdout": "",
                    "stderr": "denied",
                    "outcome": {"type": "exit", "exit_code": 1},
                }
            ],
            "max_output_length": 200,
        },
        {"type": "local_shell_call_output", "output": '{"error":"denied"}'},
    ],
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_native_tool_output(output: dict[str, Any], wrapped: bool) -> None:
    content = {"type": "non_standard", "value": output} if wrapped else output
    message = ToolMessage(content=[content], tool_call_id="call_1")
    key = "id" if output["type"] == "local_shell_call_output" else "call_id"
    assert _construct_responses_api_input([message]) == [{**output, key: "call_1"}]
    assert key not in output


@pytest.mark.parametrize("wrapped", [False, True])
def test_reject_mismatched_native_call_id(wrapped: bool) -> None:
    output = {
        "type": "apply_patch_call_output",
        "call_id": "wrong_call",
        "status": "completed",
    }
    content = {"type": "non_standard", "value": output} if wrapped else output
    with pytest.raises(ValueError, match="identity"):
        _construct_responses_api_input(
            [ToolMessage(content=[content], tool_call_id="call_1")]
        )


def test_reject_ambiguous_native_output() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        _construct_responses_api_input(
            [
                ToolMessage(
                    content=[
                        {"type": "apply_patch_call_output", "status": "failed"},
                        "extra",
                    ],
                    tool_call_id="call_1",
                )
            ]
        )


def context_model() -> ChatOpenAI:
    return ChatOpenAI(
        model="test-model", api_key=SecretStr("test"), use_previous_response_id=True
    )


class Answer(BaseModel):
    answer: str


def test_count_input_tokens_matches_generation_context() -> None:
    model = context_model()
    count = MagicMock(return_value=SimpleNamespace(input_tokens=42))
    model.root_client.responses.input_tokens.count = count
    tools = cast(
        "RunnableBinding",
        model.bind_tools(
            [
                {"type": "apply_patch"},
                {"type": "shell", "environment": {"type": "local"}},
            ]
        ),
    ).kwargs["tools"]
    messages = [
        HumanMessage(content="hello"),
        AIMessage(
            content=[{"type": "reasoning", "id": "rs_1", "summary": []}],
            response_metadata={"id": "resp_old"},
        ),
        HumanMessage(content="world"),
    ]
    assert model.count_input_tokens(messages, tools=tools, response_format=Answer) == 42
    payload = count.call_args.kwargs
    assert payload["model"] == "test-model"
    assert payload["tools"] == tools
    assert payload["text"]["format"]["schema"]["required"] == ["answer"]
    assert payload["input"] == _construct_responses_api_input(messages, store=False)
    assert "previous_response_id" not in payload
    assert "store" not in payload
    assert "stream" not in payload


def compacted_result() -> CompactedResponse:
    return CompactedResponse.model_construct(
        id="cmp_1",
        output=[
            response_for(
                {
                    "type": "compaction",
                    "id": "cmp_item_1",
                    "encrypted_content": "opaque",
                }
            ).output[0]
        ],
    )


def test_compact_preserves_opaque_output_and_model() -> None:
    model = context_model()
    compact = MagicMock(return_value=compacted_result())
    model.root_client.responses.compact = compact
    message = model.compact(
        [HumanMessage(content="hello")],
        model="override-model",
        tools=[{"type": "apply_patch"}],
    )
    assert message.response_metadata["model_name"] == "override-model"
    assert message.content == [
        {"type": "compaction", "id": "cmp_item_1", "encrypted_content": "opaque"}
    ]
    assert _construct_responses_api_input([message], store=False) == message.content
    assert compact.call_args.kwargs == {
        "model": "override-model",
        "input": [{"role": "user", "content": "hello", "type": "message"}],
    }


def test_compacted_user_messages_replay_without_rewriting() -> None:
    item = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "hello"}],
    }
    opaque = {"type": "compaction", "encrypted_content": "opaque"}
    message = AIMessage(content=[item, opaque])
    assert _construct_responses_api_input([message], store=False) == [item, opaque]


def test_reject_native_output_without_status() -> None:
    with pytest.raises(ValueError, match="status"):
        _construct_responses_api_input(
            [
                ToolMessage(
                    content=[{"type": "apply_patch_call_output", "output": "denied"}],
                    tool_call_id="call_1",
                )
            ]
        )


async def test_async_context_endpoints() -> None:
    model = context_model()
    count = AsyncMock(return_value=SimpleNamespace(input_tokens=12))
    compact = AsyncMock(return_value=compacted_result())
    model.root_async_client.responses.input_tokens.count = count
    model.root_async_client.responses.compact = compact
    assert await model.acount_input_tokens("hello", response_format=Answer) == 12
    message = await model.acompact("hello")
    assert message.response_metadata["model_name"] == "test-model"
    assert _construct_responses_api_input([message], store=False) == message.content
    assert count.call_args.kwargs["input"] == compact.call_args.kwargs["input"]


@pytest.mark.parametrize("block", [PATCH, SHELL, LOCAL_SHELL])
@pytest.mark.parametrize("output_version", ["v0", "responses/v1"])
def test_native_replay_uses_approved_edits(
    block: dict[str, Any], output_version: str
) -> None:
    message = (
        _construct_lc_result_from_responses_api(
            response_for(block), output_version=output_version
        )
        .generations[0]
        .message
    )
    assert isinstance(message, AIMessage)
    key = "operation" if block["type"] == "apply_patch_call" else "action"
    edited = {
        **block[key],
        **({"path": "approved.txt"} if key == "operation" else {"timeout_ms": 500}),
    }
    message.tool_calls[0]["args"] = {key: edited}
    assert _construct_responses_api_input([message]) == [{**block, key: edited}]
    assert block[key] != edited


@pytest.mark.parametrize(
    "changes",
    [
        {"id": "wrong_call"},
        {"name": "different_tool"},
        {"args": {"command": "untyped"}},
    ],
)
def test_native_replay_rejects_identity_changes(changes: dict[str, Any]) -> None:
    message = (
        _construct_lc_result_from_responses_api(response_for(PATCH))
        .generations[0]
        .message
    )
    assert isinstance(message, AIMessage)
    cast("dict[str, Any]", message.tool_calls[0]).update(changes)
    with pytest.raises(ValueError, match=r"native|Native"):
        _construct_responses_api_input([message])


@pytest.mark.parametrize("block", [PATCH, SHELL, LOCAL_SHELL])
@pytest.mark.parametrize("previous", [False, True])
@pytest.mark.parametrize("status", ["error", "success"])
def test_native_hitl_rejection_resume(
    block: dict[str, Any], previous: bool, status: str
) -> None:
    message = (
        _construct_lc_result_from_responses_api(response_for(block))
        .generations[0]
        .message
    )
    model = ChatOpenAI(
        model="test-model",
        api_key=SecretStr("test"),
        use_responses_api=True,
        use_previous_response_id=previous,
    )
    rejection = ToolMessage(
        content="Human declined this operation",
        tool_call_id="call_1",
        status=cast("Any", status),
    )
    payload = model._get_request_payload([message, rejection], store=False)
    items = payload["input"]
    assert not any(item["type"] == "function_call_output" for item in items)
    assert payload.get("previous_response_id") == ("resp_1" if previous else None)
    if block["type"] == "apply_patch_call":
        assert items[-1] == {
            "type": "apply_patch_call_output",
            "call_id": "call_1",
            "status": "failed",
            "output": rejection.content,
        }
    elif block["type"] == "shell_call":
        assert items[-2] == {
            "type": "shell_call_output",
            "call_id": "call_1",
            "output": [],
            "status": "incomplete",
        }
        assert rejection.content in items[-1]["content"]
    else:
        assert items[-1] == {
            "type": "local_shell_call_output",
            "id": "call_1",
            "output": '{"error": "Human declined this operation"}',
        }
    assert rejection.content == "Human declined this operation"


def test_context_request_overrides_preserve_tools_schema_and_instructions() -> None:
    model = context_model()
    count = MagicMock(return_value=SimpleNamespace(input_tokens=1))
    model.root_client.responses.input_tokens.count = count
    tools = [{"type": "shell", "environment": {"type": "local"}}]
    text = {"format": {"type": "json_object"}}
    assert (
        model.count_input_tokens(
            "hello",
            instructions="system",
            extra_body={"instructions": "override", "tools": tools, "text": text},
        )
        == 1
    )
    payload = count.call_args.kwargs
    assert payload["instructions"] == "override"
    assert payload["tools"] == tools
    assert payload["text"] == text
    assert "extra_body" not in payload


def test_native_resume_rejects_wrong_output_type() -> None:
    message = (
        _construct_lc_result_from_responses_api(response_for(PATCH))
        .generations[0]
        .message
    )
    with pytest.raises(ValueError, match="output type"):
        _construct_responses_api_input(
            [
                message,
                ToolMessage(
                    content=[{"type": "shell_call_output", "output": []}],
                    tool_call_id="call_1",
                ),
            ]
        )
