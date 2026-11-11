"""Replay standardized refusals through OpenAI APIs."""

from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_openai.chat_models._compat import _convert_from_v1_to_chat_completions
from langchain_openai.chat_models.base import (
    OpenAIRefusalError,
    _construct_responses_api_input,
    _oai_structured_outputs_parser,
)


@pytest.mark.parametrize("native", [False, True])
def test_refusal_responses_replay(*, native: bool) -> None:
    message = AIMessage(
        content=[
            {
                "type": "refusal",
                "refusal": "blocked",
                "id": "msg_1",
                **(
                    {"phase": "final_answer"}
                    if native
                    else {"extras": {"phase": "final_answer"}}
                ),
            }
        ]
    )
    result = _construct_responses_api_input([message])
    assert result == [
        {
            "type": "message",
            "role": "assistant",
            "id": "msg_1",
            "phase": "final_answer",
            "content": [{"type": "refusal", "refusal": "blocked"}],
        }
    ]


def test_refusal_chat_completions_replay() -> None:
    message = AIMessage(
        content_blocks=[
            {
                "type": "refusal",
                "refusal": "blocked",
                "extras": {"phase": "final_answer"},
            }
        ]
    )
    assert _convert_from_v1_to_chat_completions(message).content == [
        {"type": "refusal", "refusal": "blocked"}
    ]


ANTHROPIC_REFUSAL_HISTORY = AIMessage(
    content=[
        {"type": "text", "text": "Here is"},
        {"type": "refusal", "refusal": "", "index": "lc_refusal"},
    ],
    response_metadata={"model_provider": "anthropic", "output_version": "v1"},
)


class _Answer(BaseModel):
    answer: str


@pytest.mark.parametrize("use_responses_api", [False, True])
def test_empty_refusal_from_other_provider_not_sent(*, use_responses_api: bool) -> None:
    llm = ChatOpenAI(model="gpt-5", use_responses_api=use_responses_api)
    payload = llm._get_request_payload(
        [HumanMessage("hi"), ANTHROPIC_REFUSAL_HISTORY, HumanMessage("next")]
    )
    sent = payload["input"] if use_responses_api else payload["messages"]
    assert "refusal" not in str(sent)


def test_empty_refusal_raises_openai_refusal_error() -> None:
    message = AIMessage(
        content=[{"type": "refusal", "refusal": "", "id": "msg_1"}],
        response_metadata={"model_provider": "openai"},
    )
    with pytest.raises(OpenAIRefusalError) as exc_info:
        _oai_structured_outputs_parser(message, _Answer)
    assert str(exc_info.value) == "Model refused to produce the requested output"


def _refusal_chunks() -> list[ChatGenerationChunk]:
    return [
        ChatGenerationChunk(
            message=AIMessageChunk(
                content=[{"type": "refusal", "refusal": "No.", "index": 0}],
                response_metadata={"model_provider": "openai"},
            )
        )
    ]


def test_streamed_responses_refusal_raises() -> None:
    llm = ChatOpenAI(model="gpt-5", use_responses_api=True)
    structured = llm.with_structured_output(_Answer, method="json_schema")
    with (
        patch.object(ChatOpenAI, "_stream", return_value=iter(_refusal_chunks())),
        pytest.raises(OpenAIRefusalError) as exc_info,
    ):
        list(structured.stream("hi"))
    assert str(exc_info.value) == "No."


async def test_streamed_responses_refusal_raises_async() -> None:
    llm = ChatOpenAI(model="gpt-5", use_responses_api=True)
    structured = llm.with_structured_output(_Answer, method="json_schema")

    async def _astream(*_: Any, **__: Any) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in _refusal_chunks():
            yield chunk

    with (
        patch.object(ChatOpenAI, "_astream", _astream),
        pytest.raises(OpenAIRefusalError),
    ):
        async for _ in structured.astream("hi"):
            pass


def _refusal_delta(refusal: str, *, first: bool = False) -> dict[str, Any]:
    delta: dict[str, Any] = {"refusal": refusal}
    if first:
        delta.update(role="assistant", content="")
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-5",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


STREAMED_REFUSAL = [
    _refusal_delta("I'm sorry, ", first=True),
    _refusal_delta("I cannot comply."),
]


class _BetaStream:
    """`beta.chat.completions.stream`: chunk events, then the final completion."""

    def __enter__(self) -> "_BetaStream":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter({"type": "chunk", "chunk": chunk} for chunk in STREAMED_REFUSAL)

    def get_final_completion(self) -> ChatCompletion:
        return ChatCompletion.model_validate(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-5",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "refusal": "I'm sorry, I cannot comply.",
                        },
                    }
                ],
            }
        )


@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_streamed_chat_completions_refusal_raises(output_version: str) -> None:
    llm = ChatOpenAI(model="gpt-5", output_version=output_version)
    structured = llm.with_structured_output(_Answer, method="json_schema")
    with patch.object(llm, "root_client") as root_client:
        root_client.beta.chat.completions.stream.return_value = _BetaStream()
        with pytest.raises(OpenAIRefusalError) as exc_info:
            list(structured.stream("hi"))
    assert str(exc_info.value) == "I'm sorry, I cannot comply."
