"""Refusal metadata added after provider generation."""

from collections.abc import Iterator
from typing import Any

import pytest
from typing_extensions import override

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class RefusalModel(GenericFakeChatModel):
    metadata_source: str

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        metadata = {"stop_reason": "refusal"}
        generation = ChatGeneration(
            message=AIMessage("blocked"),
            generation_info=metadata if self.metadata_source == "generation" else None,
        )
        return ChatResult(
            generations=[generation],
            llm_output=metadata if self.metadata_source == "output" else None,
        )

    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content="partial"))
        yield ChatGenerationChunk(
            message=AIMessageChunk(content=""),
            generation_info={"stop_reason": "refusal"},
        )


@pytest.mark.parametrize("source", ["generation", "output"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_late_metadata(source: str, *, asynchronous: bool) -> None:
    model = RefusalModel(messages=iter([]), metadata_source=source)
    message = (
        await model.ainvoke("request") if asynchronous else model.invoke("request")
    )
    assert message.response_metadata["refusal"] is True
    assert message.response_metadata["stop_reason"] == "refusal"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_stream_metadata(*, asynchronous: bool) -> None:
    model = RefusalModel(messages=iter([]), metadata_source="generation")
    chunks = (
        [chunk async for chunk in model.astream("request")]
        if asynchronous
        else list(model.stream("request"))
    )
    assert any(chunk.response_metadata.get("refusal") is True for chunk in chunks)
    combined = chunks[0]
    for chunk in chunks[1:]:
        combined += chunk
    assert combined.response_metadata["refusal"] is True
    assert combined.content == "partial"
