"""V1 parity tests: `stream_events(version="v3")` must match `model.stream()` output.

These are the acceptance criteria for the v3 streaming API — if any test fails,
v3 has a regression vs v1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from typing_extensions import override

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
    from langchain_core.messages import BaseMessage


class _ScriptedChunkModel(BaseChatModel):
    """Fake chat model that streams a fixed, pre-built sequence of chunks.

    Lets us write parity tests that exercise tool calls, reasoning,
    usage metadata, and response metadata — shapes `FakeListChatModel`
    cannot produce.
    """

    scripted_chunks: list[AIMessageChunk]
    raise_after: bool = False
    """If True, raise `_FakeStreamError` after yielding all scripted chunks."""

    @property
    @override
    def _llm_type(self) -> str:
        return "scripted-chunk-fake"

    def _merged(self) -> AIMessageChunk:
        merged = self.scripted_chunks[0]
        for c in self.scripted_chunks[1:]:
            merged = merged + c
        return merged

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        merged = self._merged()
        final = AIMessage(
            content=merged.content,
            id=merged.id,
            tool_calls=merged.tool_calls,
            usage_metadata=merged.usage_metadata,
            response_metadata=merged.response_metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=final)])

    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in self.scripted_chunks:
            yield ChatGenerationChunk(message=chunk)
        if self.raise_after:
            msg = "scripted failure"
            raise _FakeStreamError(msg)

    @override
    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self.scripted_chunks:
            yield ChatGenerationChunk(message=chunk)
        if self.raise_after:
            msg = "scripted failure"
            raise _FakeStreamError(msg)


class _FakeStreamError(RuntimeError):
    """Marker exception raised by `_ScriptedChunkModel` during streaming."""


def _collect_v1_message(model: BaseChatModel, input_text: str) -> AIMessage:
    """Run model.stream() (in v1 output mode) and merge chunks into an AIMessage.

    `ChatModelStream.output` is always v1-shaped (content is a list of
    protocol blocks when blocks arrived). The legacy stream path only
    emits v1-shaped content when `output_version="v1"` is set on the
    model, so force it here for a like-for-like parity comparison.
    """
    model.output_version = "v1"
    chunks: list[AIMessageChunk] = [
        chunk for chunk in model.stream(input_text) if isinstance(chunk, AIMessageChunk)
    ]
    if not chunks:
        msg = "No chunks produced"
        raise RuntimeError(msg)
    merged = chunks[0]
    for c in chunks[1:]:
        merged = merged + c
    return AIMessage(
        content=merged.content,
        id=merged.id,
        tool_calls=merged.tool_calls,
        usage_metadata=merged.usage_metadata,
        response_metadata=merged.response_metadata,
    )


def _collect_v2_message(model: BaseChatModel, input_text: str) -> AIMessage:
    """Run `model.stream_events(version="v3")` and get `.output`."""
    stream = model.stream_events(input_text, version="v3")
    return stream.output


class TestV1ParityBasic:
    """Smoke-level parity using the simple text-only fake."""

    def test_text_only_content_matches(self) -> None:
        model = FakeListChatModel(responses=["Hello world!"])
        v1 = _collect_v1_message(model, "test")
        model.i = 0
        v2 = _collect_v2_message(model, "test")

        assert v1.content == v2.content

    def test_message_id_present(self) -> None:
        model = FakeListChatModel(responses=["Hi"])
        v1 = _collect_v1_message(model, "test")
        model.i = 0
        v2 = _collect_v2_message(model, "test")

        assert v1.id is not None
        assert v2.id is not None

    def test_empty_response(self) -> None:
        """A truly empty stream is an error, matching `stream()` parity.

        `stream_events(version="v3")` distinguishes "producer emitted events but no
        terminal `message-finish`" (which is synthesized, for native-event providers
        that omit it) from "producer emitted nothing at all" (which fails
        with `ValueError`, same as `stream()`).
        """
        model = FakeListChatModel(responses=[""])
        stream = model.stream_events("test", version="v3")
        with pytest.raises(ValueError, match="No generation chunks"):
            _ = stream.output

    def test_multi_character_response(self) -> None:
        text = "The quick brown fox"
        model = FakeListChatModel(responses=[text])
        v2 = _collect_v2_message(model, "test")
        assert isinstance(v2.content, list)
        assert len(v2.content) == 1
        text_block = v2.content[0]
        assert isinstance(text_block, dict)
        assert text_block["type"] == "text"
        assert text_block["text"] == text

    def test_text_deltas_reconstruct_content(self) -> None:
        model = FakeListChatModel(responses=["Hello!"])
        stream = model.stream_events("test", version="v3")

        deltas = list(stream.text)
        content = stream.output.content
        assert isinstance(content, list)
        first_block = content[0]
        assert isinstance(first_block, dict)
        assert "".join(deltas) == first_block["text"]


class TestV1ParityToolCalls:
    """Tool-call parity — the most load-bearing v1 shape."""

    @staticmethod
    def _make_model() -> _ScriptedChunkModel:
        chunks = [
            AIMessageChunk(
                content="",
                id="run-tool-1",
                tool_call_chunks=[
                    {"index": 0, "id": "call_1", "name": "get_weather", "args": ""},
                ],
            ),
            AIMessageChunk(
                content="",
                id="run-tool-1",
                tool_call_chunks=[
                    {"index": 0, "id": None, "name": None, "args": '{"city": "'},
                ],
            ),
            AIMessageChunk(
                content="",
                id="run-tool-1",
                tool_call_chunks=[
                    {"index": 0, "id": None, "name": None, "args": 'Paris"}'},
                ],
                response_metadata={"finish_reason": "tool_use"},
            ),
        ]
        return _ScriptedChunkModel(scripted_chunks=chunks)

    def test_tool_calls_match(self) -> None:
        model = self._make_model()
        v1 = _collect_v1_message(model, "weather?")
        v2 = _collect_v2_message(self._make_model(), "weather?")

        assert len(v1.tool_calls) == 1
        assert len(v2.tool_calls) == 1
        assert v1.tool_calls[0]["id"] == v2.tool_calls[0]["id"] == "call_1"
        assert v1.tool_calls[0]["name"] == v2.tool_calls[0]["name"] == "get_weather"
        assert v1.tool_calls[0]["args"] == v2.tool_calls[0]["args"] == {"city": "Paris"}

    def test_tool_calls_via_projection(self) -> None:
        model = self._make_model()
        stream = model.stream_events("weather?", version="v3")
        finalized = stream.tool_calls.get()
        assert len(finalized) == 1
        assert finalized[0]["name"] == "get_weather"
        assert finalized[0]["args"] == {"city": "Paris"}

    def test_finish_reason_tool_use(self) -> None:
        model = self._make_model()
        v2 = _collect_v2_message(model, "weather?")
        assert v2.response_metadata.get("finish_reason") == "tool_use"


class TestV1ParityUsage:
    """Usage metadata parity."""

    @staticmethod
    def _make_model() -> _ScriptedChunkModel:
        chunks = [
            AIMessageChunk(content="Hi", id="run-usage-1"),
            AIMessageChunk(
                content=" there",
                id="run-usage-1",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
                response_metadata={"finish_reason": "stop"},
            ),
        ]
        return _ScriptedChunkModel(scripted_chunks=chunks)

    def test_usage_metadata_present(self) -> None:
        v1 = _collect_v1_message(self._make_model(), "hello")
        v2 = _collect_v2_message(self._make_model(), "hello")

        assert v1.usage_metadata is not None
        assert v2.usage_metadata is not None
        assert v1.usage_metadata["input_tokens"] == v2.usage_metadata["input_tokens"]
        assert v1.usage_metadata["output_tokens"] == v2.usage_metadata["output_tokens"]
        assert v1.usage_metadata["total_tokens"] == v2.usage_metadata["total_tokens"]

    def test_usage_projection_matches(self) -> None:
        stream = self._make_model().stream_events("hello", version="v3")
        # Drain so usage is available
        for _ in stream.text:
            pass
        usage = stream.output.usage_metadata
        assert usage is not None
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 5


class TestV1ParityResponseMetadata:
    """Response metadata preservation (fix 5b)."""

    @staticmethod
    def _make_model() -> _ScriptedChunkModel:
        chunks = [
            AIMessageChunk(
                content="ok",
                id="run-meta-1",
                response_metadata={
                    "finish_reason": "stop",
                    "model_provider": "fake-provider",
                    "stop_sequence": None,
                },
            ),
        ]
        return _ScriptedChunkModel(scripted_chunks=chunks)

    def test_finish_reason_preserved(self) -> None:
        v2 = _collect_v2_message(self._make_model(), "hi")
        assert v2.response_metadata.get("finish_reason") == "stop"

    def test_provider_metadata_preserved(self) -> None:
        """Non-finish-reason keys should survive the round-trip."""
        v2 = _collect_v2_message(self._make_model(), "hi")
        # stop_sequence came from response_metadata on chunks; the bridge
        # should carry it through via MessageFinishData.metadata.
        assert "stop_sequence" in v2.response_metadata


class TestV1ParityReasoning:
    """Reasoning content parity — order must be preserved."""

    @staticmethod
    def _make_model() -> _ScriptedChunkModel:
        chunks = [
            AIMessageChunk(
                content=[
                    {"type": "reasoning", "reasoning": "Let me think. ", "index": 0},
                ],
                id="run-reason-1",
            ),
            AIMessageChunk(
                content=[
                    {"type": "reasoning", "reasoning": "Done.", "index": 0},
                ],
                id="run-reason-1",
            ),
            AIMessageChunk(
                content=[
                    {"type": "text", "text": "The answer is 42.", "index": 1},
                ],
                id="run-reason-1",
                response_metadata={"finish_reason": "stop"},
            ),
        ]
        return _ScriptedChunkModel(scripted_chunks=chunks)

    def test_reasoning_text_order(self) -> None:
        """Reasoning block should come before text block in .output.content."""
        v2 = _collect_v2_message(self._make_model(), "think")
        assert isinstance(v2.content, list)
        types_in_order = [b.get("type") for b in v2.content if isinstance(b, dict)]
        assert types_in_order == ["reasoning", "text"]

    def test_reasoning_projection(self) -> None:
        stream = self._make_model().stream_events("think", version="v3")
        full_reasoning = str(stream.reasoning)
        assert full_reasoning == "Let me think. Done."


class TestV1ParityError:
    """Errors during streaming must propagate on both paths."""

    def test_error_propagates_sync(self) -> None:
        chunks = [
            AIMessageChunk(content="partial", id="run-err-1"),
        ]
        model = _ScriptedChunkModel(scripted_chunks=chunks, raise_after=True)

        stream = model.stream_events("boom", version="v3")
        # Drain first; error may surface here or at .output access.
        try:
            list(stream.text)
        except _FakeStreamError:
            return  # Error surfaced during iteration — pass
        with pytest.raises(_FakeStreamError):
            _ = stream.output

    @pytest.mark.asyncio
    async def test_error_propagates_async(self) -> None:
        chunks = [
            AIMessageChunk(content="partial", id="run-err-2"),
        ]
        model = _ScriptedChunkModel(scripted_chunks=chunks, raise_after=True)

        stream = await model.astream_events("boom", version="v3")
        try:
            async for _ in stream.text:
                pass
        except _FakeStreamError:
            return
        with pytest.raises(_FakeStreamError):
            _ = await stream
