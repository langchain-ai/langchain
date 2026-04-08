"""Integration tests for ChatSarvam.

These tests make real network calls to the Sarvam API.
Set ``SARVAM_API_KEY`` in your environment before running.

Run with:
    pytest tests/integration_tests/
"""

import os

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_sarvam.chat_models import ChatSarvam

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")

requires_api_key = pytest.mark.skipif(
    not SARVAM_API_KEY,
    reason="SARVAM_API_KEY environment variable not set",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def model() -> ChatSarvam:
    return ChatSarvam(model="sarvam-m", temperature=0.0)


@pytest.fixture()
def model_30b() -> ChatSarvam:
    return ChatSarvam(model="sarvam-30b", temperature=0.0)


# ---------------------------------------------------------------------------
# Basic invocation
# ---------------------------------------------------------------------------


@requires_api_key
class TestBasicInvoke:
    def test_invoke_returns_ai_message(self, model: ChatSarvam) -> None:
        from langchain_core.messages import AIMessage

        result = model.invoke([HumanMessage(content="Say 'hello' and nothing else.")])
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, str)
        assert len(result.content) > 0

    def test_invoke_with_system_message(self, model: ChatSarvam) -> None:
        messages = [
            SystemMessage(content="Reply only in Hindi."),
            HumanMessage(content="What is 2 + 2?"),
        ]
        result = model.invoke(messages)
        assert len(result.content) > 0

    def test_invoke_usage_metadata(self, model: ChatSarvam) -> None:
        result = model.invoke([HumanMessage(content="Hi")])
        assert result.usage_metadata is not None
        assert result.usage_metadata["input_tokens"] > 0
        assert result.usage_metadata["output_tokens"] > 0

    def test_response_metadata(self, model: ChatSarvam) -> None:
        result = model.invoke([HumanMessage(content="Hi")])
        assert result.response_metadata.get("finish_reason") == "stop"

    def test_model_30b_works(self, model_30b: ChatSarvam) -> None:
        result = model_30b.invoke(
            [HumanMessage(content="Name the capital of France in one word.")]
        )
        assert len(result.content) > 0


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@requires_api_key
class TestStreaming:
    def test_stream_yields_chunks(self, model: ChatSarvam) -> None:
        from langchain_core.messages import AIMessageChunk

        chunks = list(model.stream([HumanMessage(content="Count to 3.")]))
        assert len(chunks) > 0
        assert all(isinstance(c, AIMessageChunk) for c in chunks)

    def test_stream_content_non_empty(self, model: ChatSarvam) -> None:
        chunks = list(model.stream([HumanMessage(content="Say hello.")]))
        combined = "".join(c.content for c in chunks)
        assert len(combined) > 0

    def test_streaming_flag(self) -> None:
        m = ChatSarvam(model="sarvam-m", streaming=True)
        chunks = list(m.stream([HumanMessage(content="Hi")]))
        combined = "".join(c.content for c in chunks)
        assert len(combined) > 0


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


@requires_api_key
class TestAsync:
    @pytest.mark.asyncio
    async def test_ainvoke_returns_content(self, model: ChatSarvam) -> None:
        result = await model.ainvoke([HumanMessage(content="Name one planet.")])
        assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_astream_yields_chunks(self, model: ChatSarvam) -> None:
        chunks = []
        async for chunk in model.astream([HumanMessage(content="Count to 3.")]):
            chunks.append(chunk)
        assert len(chunks) > 0
        combined = "".join(c.content for c in chunks)
        assert len(combined) > 0


# ---------------------------------------------------------------------------
# Reasoning effort
# ---------------------------------------------------------------------------


@requires_api_key
class TestReasoningEffort:
    def test_low_reasoning_effort(self) -> None:
        m = ChatSarvam(model="sarvam-m", reasoning_effort="low")
        result = m.invoke([HumanMessage(content="What is 5 * 5?")])
        assert len(result.content) > 0

    def test_high_reasoning_effort(self) -> None:
        m = ChatSarvam(model="sarvam-105b", reasoning_effort="high")
        result = m.invoke(
            [HumanMessage(content="Explain the Pythagorean theorem briefly.")]
        )
        assert len(result.content) > 0


# ---------------------------------------------------------------------------
# Stop sequences
# ---------------------------------------------------------------------------


@requires_api_key
class TestStopSequences:
    def test_stop_sequence_respected(self, model: ChatSarvam) -> None:
        result = model.invoke(
            [HumanMessage(content="Write the numbers 1 2 3 4 5")],
            stop=["3"],
        )
        assert "4" not in result.content
        assert "5" not in result.content


# ---------------------------------------------------------------------------
# Indian language support
# ---------------------------------------------------------------------------


@requires_api_key
class TestIndicLanguages:
    def test_hindi_response(self, model: ChatSarvam) -> None:
        messages = [
            SystemMessage(content="Respond only in Hindi."),
            HumanMessage(content="भारत की राजधानी क्या है?"),
        ]
        result = model.invoke(messages)
        assert len(result.content) > 0

    def test_mixed_language_input(self, model: ChatSarvam) -> None:
        result = model.invoke(
            [HumanMessage(content="Translate 'Good morning' to Hindi.")]
        )
        assert len(result.content) > 0


# ---------------------------------------------------------------------------
# Compile marker (runs without API key — just imports)
# ---------------------------------------------------------------------------


@pytest.mark.compile
def test_compile() -> None:
    """Placeholder to compile integration tests without running them."""
    assert True
