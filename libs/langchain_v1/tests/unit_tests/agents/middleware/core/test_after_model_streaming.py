"""Regression tests for streaming with `after_model` middleware."""

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from langchain.agents import create_agent
from langchain.agents.middleware.pii import PIIDetectionError, PIIMiddleware
from langchain.agents.middleware.types import AgentMiddleware, AgentState

EMAIL = "john.doe@example.com"
RAW_RESPONSE = f"Contact me at {EMAIL}"
REDACTED_RESPONSE = "Contact me at [REDACTED_EMAIL]"
SAFE_RESPONSE = "All clear and safe."


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class NoOpAfterModelMiddleware(AgentMiddleware[AgentState[Any], None, Any]):
    """Middleware that keeps model output unchanged."""

    def after_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[None],
    ) -> dict[str, Any] | None:
        return None

    async def aafter_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[None],
    ) -> dict[str, Any] | None:
        return None


class PrefixAfterModelMiddleware(AgentMiddleware[AgentState[Any], None, Any]):
    """Middleware that rewrites the last AI message content."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def after_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[None],
    ) -> dict[str, Any] | None:
        messages = list(state["messages"])
        last_message = messages[-1]
        messages[-1] = last_message.model_copy(
            update={"content": f"{self.prefix}{last_message.content}"}
        )
        return {"messages": messages}

    async def aafter_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[None],
    ) -> dict[str, Any] | None:
        return self.after_model(state, runtime)


def _collect_stream_contents(agent: Any) -> list[str]:
    contents = []
    for chunk, _metadata in agent.stream(
        {"messages": [HumanMessage(content="say hi")]},
        stream_mode="messages",
    ):
        content = getattr(chunk, "content", None)
        if isinstance(content, str) and content:
            contents.append(content)
    return contents


async def _acollect_stream_contents(agent: Any) -> list[str]:
    contents = []
    async for chunk, _metadata in agent.astream(
        {"messages": [HumanMessage(content="say hi")]},
        stream_mode="messages",
    ):
        content = getattr(chunk, "content", None)
        if isinstance(content, str) and content:
            contents.append(content)
    return contents


def test_after_model_blocking_hides_streamed_tokens() -> None:
    """Blocked output should raise before any raw streamed content leaks."""
    agent = create_agent(
        model=FakeListChatModel(responses=[RAW_RESPONSE]),
        middleware=[
            PIIMiddleware(
                "email",
                strategy="block",
                apply_to_input=False,
                apply_to_output=True,
            )
        ],
    )

    seen = []
    with pytest.raises(PIIDetectionError):
        for chunk, _metadata in agent.stream(
            {"messages": [HumanMessage(content="say hi")]},
            stream_mode="messages",
        ):
            content = getattr(chunk, "content", None)
            if isinstance(content, str) and content:
                seen.append(content)

    assert seen == []


@pytest.mark.anyio
async def test_after_model_blocking_hides_streamed_tokens_async() -> None:
    """Async blocked output should raise before any raw streamed content leaks."""
    agent = create_agent(
        model=FakeListChatModel(responses=[RAW_RESPONSE]),
        middleware=[
            PIIMiddleware(
                "email",
                strategy="block",
                apply_to_input=False,
                apply_to_output=True,
            )
        ],
    )

    seen = []
    with pytest.raises(PIIDetectionError):
        async for chunk, _metadata in agent.astream(
            {"messages": [HumanMessage(content="say hi")]},
            stream_mode="messages",
        ):
            content = getattr(chunk, "content", None)
            if isinstance(content, str) and content:
                seen.append(content)

    assert seen == []


def test_after_model_redaction_streams_only_sanitized_content() -> None:
    """Redacted output should be visible in stream without raw PII."""
    agent = create_agent(
        model=FakeListChatModel(responses=[RAW_RESPONSE]),
        middleware=[
            PIIMiddleware(
                "email",
                strategy="redact",
                apply_to_input=False,
                apply_to_output=True,
            )
        ],
    )

    seen = _collect_stream_contents(agent)
    joined = "".join(seen)

    assert joined == REDACTED_RESPONSE
    assert EMAIL not in joined


@pytest.mark.anyio
async def test_after_model_redaction_streams_only_sanitized_content_async() -> None:
    """Async redacted output should be visible in stream without raw PII."""
    agent = create_agent(
        model=FakeListChatModel(responses=[RAW_RESPONSE]),
        middleware=[
            PIIMiddleware(
                "email",
                strategy="redact",
                apply_to_input=False,
                apply_to_output=True,
            )
        ],
    )

    seen = await _acollect_stream_contents(agent)
    joined = "".join(seen)

    assert joined == REDACTED_RESPONSE
    assert EMAIL not in joined


def test_after_model_noop_still_streams_model_output() -> None:
    """A no-op `after_model` hook should still allow safe streaming."""
    agent = create_agent(
        model=FakeListChatModel(responses=[SAFE_RESPONSE]),
        middleware=[NoOpAfterModelMiddleware()],
    )

    seen = _collect_stream_contents(agent)

    assert "".join(seen) == SAFE_RESPONSE


@pytest.mark.anyio
async def test_after_model_noop_still_streams_model_output_async() -> None:
    """Async no-op `after_model` hook should still allow safe streaming."""
    agent = create_agent(
        model=FakeListChatModel(responses=[SAFE_RESPONSE]),
        middleware=[NoOpAfterModelMiddleware()],
    )

    seen = await _acollect_stream_contents(agent)

    assert "".join(seen) == SAFE_RESPONSE


def test_after_model_chain_commits_buffered_messages_from_noop_terminal_node() -> None:
    """A terminal no-op middleware should commit sanitized buffered output."""
    agent = create_agent(
        model=FakeListChatModel(responses=[RAW_RESPONSE]),
        middleware=[
            NoOpAfterModelMiddleware(),
            PIIMiddleware(
                "email",
                strategy="redact",
                apply_to_input=False,
                apply_to_output=True,
            ),
        ],
    )

    seen = _collect_stream_contents(agent)
    joined = "".join(seen)

    assert joined == REDACTED_RESPONSE
    assert EMAIL not in joined


@pytest.mark.anyio
async def test_after_model_chain_commits_buffered_messages_from_noop_terminal_node_async() -> None:
    """Async terminal no-op middleware should commit sanitized buffered output."""
    agent = create_agent(
        model=FakeListChatModel(responses=[RAW_RESPONSE]),
        middleware=[
            NoOpAfterModelMiddleware(),
            PIIMiddleware(
                "email",
                strategy="redact",
                apply_to_input=False,
                apply_to_output=True,
            ),
        ],
    )

    seen = await _acollect_stream_contents(agent)
    joined = "".join(seen)

    assert joined == REDACTED_RESPONSE
    assert EMAIL not in joined


def test_after_model_chain_streams_final_rewritten_content() -> None:
    """Only the terminal post-middleware message should be streamed."""
    agent = create_agent(
        model=FakeListChatModel(responses=[SAFE_RESPONSE]),
        middleware=[
            NoOpAfterModelMiddleware(),
            PrefixAfterModelMiddleware(prefix="Reviewed: "),
        ],
    )

    seen = _collect_stream_contents(agent)

    assert "".join(seen) == "Reviewed: All clear and safe."


@pytest.mark.anyio
async def test_after_model_chain_streams_final_rewritten_content_async() -> None:
    """Async streaming should surface only the final post-middleware message."""
    agent = create_agent(
        model=FakeListChatModel(responses=[SAFE_RESPONSE]),
        middleware=[
            NoOpAfterModelMiddleware(),
            PrefixAfterModelMiddleware(prefix="Reviewed: "),
        ],
    )

    seen = await _acollect_stream_contents(agent)

    assert "".join(seen) == "Reviewed: All clear and safe."
