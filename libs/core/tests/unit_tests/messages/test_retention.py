"""Tests for find_safe_message_cutoff and partition_messages.

These tests validate tool-boundary-safe message pruning utilities
proposed in https://github.com/langchain-ai/langchain/issues/38249.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    find_safe_message_cutoff,
    partition_messages,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def char_token_counter(messages: list) -> int:
    """Simple token counter that counts characters in message content."""
    return sum(len(msg.content) for msg in messages)


# ---------------------------------------------------------------------------
# find_safe_message_cutoff
# ---------------------------------------------------------------------------


class TestFindSafeMessageCutoff:
    """Tests for find_safe_message_cutoff."""

    def test_empty_messages(self) -> None:
        assert find_safe_message_cutoff([], max_tokens=10, token_counter=char_token_counter) == 0

    def test_retain_end_basic(self) -> None:
        """Keeps the last messages that fit within max_tokens."""
        messages = [
            HumanMessage(content="A" * 10),
            HumanMessage(content="B" * 20),
            HumanMessage(content="C" * 30),
        ]
        # limit=35 fits C (30) but not B+C (50)
        cutoff = find_safe_message_cutoff(
            messages, max_tokens=35, token_counter=char_token_counter, retain="end"
        )
        assert cutoff == 2

    def test_retain_start_basic(self) -> None:
        """Keeps the first messages that fit within max_tokens."""
        messages = [
            HumanMessage(content="A" * 10),
            HumanMessage(content="B" * 20),
            HumanMessage(content="C" * 30),
        ]
        # limit=35 fits A (10) + B (20) = 30, but not A+B+C (60)
        cutoff = find_safe_message_cutoff(
            messages, max_tokens=35, token_counter=char_token_counter, retain="start"
        )
        assert cutoff == 2

    def test_tool_boundary_retain_end(self) -> None:
        """AIMessage + ToolMessage are kept as an atomic block (retain=end)."""
        messages = [
            HumanMessage(content="H1"),
            AIMessage(
                content="AI1",
                tool_calls=[{"name": "t1", "args": {}, "id": "call1"}],
            ),
            ToolMessage(content="R1", tool_call_id="call1"),
            HumanMessage(content="H2"),
        ]

        def custom_counter(msgs: list) -> int:
            tokens = 0
            for m in msgs:
                if isinstance(m, ToolMessage) or (
                    isinstance(m, AIMessage) and m.tool_calls
                ):
                    tokens += 15  # AI+Tool block = 30 total
                else:
                    tokens += 10
            return tokens

        # limit=20: fits H2 (10) but not [AI1,R1] block (30)
        cutoff = find_safe_message_cutoff(
            messages, max_tokens=20, token_counter=custom_counter, retain="end"
        )
        assert cutoff == 3

        # limit=45: fits H2 (10) + [AI1,R1] (30) = 40, but not + H1 (10) = 50
        cutoff = find_safe_message_cutoff(
            messages, max_tokens=45, token_counter=custom_counter, retain="end"
        )
        assert cutoff == 1

    def test_tool_boundary_retain_start(self) -> None:
        """AIMessage + ToolMessage are kept as an atomic block (retain=start)."""
        messages = [
            HumanMessage(content="H1"),
            AIMessage(
                content="AI1",
                tool_calls=[{"name": "t1", "args": {}, "id": "call1"}],
            ),
            ToolMessage(content="R1", tool_call_id="call1"),
            HumanMessage(content="H2"),
        ]

        def custom_counter(msgs: list) -> int:
            tokens = 0
            for m in msgs:
                if isinstance(m, ToolMessage) or (
                    isinstance(m, AIMessage) and m.tool_calls
                ):
                    tokens += 15
                else:
                    tokens += 10
            return tokens

        # limit=25: fits H1 (10) but not + [AI1,R1] (30)
        cutoff = find_safe_message_cutoff(
            messages, max_tokens=25, token_counter=custom_counter, retain="start"
        )
        assert cutoff == 1

        # limit=45: fits H1 (10) + [AI1,R1] (30) = 40, but not + H2 (10) = 50
        cutoff = find_safe_message_cutoff(
            messages, max_tokens=45, token_counter=custom_counter, retain="start"
        )
        assert cutoff == 3

    def test_nothing_fits(self) -> None:
        """When no block fits, return appropriate edge values."""
        messages = [HumanMessage(content="A" * 100)]

        # retain=end, nothing fits -> cutoff = len(messages)
        cutoff = find_safe_message_cutoff(
            messages, max_tokens=5, token_counter=char_token_counter, retain="end"
        )
        assert cutoff == len(messages)

        # retain=start, nothing fits -> cutoff = 0
        cutoff = find_safe_message_cutoff(
            messages, max_tokens=5, token_counter=char_token_counter, retain="start"
        )
        assert cutoff == 0

    def test_everything_fits(self) -> None:
        """When all messages fit, retain everything."""
        messages = [
            HumanMessage(content="Hi"),
            AIMessage(content="Hello!"),
        ]

        # retain=end -> cutoff = 0 (all retained)
        cutoff = find_safe_message_cutoff(
            messages, max_tokens=1000, token_counter=char_token_counter, retain="end"
        )
        assert cutoff == 0

        # retain=start -> cutoff = len(messages) (all retained)
        cutoff = find_safe_message_cutoff(
            messages, max_tokens=1000, token_counter=char_token_counter, retain="start"
        )
        assert cutoff == len(messages)

    def test_invalid_retain_raises(self) -> None:
        """Invalid retain value raises ValueError."""
        with pytest.raises(ValueError, match="Invalid retain"):
            find_safe_message_cutoff(
                [HumanMessage(content="Hi")],
                max_tokens=10,
                token_counter=char_token_counter,
                retain="middle",  # type: ignore[arg-type]
            )

    def test_approximate_token_counter_default(self) -> None:
        """The 'approximate' shortcut works as default token counter."""
        messages = [
            HumanMessage(content="Hello world!"),
            AIMessage(content="This is a test message."),
        ]
        # Should not raise; uses count_tokens_approximately internally
        cutoff = find_safe_message_cutoff(messages, max_tokens=100)
        assert 0 <= cutoff <= len(messages)

    def test_multiple_tool_calls_in_sequence(self) -> None:
        """Multiple tool calls each form their own atomic block."""
        messages = [
            HumanMessage(content="Do two things"),
            AIMessage(
                content="",
                tool_calls=[{"name": "tool_a", "args": {}, "id": "a1"}],
            ),
            ToolMessage(content="result_a", tool_call_id="a1"),
            AIMessage(
                content="",
                tool_calls=[{"name": "tool_b", "args": {}, "id": "b1"}],
            ),
            ToolMessage(content="result_b", tool_call_id="b1"),
            HumanMessage(content="Thanks"),
        ]

        def fixed_counter(msgs: list) -> int:
            return len(msgs) * 10

        # limit=25: fits last 2 messages [ToolMessage, HumanMessage]?
        # No, blocks: H(10), [AI+T](20), [AI+T](20), H(10)
        # From end: H(10) ok, [AI+T](20) -> 30 > 25 -> stop
        cutoff = find_safe_message_cutoff(
            messages, max_tokens=25, token_counter=fixed_counter, retain="end"
        )
        assert cutoff == 5  # only last HumanMessage retained


# ---------------------------------------------------------------------------
# partition_messages
# ---------------------------------------------------------------------------


class TestPartitionMessages:
    """Tests for partition_messages."""

    def test_empty_messages(self) -> None:
        removable, retained = partition_messages(
            [], max_tokens=10, token_counter=char_token_counter
        )
        assert removable == []
        assert retained == []

    def test_partition_retain_end(self) -> None:
        messages = [
            HumanMessage(content="A" * 10),
            HumanMessage(content="B" * 20),
            HumanMessage(content="C" * 30),
        ]
        removable, retained = partition_messages(
            messages, max_tokens=35, token_counter=char_token_counter, retain="end"
        )
        assert removable == [messages[0], messages[1]]
        assert retained == [messages[2]]

    def test_partition_retain_start(self) -> None:
        messages = [
            HumanMessage(content="A" * 10),
            HumanMessage(content="B" * 20),
            HumanMessage(content="C" * 30),
        ]
        removable, retained = partition_messages(
            messages, max_tokens=35, token_counter=char_token_counter, retain="start"
        )
        assert removable == [messages[2]]
        assert retained == [messages[0], messages[1]]

    def test_partition_with_tool_boundary(self) -> None:
        """Tool boundaries respected in partition output."""
        messages = [
            HumanMessage(content="H1"),
            AIMessage(
                content="AI1",
                tool_calls=[{"name": "t1", "args": {}, "id": "call1"}],
            ),
            ToolMessage(content="R1", tool_call_id="call1"),
            HumanMessage(content="H2"),
        ]

        def custom_counter(msgs: list) -> int:
            tokens = 0
            for m in msgs:
                if isinstance(m, ToolMessage) or (
                    isinstance(m, AIMessage) and m.tool_calls
                ):
                    tokens += 15
                else:
                    tokens += 10
            return tokens

        removable, retained = partition_messages(
            messages, max_tokens=45, token_counter=custom_counter, retain="end"
        )
        assert removable == [messages[0]]
        assert retained == messages[1:]
