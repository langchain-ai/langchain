import base64
import json
import math
import re
from collections.abc import Callable, Sequence
from typing import Any, TypedDict

import pytest
from typing_extensions import NotRequired, override

from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.utils import (
    MessageLikeRepresentation,
    convert_to_messages,
    convert_to_openai_messages,
    count_tokens_approximately,
    filter_messages,
    get_buffer_string,
    merge_message_runs,
    trim_messages,
)
from langchain_core.tools import BaseTool


@pytest.mark.parametrize("msg_cls", [HumanMessage, AIMessage, SystemMessage])
def test_merge_message_runs_str(msg_cls: type[BaseMessage]) -> None:
    messages = [msg_cls("foo"), msg_cls("bar"), msg_cls("baz")]
    messages_model_copy = [m.model_copy(deep=True) for m in messages]
    expected = [msg_cls("foo\nbar\nbaz")]
    actual = merge_message_runs(messages)
    assert actual == expected
    assert messages == messages_model_copy


@pytest.mark.parametrize("msg_cls", [HumanMessage, AIMessage, SystemMessage])
def test_merge_message_runs_str_with_specified_separator(
    msg_cls: type[BaseMessage],
) -> None:
    messages = [msg_cls("foo"), msg_cls("bar"), msg_cls("baz")]
    messages_model_copy = [m.model_copy(deep=True) for m in messages]
    expected = [msg_cls("foo<sep>bar<sep>baz")]
    actual = merge_message_runs(messages, chunk_separator="<sep>")
    assert actual == expected
    assert messages == messages_model_copy


@pytest.mark.parametrize("msg_cls", [HumanMessage, AIMessage, SystemMessage])
def test_merge_message_runs_str_without_separator(
    msg_cls: type[BaseMessage],
) -> None:
    messages = [msg_cls("foo"), msg_cls("bar"), msg_cls("baz")]
    messages_model_copy = [m.model_copy(deep=True) for m in messages]
    expected = [msg_cls("foobarbaz")]
    actual = merge_message_runs(messages, chunk_separator="")
    assert actual == expected
    assert messages == messages_model_copy


def test_merge_message_runs_response_metadata() -> None:
    messages = [
        AIMessage("foo", id="1", response_metadata={"input_tokens": 1}),
        AIMessage("bar", id="2", response_metadata={"input_tokens": 2}),
    ]
    expected = [
        AIMessage(
            "foo\nbar",
            id="1",
            response_metadata={"input_tokens": 1},
        )
    ]
    actual = merge_message_runs(messages)
    assert actual == expected
    # Check it's not mutated
    assert messages[1].response_metadata == {"input_tokens": 2}


def test_merge_message_runs_content() -> None:
    messages = [
        AIMessage("foo", id="1"),
        AIMessage(
            [
                {"text": "bar", "type": "text"},
                {"image_url": "...", "type": "image_url"},
            ],
            tool_calls=[
                ToolCall(name="foo_tool", args={"x": 1}, id="tool1", type="tool_call")
            ],
            id="2",
        ),
        AIMessage(
            "baz",
            tool_calls=[
                ToolCall(name="foo_tool", args={"x": 5}, id="tool2", type="tool_call")
            ],
            id="3",
        ),
    ]
    messages_model_copy = [m.model_copy(deep=True) for m in messages]
    expected = [
        AIMessage(
            [
                "foo",
                {"text": "bar", "type": "text"},
                {"image_url": "...", "type": "image_url"},
                "baz",
            ],
            tool_calls=[
                ToolCall(name="foo_tool", args={"x": 1}, id="tool1", type="tool_call"),
                ToolCall(name="foo_tool", args={"x": 5}, id="tool2", type="tool_call"),
            ],
            id="1",
        ),
    ]
    actual = merge_message_runs(messages)
    assert actual == expected
    invoked = merge_message_runs().invoke(messages)
    assert actual == invoked
    assert messages == messages_model_copy


def test_merge_messages_tool_messages() -> None:
    messages = [
        ToolMessage("foo", tool_call_id="1"),
        ToolMessage("bar", tool_call_id="2"),
    ]
    messages_model_copy = [m.model_copy(deep=True) for m in messages]
    actual = merge_message_runs(messages)
    assert actual == messages
    assert messages == messages_model_copy


class FilterFields(TypedDict):
    include_names: NotRequired[Sequence[str]]
    exclude_names: NotRequired[Sequence[str]]
    include_types: NotRequired[Sequence[str | type[BaseMessage]]]
    exclude_types: NotRequired[Sequence[str | type[BaseMessage]]]
    include_ids: NotRequired[Sequence[str]]
    exclude_ids: NotRequired[Sequence[str]]
    exclude_tool_calls: NotRequired[Sequence[str] | bool]


@pytest.mark.parametrize(
    "filters",
    [
        {"include_names": ["blur"]},
        {"exclude_names": ["blah"]},
        {"include_ids": ["2"]},
        {"exclude_ids": ["1"]},
        {"include_types": "human"},
        {"include_types": ["human"]},
        {"include_types": HumanMessage},
        {"include_types": [HumanMessage]},
        {"exclude_types": "system"},
        {"exclude_types": ["system"]},
        {"exclude_types": SystemMessage},
        {"exclude_types": [SystemMessage]},
        {"include_names": ["blah", "blur"], "exclude_types": [SystemMessage]},
    ],
)
def test_filter_message(filters: FilterFields) -> None:
    messages = [
        SystemMessage("foo", name="blah", id="1"),
        HumanMessage("bar", name="blur", id="2"),
    ]
    messages_model_copy = [m.model_copy(deep=True) for m in messages]
    expected = messages[1:2]
    actual = filter_messages(messages, **filters)
    assert expected == actual
    invoked = filter_messages(**filters).invoke(messages)
    assert invoked == actual
    assert messages == messages_model_copy


def test_filter_message_exclude_tool_calls() -> None:
    tool_calls = [
        {"name": "foo", "id": "1", "args": {}, "type": "tool_call"},
        {"name": "bar", "id": "2", "args": {}, "type": "tool_call"},
    ]
    messages = [
        HumanMessage("foo", name="blah", id="1"),
        AIMessage("foo-response", name="blah", id="2"),
        HumanMessage("bar", name="blur", id="3"),
        AIMessage(
            "bar-response",
            tool_calls=tool_calls,
            id="4",
        ),
        ToolMessage("baz", tool_call_id="1", id="5"),
        ToolMessage("qux", tool_call_id="2", id="6"),
    ]
    messages_model_copy = [m.model_copy(deep=True) for m in messages]
    expected = messages[:3]

    # test excluding all tool calls
    actual = filter_messages(messages, exclude_tool_calls=True)
    assert expected == actual

    # test explicitly excluding all tool calls
    actual = filter_messages(messages, exclude_tool_calls=["1", "2"])
    assert expected == actual

    # test excluding a specific tool call
    expected = messages[:5]
    expected[3] = expected[3].model_copy(update={"tool_calls": [tool_calls[0]]})
    actual = filter_messages(messages, exclude_tool_calls=["2"])
    assert expected == actual

    # assert that we didn't mutate the original messages
    assert messages == messages_model_copy


def test_filter_message_exclude_tool_calls_content_blocks() -> None:
    tool_calls = [
        {"name": "foo", "id": "1", "args": {}, "type": "tool_call"},
        {"name": "bar", "id": "2", "args": {}, "type": "tool_call"},
    ]
    messages = [
        HumanMessage("foo", name="blah", id="1"),
        AIMessage("foo-response", name="blah", id="2"),
        HumanMessage("bar", name="blur", id="3"),
        AIMessage(
            [
                {"text": "bar-response", "type": "text"},
                {"name": "foo", "type": "tool_use", "id": "1"},
                {"name": "bar", "type": "tool_use", "id": "2"},
            ],
            tool_calls=tool_calls,
            id="4",
        ),
        ToolMessage("baz", tool_call_id="1", id="5"),
        ToolMessage("qux", tool_call_id="2", id="6"),
    ]
    messages_model_copy = [m.model_copy(deep=True) for m in messages]
    expected = messages[:3]

    # test excluding all tool calls
    actual = filter_messages(messages, exclude_tool_calls=True)
    assert expected == actual

    # test explicitly excluding all tool calls
    actual = filter_messages(messages, exclude_tool_calls=["1", "2"])
    assert expected == actual

    # test excluding a specific tool call
    expected = messages[:4] + messages[-1:]
    expected[3] = expected[3].model_copy(
        update={
            "tool_calls": [tool_calls[1]],
            "content": [
                {"text": "bar-response", "type": "text"},
                {"name": "bar", "type": "tool_use", "id": "2"},
            ],
        }
    )
    actual = filter_messages(messages, exclude_tool_calls=["1"])
    assert expected == actual

    # assert that we didn't mutate the original messages
    assert messages == messages_model_copy


_MESSAGES_TO_TRIM = [
    SystemMessage("This is a 4 token text."),
    HumanMessage("This is a 4 token text.", id="first"),
    AIMessage(
        [
            {"type": "text", "text": "This is the FIRST 4 token block."},
            {"type": "text", "text": "This is the SECOND 4 token block."},
        ],
        id="second",
    ),
    HumanMessage("This is a 4 token text.", id="third"),
    AIMessage("This is a 4 token text.", id="fourth"),
]
_MESSAGES_TO_TRIM_COPY = [m.model_copy(deep=True) for m in _MESSAGES_TO_TRIM]


def test_trim_messages_first_30() -> None:
    expected = [
        SystemMessage("This is a 4 token text."),
        HumanMessage("This is a 4 token text.", id="first"),
    ]
    actual = trim_messages(
        _MESSAGES_TO_TRIM,
        max_tokens=30,
        token_counter=dummy_token_counter,
        strategy="first",
    )
    assert actual == expected
    assert _MESSAGES_TO_TRIM == _MESSAGES_TO_TRIM_COPY


def test_trim_messages_first_30_allow_partial() -> None:
    expected = [
        SystemMessage("This is a 4 token text."),
        HumanMessage("This is a 4 token text.", id="first"),
        AIMessage(
            [{"type": "text", "text": "This is the FIRST 4 token block."}], id="second"
        ),
    ]
    actual = trim_messages(
        _MESSAGES_TO_TRIM,
        max_tokens=30,
        token_counter=dummy_token_counter,
        strategy="first",
        allow_partial=True,
    )
    assert actual == expected
    assert _MESSAGES_TO_TRIM == _MESSAGES_TO_TRIM_COPY


def test_trim_messages_first_30_allow_partial_end_on_human() -> None:
    expected = [
        SystemMessage("This is a 4 token text."),
        HumanMessage("This is a 4 token text.", id="first"),
    ]

    actual = trim_messages(
        _MESSAGES_TO_TRIM,
        max_tokens=30,
        token_counter=dummy_token_counter,
        strategy="first",
        allow_partial=True,
        end_on="human",
    )
    assert actual == expected
    assert _MESSAGES_TO_TRIM == _MESSAGES_TO_TRIM_COPY


def test_trim_messages_last_30_include_system() -> None:
    expected = [
        SystemMessage("This is a 4 token text."),
        HumanMessage("This is a 4 token text.", id="third"),
        AIMessage("This is a 4 token text.", id="fourth"),
    ]

    actual = trim_messages(
        _MESSAGES_TO_TRIM,
        max_tokens=30,
        include_system=True,
        token_counter=dummy_token_counter,
        strategy="last",
    )
    assert actual == expected
    assert _MESSAGES_TO_TRIM == _MESSAGES_TO_TRIM_COPY


def test_trim_messages_last_40_include_system_allow_partial() -> None:
    expected = [
        SystemMessage("This is a 4 token text."),
        AIMessage(
            [
                {"type": "text", "text": "This is the SECOND 4 token block."},
            ],
            id="second",
        ),
        HumanMessage("This is a 4 token text.", id="third"),
        AIMessage("This is a 4 token text.", id="fourth"),
    ]

    actual = trim_messages(
        _MESSAGES_TO_TRIM,
        max_tokens=40,
        token_counter=dummy_token_counter,
        strategy="last",
        allow_partial=True,
        include_system=True,
    )

    assert actual == expected
    assert _MESSAGES_TO_TRIM == _MESSAGES_TO_TRIM_COPY


def test_trim_messages_last_30_include_system_allow_partial_end_on_human() -> None:
    expected = [
        SystemMessage("This is a 4 token text."),
        AIMessage(
            [
                {"type": "text", "text": "This is the SECOND 4 token block."},
            ],
            id="second",
        ),
        HumanMessage("This is a 4 token text.", id="third"),
    ]

    actual = trim_messages(
        _MESSAGES_TO_TRIM,
        max_tokens=30,
        token_counter=dummy_token_counter,
        strategy="last",
        allow_partial=True,
        include_system=True,
        end_on="human",
    )

    assert actual == expected
    assert _MESSAGES_TO_TRIM == _MESSAGES_TO_TRIM_COPY


def test_trim_messages_last_40_include_system_allow_partial_start_on_human() -> None:
    expected = [
        SystemMessage("This is a 4 token text."),
        HumanMessage("This is a 4 token text.", id="third"),
        AIMessage("This is a 4 token text.", id="fourth"),
    ]

    actual = trim_messages(
        _MESSAGES_TO_TRIM,
        max_tokens=30,
        token_counter=dummy_token_counter,
        strategy="last",
        allow_partial=True,
        include_system=True,
        start_on="human",
    )

    assert actual == expected
    assert _MESSAGES_TO_TRIM == _MESSAGES_TO_TRIM_COPY


def test_trim_messages_allow_partial_one_message() -> None:
    expected = [
        HumanMessage("Th", id="third"),
    ]

    actual = trim_messages(
        [HumanMessage("This is a funky text.", id="third")],
        max_tokens=2,
        token_counter=lambda messages: sum(len(m.content) for m in messages),
        text_splitter=list,
        strategy="first",
        allow_partial=True,
    )

    assert actual == expected
    assert _MESSAGES_TO_TRIM == _MESSAGES_TO_TRIM_COPY


def test_trim_messages_last_allow_partial_one_message() -> None:
    expected = [
        HumanMessage("t.", id="third"),
    ]

    actual = trim_messages(
        [HumanMessage("This is a funky text.", id="third")],
        max_tokens=2,
        token_counter=lambda messages: sum(len(m.content) for m in messages),
        text_splitter=list,
        strategy="last",
        allow_partial=True,
    )

    assert actual == expected
    assert _MESSAGES_TO_TRIM == _MESSAGES_TO_TRIM_COPY


def test_trim_messages_allow_partial_text_splitter() -> None:
    expected = [
        HumanMessage("a 4 token text.", id="third"),
        AIMessage("This is a 4 token text.", id="fourth"),
    ]

    def count_words(msgs: list[BaseMessage]) -> int:
        count = 0
        for msg in msgs:
            if isinstance(msg.content, str):
                count += len(msg.content.split(" "))
            else:
                count += len(
                    " ".join(block["text"] for block in msg.content).split(" ")  # type: ignore[index]
                )
        return count

    def _split_on_space(text: str) -> list[str]:
        splits = text.split(" ")
        return [s + " " for s in splits[:-1]] + splits[-1:]

    actual = trim_messages(
        _MESSAGES_TO_TRIM,
        max_tokens=10,
        token_counter=count_words,
        strategy="last",
        allow_partial=True,
        text_splitter=_split_on_space,
    )
    assert actual == expected
    assert _MESSAGES_TO_TRIM == _MESSAGES_TO_TRIM_COPY


def test_trim_messages_include_system_strategy_last_empty_messages() -> None:
    expected: list[BaseMessage] = []

    actual = trim_messages(
        max_tokens=10,
        token_counter=dummy_token_counter,
        strategy="last",
        include_system=True,
    ).invoke([])

    assert actual == expected


def test_trim_messages_invoke() -> None:
    actual = trim_messages(max_tokens=10, token_counter=dummy_token_counter).invoke(
        _MESSAGES_TO_TRIM
    )
    expected = trim_messages(
        _MESSAGES_TO_TRIM, max_tokens=10, token_counter=dummy_token_counter
    )
    assert actual == expected


def test_trim_messages_bound_model_token_counter() -> None:
    trimmer = trim_messages(
        max_tokens=10,
        token_counter=FakeTokenCountingModel().bind(foo="bar"),  # type: ignore[call-overload]
    )
    trimmer.invoke([HumanMessage("foobar")])


def test_trim_messages_bad_token_counter() -> None:
    trimmer = trim_messages(max_tokens=10, token_counter={})  # type: ignore[call-overload]
    with pytest.raises(
        ValueError,
        match=re.escape(
            "'token_counter' expected to be a model that implements "
            "'get_num_tokens_from_messages()' or a function. "
            "Received object of type <class 'dict'>."
        ),
    ):
        trimmer.invoke([HumanMessage("foobar")])


def dummy_token_counter(messages: list[BaseMessage]) -> int:
    # treat each message like it adds 3 default tokens at the beginning
    # of the message and at the end of the message. 3 + 4 + 3 = 10 tokens
    # per message.

    default_content_len = 4
    default_msg_prefix_len = 3
    default_msg_suffix_len = 3

    count = 0
    for msg in messages:
        if isinstance(msg.content, str):
            count += (
                default_msg_prefix_len + default_content_len + default_msg_suffix_len
            )
        if isinstance(msg.content, list):
            count += (
                default_msg_prefix_len
                + len(msg.content) * default_content_len
                + default_msg_suffix_len
            )
    return count


def test_trim_messages_partial_text_splitting() -> None:
    messages = [HumanMessage(content="This is a long message that needs trimming")]
    messages_copy = [m.model_copy(deep=True) for m in messages]

    def count_characters(msgs: list[BaseMessage]) -> int:
        return sum(len(m.content) if isinstance(m.content, str) else 0 for m in msgs)

    # Return individual characters to test text splitting
    def char_splitter(text: str) -> list[str]:
        return list(text)

    result = trim_messages(
        messages,
        max_tokens=10,  # Only allow 10 characters
        token_counter=count_characters,
        strategy="first",
        allow_partial=True,
        text_splitter=char_splitter,
    )

    assert len(result) == 1
    assert result[0].content == "This is a "  # First 10 characters
    assert messages == messages_copy


def test_trim_messages_mixed_content_with_partial() -> None:
    messages = [
        AIMessage(
            content=[
                {"type": "text", "text": "First part of text."},
                {"type": "text", "text": "Second part that should be trimmed."},
            ]
        )
    ]
    messages_copy = [m.model_copy(deep=True) for m in messages]

    # Count total length of all text parts
    def count_text_length(msgs: list[BaseMessage]) -> int:
        total = 0
        for msg in msgs:
            if isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total += len(block["text"])
            elif isinstance(msg.content, str):
                total += len(msg.content)
        return total

    result = trim_messages(
        messages,
        max_tokens=20,  # Only allow first text block
        token_counter=count_text_length,
        strategy="first",
        allow_partial=True,
    )

    assert len(result) == 1
    assert len(result[0].content) == 1
    content = result[0].content[0]
    assert isinstance(content, dict)
    assert content["text"] == "First part of text."
    assert messages == messages_copy


def test_trim_messages_exact_token_boundary() -> None:
    messages = [
        SystemMessage(content="10 tokens exactly."),
        HumanMessage(content="Another 10 tokens."),
    ]

    # First message only
    result1 = trim_messages(
        messages,
        max_tokens=10,  # Exactly the size of first message
        token_counter=dummy_token_counter,
        strategy="first",
    )
    assert len(result1) == 1
    assert result1[0].content == "10 tokens exactly."

    # Both messages exactly fit
    result2 = trim_messages(
        messages,
        max_tokens=20,  # Exactly the size of both messages
        token_counter=dummy_token_counter,
        strategy="first",
    )
    assert len(result2) == 2
    assert result2 == messages


def test_trim_messages_start_on_with_allow_partial() -> None:
    messages = [
        HumanMessage(content="First human message"),
        AIMessage(content="AI response"),
        HumanMessage(content="Second human message"),
    ]
    messages_copy = [m.model_copy(deep=True) for m in messages]
    result = trim_messages(
        messages,
        max_tokens=20,
        token_counter=dummy_token_counter,
        strategy="last",
        allow_partial=True,
        start_on="human",
    )

    assert len(result) == 1
    assert result[0].content == "Second human message"
    assert messages == messages_copy


def test_trim_messages_token_counter_shortcut_approximate() -> None:
    """Test that `'approximate'` shortcut works for `token_counter`."""
    messages = [
        SystemMessage("This is a test message"),
        HumanMessage("Another test message", id="first"),
        AIMessage("AI response here", id="second"),
    ]
    messages_copy = [m.model_copy(deep=True) for m in messages]

    # Test using the "approximate" shortcut
    result_shortcut = trim_messages(
        messages,
        max_tokens=50,
        token_counter="approximate",
        strategy="last",
    )

    # Test using count_tokens_approximately directly
    result_direct = trim_messages(
        messages,
        max_tokens=50,
        token_counter=count_tokens_approximately,
        strategy="last",
    )

    # Both should produce the same result
    assert result_shortcut == result_direct
    assert messages == messages_copy


def test_trim_messages_token_counter_shortcut_invalid() -> None:
    """Test that invalid `token_counter` shortcut raises `ValueError`."""
    messages = [
        SystemMessage("This is a test message"),
        HumanMessage("Another test message"),
    ]

    # Test with invalid shortcut - intentionally passing invalid string to verify
    # runtime error handling for dynamically-constructed inputs
    with pytest.raises(ValueError, match="Invalid token_counter shortcut 'invalid'"):
        trim_messages(  # type: ignore[call-overload]
            messages,
            max_tokens=50,
            token_counter="invalid",
            strategy="last",
        )


def test_trim_messages_token_counter_shortcut_with_options() -> None:
    """Test that `'approximate'` shortcut works with different trim options."""
    messages = [
        SystemMessage("System instructions"),
        HumanMessage("First human message", id="first"),
        AIMessage("First AI response", id="ai1"),
        HumanMessage("Second human message", id="second"),
        AIMessage("Second AI response", id="ai2"),
    ]
    messages_copy = [m.model_copy(deep=True) for m in messages]

    # Test with various options
    result = trim_messages(
        messages,
        max_tokens=100,
        token_counter="approximate",
        strategy="last",
        include_system=True,
        start_on="human",
    )

    # Should include system message and start on human
    assert len(result) >= 2
    assert isinstance(result[0], SystemMessage)
    assert any(isinstance(msg, HumanMessage) for msg in result[1:])
    assert messages == messages_copy


class FakeTokenCountingModel(FakeChatModel):
    @override
    def get_num_tokens_from_messages(
        self,
        messages: list[BaseMessage],
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool] | None = None,
    ) -> int:
        return dummy_token_counter(messages)


def test_convert_to_messages() -> None:
    message_like: list = [
        # BaseMessage
        SystemMessage("1"),
        SystemMessage("1.1", additional_kwargs={"__openai_role__": "developer"}),
        HumanMessage([{"type": "image_url", "image_url": {"url": "2.1"}}], name="2.2"),
        AIMessage(
            [
                {"type": "text", "text": "3.1"},
                {
                    "type": "tool_use",
                    "id": "3.2",
                    "name": "3.3",
                    "input": {"3.4": "3.5"},
                },
            ]
        ),
        AIMessage(
            [
                {"type": "text", "text": "4.1"},
                {
                    "type": "tool_use",
                    "id": "4.2",
                    "name": "4.3",
                    "input": {"4.4": "4.5"},
                },
            ],
            tool_calls=[
                {
                    "name": "4.3",
                    "args": {"4.4": "4.5"},
                    "id": "4.2",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage("5.1", tool_call_id="5.2", name="5.3"),
        # OpenAI dict
        {"role": "system", "content": "6"},
        {"role": "developer", "content": "6.1"},
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "7.1"}}],
            "name": "7.2",
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "8.1"}],
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "arguments": json.dumps({"8.2": "8.3"}),
                        "name": "8.4",
                    },
                    "id": "8.5",
                }
            ],
            "name": "8.6",
        },
        {"role": "tool", "content": "10.1", "tool_call_id": "10.2"},
        # Tuple/List
        ("system", "11.1"),
        ("developer", "11.2"),
        ("human", [{"type": "image_url", "image_url": {"url": "12.1"}}]),
        (
            "ai",
            [
                {"type": "text", "text": "13.1"},
                {
                    "type": "tool_use",
                    "id": "13.2",
                    "name": "13.3",
                    "input": {"13.4": "13.5"},
                },
            ],
        ),
        # String
        "14.1",
        # LangChain dict
        {
            "role": "ai",
            "content": [{"type": "text", "text": "15.1"}],
            "tool_calls": [{"args": {"15.2": "15.3"}, "name": "15.4", "id": "15.5"}],
            "name": "15.6",
        },
    ]
    expected = [
        SystemMessage(content="1"),
        SystemMessage(
            content="1.1", additional_kwargs={"__openai_role__": "developer"}
        ),
        HumanMessage(
            content=[{"type": "image_url", "image_url": {"url": "2.1"}}], name="2.2"
        ),
        AIMessage(
            content=[
                {"type": "text", "text": "3.1"},
                {
                    "type": "tool_use",
                    "id": "3.2",
                    "name": "3.3",
                    "input": {"3.4": "3.5"},
                },
            ]
        ),
        AIMessage(
            content=[
                {"type": "text", "text": "4.1"},
                {
                    "type": "tool_use",
                    "id": "4.2",
                    "name": "4.3",
                    "input": {"4.4": "4.5"},
                },
            ],
            tool_calls=[
                {
                    "name": "4.3",
                    "args": {"4.4": "4.5"},
                    "id": "4.2",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="5.1", name="5.3", tool_call_id="5.2"),
        SystemMessage(content="6"),
        SystemMessage(
            content="6.1", additional_kwargs={"__openai_role__": "developer"}
        ),
        HumanMessage(
            content=[{"type": "image_url", "image_url": {"url": "7.1"}}], name="7.2"
        ),
        AIMessage(
            content=[{"type": "text", "text": "8.1"}],
            name="8.6",
            tool_calls=[
                {
                    "name": "8.4",
                    "args": {"8.2": "8.3"},
                    "id": "8.5",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="10.1", tool_call_id="10.2"),
        SystemMessage(content="11.1"),
        SystemMessage(
            content="11.2", additional_kwargs={"__openai_role__": "developer"}
        ),
        HumanMessage(content=[{"type": "image_url", "image_url": {"url": "12.1"}}]),
        AIMessage(
            content=[
                {"type": "text", "text": "13.1"},
                {
                    "type": "tool_use",
                    "id": "13.2",
                    "name": "13.3",
                    "input": {"13.4": "13.5"},
                },
            ]
        ),
        HumanMessage(content="14.1"),
        AIMessage(
            content=[{"type": "text", "text": "15.1"}],
            name="15.6",
            tool_calls=[
                {
                    "name": "15.4",
                    "args": {"15.2": "15.3"},
                    "id": "15.5",
                    "type": "tool_call",
                }
            ],
        ),
    ]
    actual = convert_to_messages(message_like)
    assert expected == actual


def test_convert_to_messages_openai_refusal() -> None:
    actual = convert_to_messages(
        [{"role": "assistant", "content": "", "refusal": "9.1"}]
    )
    expected = [AIMessage("", additional_kwargs={"refusal": "9.1"})]
    assert actual == expected

    # Raises error if content is missing.
    with pytest.raises(
        ValueError, match="Message dict must contain 'role' and 'content' keys"
    ):
        convert_to_messages([{"role": "assistant", "refusal": "9.1"}])


def create_image_data() -> str:
    return "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigD//2Q=="  # noqa: E501


def create_base64_image(image_format: str = "jpeg") -> str:
    data = create_image_data()
    return f"data:image/{image_format};base64,{data}"


def test_convert_to_openai_messages_string() -> None:
    message = "Hello"
    result = convert_to_openai_messages(message)
    assert result == {"role": "user", "content": "Hello"}


def test_convert_to_openai_messages_single_message() -> None:
    message: BaseMessage = HumanMessage(content="Hello")
    result = convert_to_openai_messages(message)
    assert result == {"role": "user", "content": "Hello"}

    # Test IDs
    result = convert_to_openai_messages(message, include_id=True)
    assert result == {"role": "user", "content": "Hello"}  # no ID

    message = AIMessage(content="Hello", id="resp_123")
    result = convert_to_openai_messages(message)
    assert result == {"role": "assistant", "content": "Hello"}

    result = convert_to_openai_messages(message, include_id=True)
    assert result == {"role": "assistant", "content": "Hello", "id": "resp_123"}


def test_convert_to_openai_messages_multiple_messages() -> None:
    messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="Human message"),
        AIMessage(content="AI message"),
    ]
    result = convert_to_openai_messages(messages)
    expected = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "Human message"},
        {"role": "assistant", "content": "AI message"},
    ]
    assert result == expected


def test_convert_to_openai_messages_openai_string() -> None:
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ]
        ),
        AIMessage(
            content=[{"type": "text", "text": "Hi"}, {"type": "text", "text": "there"}]
        ),
    ]
    result = convert_to_openai_messages(messages)
    expected = [
        {"role": "user", "content": "Hello\nWorld"},
        {"role": "assistant", "content": "Hi\nthere"},
    ]
    assert result == expected


def test_convert_to_openai_messages_openai_block() -> None:
    messages = [HumanMessage(content="Hello"), AIMessage(content="Hi there")]
    result = convert_to_openai_messages(messages, text_format="block")
    expected = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi there"}]},
    ]
    assert result == expected


def test_convert_to_openai_messages_invalid_format() -> None:
    with pytest.raises(ValueError, match="Unrecognized text_format="):
        convert_to_openai_messages(  # type: ignore[call-overload]
            [HumanMessage(content="Hello")],
            text_format="invalid",
        )


def test_convert_to_openai_messages_openai_image() -> None:
    base64_image = create_base64_image()
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Here's an image:"},
                {"type": "image_url", "image_url": {"url": base64_image}},
            ]
        )
    ]
    result = convert_to_openai_messages(messages, text_format="block")
    expected = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here's an image:"},
                {"type": "image_url", "image_url": {"url": base64_image}},
            ],
        }
    ]
    assert result == expected


def test_convert_to_openai_messages_anthropic() -> None:
    image_data = create_image_data()
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Here's an image:",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                },
            ]
        ),
        AIMessage(
            content=[
                {"type": "tool_use", "name": "foo", "input": {"bar": "baz"}, "id": "1"}
            ]
        ),
        HumanMessage(
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "1",
                    "is_error": False,
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                    ],
                }
            ]
        ),
    ]
    result = convert_to_openai_messages(messages)
    expected = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here's an image:"},
                {"type": "image_url", "image_url": {"url": create_base64_image()}},
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "foo",
                        "arguments": json.dumps({"bar": "baz"}),
                    },
                    "id": "1",
                }
            ],
        },
        {
            "role": "tool",
            "content": [
                {"type": "image_url", "image_url": {"url": create_base64_image()}}
            ],
            "tool_call_id": "1",
        },
    ]
    assert result == expected

    # Test thinking blocks (pass through)
    thinking_block = {
        "signature": "abc123",
        "thinking": "Thinking text.",
        "type": "thinking",
    }
    text_block = {"text": "Response text.", "type": "text"}
    messages = [AIMessage([thinking_block, text_block])]
    result = convert_to_openai_messages(messages)
    expected = [{"role": "assistant", "content": [thinking_block, text_block]}]
    assert result == expected


def test_convert_to_openai_messages_bedrock_converse_image() -> None:
    image_data = create_image_data()
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Here's an image:"},
                {
                    "image": {
                        "format": "jpeg",
                        "source": {"bytes": base64.b64decode(image_data)},
                    }
                },
            ]
        )
    ]
    result = convert_to_openai_messages(messages)
    assert result[0]["content"][1]["type"] == "image_url"
    assert result[0]["content"][1]["image_url"]["url"] == create_base64_image()


def test_convert_to_openai_messages_vertexai_image() -> None:
    image_data = create_image_data()
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Here's an image:"},
                {
                    "type": "media",
                    "mime_type": "image/jpeg",
                    "data": base64.b64decode(image_data),
                },
            ]
        )
    ]
    result = convert_to_openai_messages(messages)
    assert result[0]["content"][1]["type"] == "image_url"
    assert result[0]["content"][1]["image_url"]["url"] == create_base64_image()


def test_convert_to_openai_messages_tool_message() -> None:
    tool_message = ToolMessage(content="Tool result", tool_call_id="123")
    result = convert_to_openai_messages([tool_message], text_format="block")
    assert len(result) == 1
    assert result[0]["content"] == [{"type": "text", "text": "Tool result"}]
    assert result[0]["tool_call_id"] == "123"


def test_convert_to_openai_messages_tool_use() -> None:
    messages = [
        AIMessage(
            content=[
                {
                    "type": "tool_use",
                    "id": "123",
                    "name": "calculator",
                    "input": {"a": "b"},
                }
            ]
        )
    ]
    result = convert_to_openai_messages(messages, text_format="block")
    assert result[0]["tool_calls"][0]["type"] == "function"
    assert result[0]["tool_calls"][0]["id"] == "123"
    assert result[0]["tool_calls"][0]["function"]["name"] == "calculator"
    assert result[0]["tool_calls"][0]["function"]["arguments"] == json.dumps({"a": "b"})


def test_convert_to_openai_messages_tool_use_unicode() -> None:
    """Test that Unicode characters in tool call args are preserved correctly."""
    messages = [
        AIMessage(
            content=[
                {
                    "type": "tool_use",
                    "id": "123",
                    "name": "create_customer",
                    "input": {"customer_name": "你好啊集团"},
                }
            ]
        )
    ]
    result = convert_to_openai_messages(messages, text_format="block")
    assert result[0]["tool_calls"][0]["type"] == "function"
    assert result[0]["tool_calls"][0]["id"] == "123"
    assert result[0]["tool_calls"][0]["function"]["name"] == "create_customer"
    # Ensure Unicode characters are preserved, not escaped as \\uXXXX
    arguments_str = result[0]["tool_calls"][0]["function"]["arguments"]
    parsed_args = json.loads(arguments_str)
    assert parsed_args["customer_name"] == "你好啊集团"
    # Also ensure the raw JSON string contains Unicode, not escaped sequences
    assert "你好啊集团" in arguments_str
    assert "\\u4f60" not in arguments_str  # Should not contain escaped Unicode


def test_convert_to_openai_messages_json() -> None:
    json_data = {"key": "value"}
    messages = [HumanMessage(content=[{"type": "json", "json": json_data}])]
    result = convert_to_openai_messages(messages, text_format="block")
    assert result[0]["content"][0]["type"] == "text"
    assert json.loads(result[0]["content"][0]["text"]) == json_data


def test_convert_to_openai_messages_guard_content() -> None:
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "guard_content",
                    "guard_content": {"text": "Protected content"},
                }
            ]
        )
    ]
    result = convert_to_openai_messages(messages, text_format="block")
    assert result[0]["content"][0]["type"] == "text"
    assert result[0]["content"][0]["text"] == "Protected content"


def test_convert_to_openai_messages_invalid_block() -> None:
    messages = [HumanMessage(content=[{"type": "invalid", "foo": "bar"}])]
    with pytest.raises(ValueError, match="Unrecognized content block"):
        convert_to_openai_messages(
            messages,
            text_format="block",
            pass_through_unknown_blocks=False,
        )
    # Accept by default
    result = convert_to_openai_messages(messages, text_format="block")
    assert result == [{"role": "user", "content": [{"type": "invalid", "foo": "bar"}]}]


def test_handle_openai_responses_blocks() -> None:
    blocks: str | list[str | dict] = [
        {"type": "reasoning", "id": "1"},
        {
            "type": "function_call",
            "name": "multiply",
            "arguments": '{"x":5,"y":4}',
            "call_id": "call_abc123",
            "id": "fc_abc123",
            "status": "completed",
        },
    ]
    message = AIMessage(content=blocks)

    expected_tool_call = {
        "type": "function",
        "function": {
            "name": "multiply",
            "arguments": '{"x":5,"y":4}',
        },
        "id": "call_abc123",
    }
    result = convert_to_openai_messages(message)
    assert isinstance(result, dict)
    assert result["content"] == blocks
    assert result["tool_calls"] == [expected_tool_call]

    result = convert_to_openai_messages(message, pass_through_unknown_blocks=False)
    assert isinstance(result, dict)
    assert result["content"] == [{"type": "reasoning", "id": "1"}]
    assert result["tool_calls"] == [expected_tool_call]


def test_convert_to_openai_messages_empty_message() -> None:
    result = convert_to_openai_messages(HumanMessage(content=""))
    assert result == {"role": "user", "content": ""}


def test_convert_to_openai_messages_empty_list() -> None:
    result = convert_to_openai_messages([])
    assert result == []


def test_convert_to_openai_messages_mixed_content_types() -> None:
    messages = [
        HumanMessage(
            content=[
                "Text message",
                {"type": "text", "text": "Structured text"},
                {"type": "image_url", "image_url": {"url": create_base64_image()}},
            ]
        )
    ]
    result = convert_to_openai_messages(messages, text_format="block")
    assert len(result[0]["content"]) == 3
    assert isinstance(result[0]["content"][0], dict)
    assert isinstance(result[0]["content"][1], dict)
    assert isinstance(result[0]["content"][2], dict)


def test_convert_to_openai_messages_developer() -> None:
    messages: list[MessageLikeRepresentation] = [
        SystemMessage("a", additional_kwargs={"__openai_role__": "developer"}),
        {"role": "developer", "content": "a"},
    ]
    result = convert_to_openai_messages(messages)
    assert result == [{"role": "developer", "content": "a"}] * 2


def test_convert_to_openai_messages_multimodal() -> None:
    """v0 and v1 content to OpenAI messages conversion."""
    messages = [
        HumanMessage(
            content=[
                # Prior v0 blocks
                {"type": "text", "text": "Text message"},
                {
                    "type": "image",
                    "url": "https://example.com/test.png",
                },
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": "<base64 string>",
                    "mime_type": "image/png",
                },
                {
                    "type": "file",
                    "source_type": "base64",
                    "data": "<base64 string>",
                    "mime_type": "application/pdf",
                    "filename": "test.pdf",
                },
                {
                    # OpenAI Chat Completions file format
                    "type": "file",
                    "file": {
                        "filename": "draconomicon.pdf",
                        "file_data": "data:application/pdf;base64,<base64 string>",
                    },
                },
                {
                    "type": "file",
                    "source_type": "id",
                    "id": "file-abc123",
                },
                {
                    "type": "audio",
                    "source_type": "base64",
                    "data": "<base64 string>",
                    "mime_type": "audio/wav",
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": "<base64 string>",
                        "format": "wav",
                    },
                },
                # v1 Additions
                {
                    "type": "image",
                    "source_type": "url",  # backward compatibility v0 block field
                    "url": "https://example.com/test.png",
                },
                {
                    "type": "image",
                    "base64": "<base64 string>",
                    "mime_type": "image/png",
                },
                {
                    "type": "file",
                    "base64": "<base64 string>",
                    "mime_type": "application/pdf",
                    "filename": "test.pdf",  # backward compatibility v0 block field
                },
                {
                    "type": "file",
                    "file_id": "file-abc123",
                },
                {
                    "type": "audio",
                    "base64": "<base64 string>",
                    "mime_type": "audio/wav",
                },
            ]
        )
    ]
    result = convert_to_openai_messages(messages, text_format="block")
    assert len(result) == 1
    message = result[0]
    assert len(message["content"]) == 13

    # Test auto-adding filename
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "file",
                    "base64": "<base64 string>",
                    "mime_type": "application/pdf",
                },
            ]
        )
    ]
    with pytest.warns(match="filename"):
        result = convert_to_openai_messages(messages, text_format="block")
    assert len(result) == 1
    message = result[0]
    assert len(message["content"]) == 1
    block = message["content"][0]
    assert block == {
        # OpenAI Chat Completions file format
        "type": "file",
        "file": {
            "file_data": "data:application/pdf;base64,<base64 string>",
            "filename": "LC_AUTOGENERATED",
        },
    }


def test_count_tokens_approximately_empty_messages() -> None:
    # Test with empty message list
    assert count_tokens_approximately([]) == 0

    # Test with empty content
    messages = [HumanMessage(content="")]
    # 4 role chars -> 1 + 3 = 4 tokens
    assert count_tokens_approximately(messages) == 4


def test_count_tokens_approximately_with_names() -> None:
    messages = [
        # 5 chars + 4 role chars -> 3 + 3 = 6 tokens
        # (with name: extra 4 name chars, so total = 4 + 3 = 7 tokens)
        HumanMessage(content="Hello", name="user"),
        # 8 chars + 9 role chars -> 5 + 3 = 8 tokens
        # (with name: extra 9 name chars, so total = 7 + 3 = 10 tokens)
        AIMessage(content="Hi there", name="assistant"),
    ]
    # With names included (default)
    assert count_tokens_approximately(messages) == 17

    # Without names
    without_names = count_tokens_approximately(messages, count_name=False)
    assert without_names == 14


def test_count_tokens_approximately_openai_format() -> None:
    # same as test_count_tokens_approximately_with_names, but in OpenAI format
    messages = [
        {"role": "user", "content": "Hello", "name": "user"},
        {"role": "assistant", "content": "Hi there", "name": "assistant"},
    ]
    # With names included (default)
    assert count_tokens_approximately(messages) == 17

    # Without names
    without_names = count_tokens_approximately(messages, count_name=False)
    assert without_names == 14


def test_count_tokens_approximately_string_content() -> None:
    messages = [
        # 5 chars + 4 role chars -> 3 + 3 = 6 tokens
        HumanMessage(content="Hello"),
        # 8 chars + 9 role chars -> 5 + 3 = 8 tokens
        AIMessage(content="Hi there"),
        # 12 chars + 4 role chars -> 4 + 3 = 7 tokens
        HumanMessage(content="How are you?"),
    ]
    assert count_tokens_approximately(messages) == 21


def test_count_tokens_approximately_list_content() -> None:
    messages = [
        # '[{"foo": "bar"}]' -> 16 chars + 4 role chars -> 5 + 3 = 8 tokens
        HumanMessage(content=[{"foo": "bar"}]),
        # '[{"test": 123}]' -> 15 chars + 9 role chars -> 6 + 3 = 9 tokens
        AIMessage(content=[{"test": 123}]),
    ]
    assert count_tokens_approximately(messages) == 17


def test_count_tokens_approximately_tool_calls() -> None:
    tool_calls = [{"name": "test_tool", "args": {"foo": "bar"}, "id": "1"}]
    messages = [
        # tool calls json -> 79 chars + 9 role chars -> 22 + 3 = 25 tokens
        AIMessage(content="", tool_calls=tool_calls),
        # 15 chars + 4 role chars -> 5 + 3 = 8 tokens
        HumanMessage(content="Regular message"),
    ]
    assert count_tokens_approximately(messages) == 33
    # AI message w/ both content and tool calls
    # 94 chars + 9 role chars -> 26 + 3 = 29 tokens
    messages = [
        AIMessage(content="Regular message", tool_calls=tool_calls),
    ]
    assert count_tokens_approximately(messages) == 29


def test_count_tokens_approximately_custom_token_length() -> None:
    messages = [
        # 11 chars + 4 role chars -> (4 tokens of length 4 / 8 tokens of length 2) + 3
        HumanMessage(content="Hello world"),
        # 7 chars + 9 role chars -> (4 tokens of length 4 / 8 tokens of length 2) + 3
        AIMessage(content="Testing"),
    ]
    assert count_tokens_approximately(messages, chars_per_token=4) == 14
    assert count_tokens_approximately(messages, chars_per_token=2) == 22


def test_count_tokens_approximately_large_message_content() -> None:
    # Test with large content to ensure no issues
    large_text = "x" * 10000
    messages = [HumanMessage(content=large_text)]
    # 10,000 chars + 4 role chars -> 2501 + 3 = 2504 tokens
    assert count_tokens_approximately(messages) == 2504


def test_count_tokens_approximately_large_number_of_messages() -> None:
    # Test with large content to ensure no issues
    messages = [HumanMessage(content="x")] * 1_000
    # 1 chars + 4 role chars -> 2 + 3 = 5 tokens
    assert count_tokens_approximately(messages) == 5_000


def test_count_tokens_approximately_mixed_content_types() -> None:
    # Test with a variety of content types in the same message list
    tool_calls = [{"name": "test_tool", "args": {"foo": "bar"}, "id": "1"}]
    messages = [
        # 13 chars + 6 role chars -> 5 + 3 = 8 tokens
        SystemMessage(content="System prompt"),
        # '[{"foo": "bar"}]' -> 16 chars + 4 role chars -> 5 + 3 = 8 tokens
        HumanMessage(content=[{"foo": "bar"}]),
        # tool calls json -> 79 chars + 9 role chars -> 22 + 3 = 25 tokens
        AIMessage(content="", tool_calls=tool_calls),
        # 13 chars + 4 role chars + 9 name chars + 1 tool call ID char ->
        # 7 + 3 = 10 tokens
        ToolMessage(content="Tool response", name="test_tool", tool_call_id="1"),
    ]
    token_count = count_tokens_approximately(messages)
    assert token_count == 51

    # Ensure that count is consistent if we do one message at a time
    assert sum(count_tokens_approximately([m]) for m in messages) == token_count


def test_count_tokens_approximately_usage_metadata_scaling() -> None:
    messages = [
        HumanMessage("text"),
        AIMessage(
            "text",
            response_metadata={"model_provider": "openai"},
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 100},
        ),
        HumanMessage("text"),
        AIMessage(
            "text",
            response_metadata={"model_provider": "openai"},
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 200},
        ),
    ]

    unscaled = count_tokens_approximately(messages)
    scaled = count_tokens_approximately(messages, use_usage_metadata_scaling=True)

    ratio = scaled / unscaled
    assert 1 <= round(ratio, 1) <= 1.2  # we ceil scale token counts, so can be > 1.2

    messages.extend([ToolMessage("text", tool_call_id="abc123")] * 3)

    unscaled_extended = count_tokens_approximately(messages)
    scaled_extended = count_tokens_approximately(
        messages, use_usage_metadata_scaling=True
    )

    # scaling should still be based on the most recent AIMessage with total_tokens=200
    assert unscaled_extended > unscaled
    assert scaled_extended > scaled

    # And the scaled total should be the unscaled total multiplied by the same ratio.
    # ratio = 200 / unscaled (as of last AI message)
    expected_scaled_extended = math.ceil(unscaled_extended * ratio)
    assert scaled_extended <= expected_scaled_extended <= scaled_extended + 1


def test_count_tokens_approximately_usage_metadata_scaling_model_provider() -> None:
    messages = [
        HumanMessage("Hello"),
        AIMessage(
            "Hi",
            response_metadata={"model_provider": "openai"},
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 100},
        ),
        HumanMessage("More text"),
        AIMessage(
            "More response",
            response_metadata={"model_provider": "anthropic"},
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 200},
        ),
    ]

    unscaled = count_tokens_approximately(messages)
    scaled = count_tokens_approximately(messages, use_usage_metadata_scaling=True)
    assert scaled == unscaled


def test_count_tokens_approximately_usage_metadata_scaling_total_tokens() -> None:
    messages = [
        HumanMessage("Hello"),
        AIMessage(
            "Hi",
            response_metadata={"model_provider": "openai"},
            # no usage metadata -> skip
        ),
    ]

    unscaled = count_tokens_approximately(messages, chars_per_token=5)
    scaled = count_tokens_approximately(
        messages, chars_per_token=5, use_usage_metadata_scaling=True
    )

    assert scaled == unscaled


def test_count_tokens_approximately_usage_metadata_scaling_floor_at_one() -> None:
    messages = [
        HumanMessage("text"),
        AIMessage(
            "text",
            response_metadata={"model_provider": "openai"},
            # Set total_tokens lower than the approximate count up through this message.
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 1},
        ),
        HumanMessage("text"),
    ]

    unscaled = count_tokens_approximately(messages)
    scaled = count_tokens_approximately(messages, use_usage_metadata_scaling=True)

    # scale factor would be < 1, but we floor it at 1.0 to avoid decreasing counts
    assert scaled == unscaled


def test_get_buffer_string_with_structured_content() -> None:
    """Test get_buffer_string with structured content in messages."""
    messages = [
        HumanMessage(content=[{"type": "text", "text": "Hello, world!"}]),
        AIMessage(content=[{"type": "text", "text": "Hi there!"}]),
        SystemMessage(content=[{"type": "text", "text": "System message"}]),
    ]
    expected = "Human: Hello, world!\nAI: Hi there!\nSystem: System message"
    actual = get_buffer_string(messages)
    assert actual == expected


def test_get_buffer_string_with_mixed_content() -> None:
    """Test get_buffer_string with mixed content types in messages."""
    messages = [
        HumanMessage(content="Simple text"),
        AIMessage(content=[{"type": "text", "text": "Structured text"}]),
        SystemMessage(content=[{"type": "text", "text": "Another structured text"}]),
    ]
    expected = (
        "Human: Simple text\nAI: Structured text\nSystem: Another structured text"
    )
    actual = get_buffer_string(messages)
    assert actual == expected


def test_get_buffer_string_with_function_call() -> None:
    """Test get_buffer_string with function call in additional_kwargs."""
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(
            content="Hi",
            additional_kwargs={
                "function_call": {
                    "name": "test_function",
                    "arguments": '{"arg": "value"}',
                }
            },
        ),
    ]
    # TODO: consider changing this
    expected = (
        "Human: Hello\n"
        "AI: Hi{'name': 'test_function', 'arguments': '{\"arg\": \"value\"}'}"
    )
    actual = get_buffer_string(messages)
    assert actual == expected


def test_get_buffer_string_with_empty_content() -> None:
    """Test get_buffer_string with empty content in messages."""
    messages = [
        HumanMessage(content=[]),
        AIMessage(content=""),
        SystemMessage(content=[]),
    ]
    expected = "Human: \nAI: \nSystem: "
    actual = get_buffer_string(messages)
    assert actual == expected


def test_get_buffer_string_with_tool_calls() -> None:
    """Test `get_buffer_string` with `tool_calls` field."""
    messages = [
        HumanMessage(content="What's the weather?"),
        AIMessage(
            content="Let me check the weather",
            tool_calls=[
                {
                    "name": "get_weather",
                    "args": {"city": "NYC"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
    ]
    result = get_buffer_string(messages)
    assert "Human: What's the weather?" in result
    assert "AI: Let me check the weather" in result
    assert "get_weather" in result
    assert "NYC" in result


def test_get_buffer_string_with_tool_calls_empty_content() -> None:
    """Test `get_buffer_string` with `tool_calls` and empty `content`."""
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search",
                    "args": {"query": "test"},
                    "id": "call_2",
                    "type": "tool_call",
                }
            ],
        ),
    ]
    result = get_buffer_string(messages)
    assert "AI: " in result
    assert "search" in result


def test_get_buffer_string_tool_calls_preferred_over_function_call() -> None:
    """Test that `tool_calls` takes precedence over legacy `function_call`."""
    messages = [
        AIMessage(
            content="Calling tools",
            tool_calls=[
                {
                    "name": "modern_tool",
                    "args": {"key": "value"},
                    "id": "call_3",
                    "type": "tool_call",
                }
            ],
            additional_kwargs={
                "function_call": {"name": "legacy_function", "arguments": "{}"}
            },
        ),
    ]
    result = get_buffer_string(messages)
    assert "modern_tool" in result
    assert "legacy_function" not in result


def test_convert_to_openai_messages_reasoning_content() -> None:
    """Test convert_to_openai_messages with reasoning content blocks."""
    # Test reasoning block with empty summary
    msg = AIMessage(content=[{"type": "reasoning", "summary": []}])
    result = convert_to_openai_messages(msg, text_format="block")
    expected = {"role": "assistant", "content": [{"type": "reasoning", "summary": []}]}
    assert result == expected

    # Test reasoning block with summary content
    msg_with_summary = AIMessage(
        content=[
            {
                "type": "reasoning",
                "summary": [
                    {"type": "text", "text": "First thought"},
                    {"type": "text", "text": "Second thought"},
                ],
            }
        ]
    )
    result_with_summary = convert_to_openai_messages(
        msg_with_summary, text_format="block"
    )
    expected_with_summary = {
        "role": "assistant",
        "content": [
            {
                "type": "reasoning",
                "summary": [
                    {"type": "text", "text": "First thought"},
                    {"type": "text", "text": "Second thought"},
                ],
            }
        ],
    }
    assert result_with_summary == expected_with_summary

    # Test mixed content with reasoning and text
    mixed_msg = AIMessage(
        content=[
            {"type": "text", "text": "Regular response"},
            {
                "type": "reasoning",
                "summary": [{"type": "text", "text": "My reasoning process"}],
            },
        ]
    )
    mixed_result = convert_to_openai_messages(mixed_msg, text_format="block")
    expected_mixed = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Regular response"},
            {
                "type": "reasoning",
                "summary": [{"type": "text", "text": "My reasoning process"}],
            },
        ],
    }
    assert mixed_result == expected_mixed


# Tests for get_buffer_string XML format


def test_get_buffer_string_xml_empty_messages_list() -> None:
    """Test XML format with empty messages list."""
    messages: list[BaseMessage] = []
    result = get_buffer_string(messages, format="xml")
    expected = ""
    assert result == expected


def test_get_buffer_string_xml_basic() -> None:
    """Test XML format output with all message types."""
    messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="Human message"),
        AIMessage(content="AI message"),
        FunctionMessage(content="Function result", name="test_fn"),
        ToolMessage(content="Tool result", tool_call_id="123"),
    ]
    result = get_buffer_string(messages, format="xml")
    expected = (
        '<message type="system">System message</message>\n'
        '<message type="human">Human message</message>\n'
        '<message type="ai">AI message</message>\n'
        '<message type="function">Function result</message>\n'
        '<message type="tool">Tool result</message>'
    )
    assert result == expected


def test_get_buffer_string_xml_custom_prefixes() -> None:
    """Test XML format with custom human and ai prefixes."""
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
    ]
    result = get_buffer_string(
        messages, human_prefix="User", ai_prefix="Assistant", format="xml"
    )
    expected = (
        '<message type="user">Hello</message>\n'
        '<message type="assistant">Hi there</message>'
    )
    assert result == expected


def test_get_buffer_string_xml_custom_separator() -> None:
    """Test XML format with custom message separator."""
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
    ]
    result = get_buffer_string(messages, format="xml", message_separator="\n\n")
    expected = (
        '<message type="human">Hello</message>\n\n<message type="ai">Hi there</message>'
    )
    assert result == expected


def test_get_buffer_string_prefix_custom_separator() -> None:
    """Test prefix format with custom message separator."""
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
    ]
    result = get_buffer_string(messages, format="prefix", message_separator=" | ")
    expected = "Human: Hello | AI: Hi there"
    assert result == expected


def test_get_buffer_string_xml_escaping() -> None:
    """Test XML format properly escapes special characters in content."""
    messages = [
        HumanMessage(content="Is 5 < 10 & 10 > 5?"),
        AIMessage(content='Yes, and here\'s a "quote"'),
    ]
    result = get_buffer_string(messages, format="xml")
    # xml.sax.saxutils.escape escapes <, >, & (not quotes in content)
    expected = (
        '<message type="human">Is 5 &lt; 10 &amp; 10 &gt; 5?</message>\n'
        '<message type="ai">Yes, and here\'s a "quote"</message>'
    )
    assert result == expected


def test_get_buffer_string_xml_unicode_content() -> None:
    """Test XML format with Unicode content."""
    messages = [
        HumanMessage(content="你好世界"),  # Chinese: Hello World
        AIMessage(content="こんにちは"),  # Japanese: Hello
    ]
    result = get_buffer_string(messages, format="xml")
    expected = (
        '<message type="human">你好世界</message>\n'
        '<message type="ai">こんにちは</message>'
    )
    assert result == expected


def test_get_buffer_string_xml_chat_message_valid_role() -> None:
    """Test XML format with `ChatMessage` having valid XML tag name role."""
    messages = [
        ChatMessage(content="Hello", role="Assistant"),
    ]
    result = get_buffer_string(messages, format="xml")
    # Role is used directly as the type attribute value
    expected = '<message type="Assistant">Hello</message>'
    assert result == expected

    # Spaces in role
    messages = [
        ChatMessage(content="Hello", role="my custom role"),
    ]
    result = get_buffer_string(messages, format="xml")
    # Custom roles with spaces use quoteattr for proper escaping
    expected = '<message type="my custom role">Hello</message>'
    assert result == expected

    # Special characters in role
    messages = [
        ChatMessage(content="Hello", role='role"with<special>'),
    ]
    result = get_buffer_string(messages, format="xml")
    # quoteattr handles escaping of special characters in attribute values
    # Note: quoteattr uses single quotes when the string contains double quotes
    expected = """<message type='role"with&lt;special&gt;'>Hello</message>"""
    assert result == expected


def test_get_buffer_string_xml_empty_content() -> None:
    """Test XML format with empty content."""
    messages = [
        HumanMessage(content=""),
        AIMessage(content=""),
    ]
    result = get_buffer_string(messages, format="xml")
    expected = '<message type="human"></message>\n<message type="ai"></message>'
    assert result == expected


def test_get_buffer_string_xml_tool_calls_with_content() -> None:
    """Test XML format with `AIMessage` having both `content` and `tool_calls`."""
    messages = [
        AIMessage(
            content="Let me check that",
            tool_calls=[
                {
                    "name": "get_weather",
                    "args": {"city": "NYC"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    # Nested structure with content and tool_call elements
    expected = (
        '<message type="ai">\n'
        "  <content>Let me check that</content>\n"
        '  <tool_call id="call_1" name="get_weather">{"city": "NYC"}</tool_call>\n'
        "</message>"
    )
    assert result == expected


def test_get_buffer_string_xml_tool_calls_empty_content() -> None:
    """Test XML format with `AIMessage` having empty `content` and `tool_calls`."""
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search",
                    "args": {"query": "test"},
                    "id": "call_2",
                    "type": "tool_call",
                }
            ],
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    # No content element when content is empty
    expected = (
        '<message type="ai">\n'
        '  <tool_call id="call_2" name="search">{"query": "test"}</tool_call>\n'
        "</message>"
    )
    assert result == expected


def test_get_buffer_string_xml_tool_calls_escaping() -> None:
    """Test XML format escapes special characters in tool calls."""
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculate",
                    "args": {"expression": "5 < 10 & 10 > 5"},
                    "id": "call_3",
                    "type": "tool_call",
                }
            ],
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    # Special characters in tool_calls args should be escaped
    assert "&lt;" in result
    assert "&gt;" in result
    assert "&amp;" in result
    # Verify overall structure
    assert result.startswith('<message type="ai">')
    assert result.endswith("</message>")


def test_get_buffer_string_xml_function_call_legacy() -> None:
    """Test XML format with legacy `function_call` in `additional_kwargs`."""
    messages = [
        AIMessage(
            content="Calling function",
            additional_kwargs={
                "function_call": {"name": "test_fn", "arguments": '{"x": 1}'}
            },
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    # Nested structure with function_call element
    # Note: arguments is a string, so quotes inside are escaped
    expected = (
        '<message type="ai">\n'
        "  <content>Calling function</content>\n"
        '  <function_call name="test_fn">{"x": 1}</function_call>\n'
        "</message>"
    )
    assert result == expected


def test_get_buffer_string_xml_structured_content() -> None:
    """Test XML format with structured content (list content blocks)."""
    messages = [
        HumanMessage(content=[{"type": "text", "text": "Hello, world!"}]),
        AIMessage(content=[{"type": "text", "text": "Hi there!"}]),
    ]
    result = get_buffer_string(messages, format="xml")
    # message.text property should extract text from structured content
    expected = (
        '<message type="human">Hello, world!</message>\n'
        '<message type="ai">Hi there!</message>'
    )
    assert result == expected


def test_get_buffer_string_xml_multiline_content() -> None:
    """Test XML format with multiline content."""
    messages = [
        HumanMessage(content="Line 1\nLine 2\nLine 3"),
    ]
    result = get_buffer_string(messages, format="xml")
    expected = '<message type="human">Line 1\nLine 2\nLine 3</message>'
    assert result == expected


def test_get_buffer_string_xml_tool_calls_preferred_over_function_call() -> None:
    """Test that `tool_calls` takes precedence over legacy `function_call` in XML."""
    messages = [
        AIMessage(
            content="Calling tools",
            tool_calls=[
                {
                    "name": "modern_tool",
                    "args": {"key": "value"},
                    "id": "call_3",
                    "type": "tool_call",
                }
            ],
            additional_kwargs={
                "function_call": {"name": "legacy_function", "arguments": "{}"}
            },
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert "modern_tool" in result
    assert "legacy_function" not in result
    # Should use tool_call element, not function_call
    assert "<tool_call" in result
    assert "<function_call" not in result


def test_get_buffer_string_xml_multiple_tool_calls() -> None:
    """Test XML format with `AIMessage` having multiple `tool_calls`."""
    messages = [
        AIMessage(
            content="I'll help with that",
            tool_calls=[
                {
                    "name": "get_weather",
                    "args": {"city": "NYC"},
                    "id": "call_1",
                    "type": "tool_call",
                },
                {
                    "name": "get_time",
                    "args": {"timezone": "EST"},
                    "id": "call_2",
                    "type": "tool_call",
                },
            ],
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    # Should have nested structure with multiple tool_call elements
    expected = (
        '<message type="ai">\n'
        "  <content>I'll help with that</content>\n"
        '  <tool_call id="call_1" name="get_weather">{"city": "NYC"}</tool_call>\n'
        '  <tool_call id="call_2" name="get_time">{"timezone": "EST"}</tool_call>\n'
        "</message>"
    )
    assert result == expected


def test_get_buffer_string_xml_tool_call_special_chars_in_attrs() -> None:
    """Test that tool call attributes with quotes are properly escaped."""
    messages: list[BaseMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": 'search"with"quotes',
                    "args": {"query": "test"},
                    "id": 'call"id',
                    "type": "tool_call",
                },
            ],
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    # quoteattr uses single quotes when value contains double quotes
    assert "name='search\"with\"quotes'" in result
    assert "id='call\"id'" in result


def test_get_buffer_string_xml_tool_call_none_id() -> None:
    """Test that tool calls with `None` id are handled correctly."""
    messages: list[BaseMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search",
                    "args": {},
                    "id": None,
                    "type": "tool_call",
                },
            ],
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    # Should handle None by converting to empty string
    assert 'id=""' in result


def test_get_buffer_string_xml_function_call_special_chars_in_name() -> None:
    """Test that `function_call` name with quotes is properly escaped."""
    messages: list[BaseMessage] = [
        AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": 'func"name',
                    "arguments": "{}",
                }
            },
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    # quoteattr uses single quotes when value contains double quotes
    assert "name='func\"name'" in result


def test_get_buffer_string_invalid_format() -> None:
    """Test that invalid format values raise `ValueError`."""
    messages: list[BaseMessage] = [HumanMessage(content="Hello")]
    with pytest.raises(ValueError, match="Unrecognized format"):
        get_buffer_string(messages, format="xm")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unrecognized format"):
        get_buffer_string(messages, format="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unrecognized format"):
        get_buffer_string(messages, format="")  # type: ignore[arg-type]


def test_get_buffer_string_xml_image_url_block() -> None:
    """Test XML format with image content block containing URL."""
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {"type": "text", "text": "What is in this image?"},
                {"type": "image", "url": "https://example.com/image.png"},
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert '<message type="human">' in result
    assert "What is in this image?" in result
    assert '<image url="https://example.com/image.png" />' in result


def test_get_buffer_string_xml_image_file_id_block() -> None:
    """Test XML format with image content block containing `file_id`."""
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Describe this:"},
                {"type": "image", "file_id": "file-abc123"},
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert '<image file_id="file-abc123" />' in result


def test_get_buffer_string_xml_image_base64_skipped() -> None:
    """Test XML format skips image blocks with base64 data."""
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {"type": "text", "text": "What is this?"},
                {"type": "image", "base64": "iVBORw0KGgo...", "mime_type": "image/png"},
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert "What is this?" in result
    assert "base64" not in result
    assert "iVBORw0KGgo" not in result


def test_get_buffer_string_xml_image_data_url_skipped() -> None:
    """Test XML format skips image blocks with data: URLs."""
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Check this:"},
                {"type": "image", "url": "data:image/png;base64,iVBORw0KGgo..."},
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert "Check this:" in result
    assert "data:image" not in result


def test_get_buffer_string_xml_openai_image_url_block() -> None:
    """Test XML format with OpenAI-style `image_url` block."""
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Analyze this:"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/photo.jpg"},
                },
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert "Analyze this:" in result
    assert '<image url="https://example.com/photo.jpg" />' in result


def test_get_buffer_string_xml_openai_image_url_data_skipped() -> None:
    """Test XML format skips OpenAI-style `image_url` blocks with data: URLs."""
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {"type": "text", "text": "See this:"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."},
                },
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert "See this:" in result
    assert "data:image" not in result
    assert "/9j/4AAQ" not in result


def test_get_buffer_string_xml_audio_url_block() -> None:
    """Test XML format with audio content block containing URL."""
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Transcribe this:"},
                {"type": "audio", "url": "https://example.com/audio.mp3"},
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert "Transcribe this:" in result
    assert '<audio url="https://example.com/audio.mp3" />' in result


def test_get_buffer_string_xml_audio_base64_skipped() -> None:
    """Test XML format skips audio blocks with base64 data."""
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Listen:"},
                {"type": "audio", "base64": "UklGRi...", "mime_type": "audio/wav"},
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert "Listen:" in result
    assert "UklGRi" not in result


def test_get_buffer_string_xml_video_url_block() -> None:
    """Test XML format with video content block containing URL."""
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Describe this video:"},
                {"type": "video", "url": "https://example.com/video.mp4"},
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert "Describe this video:" in result
    assert '<video url="https://example.com/video.mp4" />' in result


def test_get_buffer_string_xml_video_base64_skipped() -> None:
    """Test XML format skips video blocks with base64 data."""
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Watch:"},
                {"type": "video", "base64": "AAAAFGZ0eXA...", "mime_type": "video/mp4"},
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert "Watch:" in result
    assert "AAAAFGZ0eXA" not in result


def test_get_buffer_string_xml_reasoning_block() -> None:
    """Test XML format with reasoning content block."""
    messages: list[BaseMessage] = [
        AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "Let me think about this..."},
                {"type": "text", "text": "The answer is 42."},
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert "<reasoning>Let me think about this...</reasoning>" in result
    assert "The answer is 42." in result


def test_get_buffer_string_xml_text_plain_block() -> None:
    """Test XML format with text-plain content block."""
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Here is a document:"},
                {
                    "type": "text-plain",
                    "text": "Document content here.",
                    "mime_type": "text/plain",
                },
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert "Here is a document:" in result
    assert "Document content here." in result


def test_get_buffer_string_xml_server_tool_call_block() -> None:
    """Test XML format with server_tool_call content block."""
    messages: list[BaseMessage] = [
        AIMessage(
            content=[
                {"type": "text", "text": "Let me search for that."},
                {
                    "type": "server_tool_call",
                    "id": "call_123",
                    "name": "web_search",
                    "args": {"query": "weather today"},
                },
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert "Let me search for that." in result
    assert '<server_tool_call id="call_123" name="web_search">' in result
    assert '{"query": "weather today"}' in result
    assert "</server_tool_call>" in result


def test_get_buffer_string_xml_server_tool_result_block() -> None:
    """Test XML format with server_tool_result content block."""
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {
                    "type": "server_tool_result",
                    "tool_call_id": "call_123",
                    "status": "success",
                    "output": {"temperature": 72, "conditions": "sunny"},
                },
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert '<server_tool_result tool_call_id="call_123" status="success">' in result
    assert '"temperature": 72' in result
    assert "</server_tool_result>" in result


def test_get_buffer_string_xml_unknown_block_type_skipped() -> None:
    """Test XML format silently skips unknown block types."""
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "unknown_type", "data": "some data"},
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert "Hello" in result
    assert "unknown_type" not in result
    assert "some data" not in result


def test_get_buffer_string_xml_mixed_content_blocks() -> None:
    """Test XML format with multiple different content block types."""
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Look at this image and document:"},
                {"type": "image", "url": "https://example.com/img.png"},
                {
                    "type": "text-plain",
                    "text": "Doc content",
                    "mime_type": "text/plain",
                },
                # This should be skipped (base64)
                {"type": "image", "base64": "abc123", "mime_type": "image/png"},
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert "Look at this image and document:" in result
    assert '<image url="https://example.com/img.png" />' in result
    assert "Doc content" in result
    assert "abc123" not in result


def test_get_buffer_string_xml_escaping_in_content_blocks() -> None:
    """Test that special XML characters are escaped in content blocks."""
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Is 5 < 10 & 10 > 5?"},
                {"type": "reasoning", "reasoning": "Let's check: <value> & </value>"},
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert "Is 5 &lt; 10 &amp; 10 &gt; 5?" in result
    assert "&lt;value&gt; &amp; &lt;/value&gt;" in result


def test_get_buffer_string_xml_url_with_special_chars() -> None:
    """Test that URLs with special characters are properly quoted."""
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {"type": "image", "url": "https://example.com/img?a=1&b=2"},
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    # quoteattr should handle the & in the URL
    assert "https://example.com/img?a=1&amp;b=2" in result


def test_get_buffer_string_xml_text_plain_truncation() -> None:
    """Test that text-plain content is truncated to 500 chars."""
    long_text = "x" * 600
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {"type": "text-plain", "text": long_text, "mime_type": "text/plain"},
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    # Should be truncated to 500 chars + "..."
    assert "x" * 500 + "..." in result
    assert "x" * 501 not in result


def test_get_buffer_string_xml_server_tool_call_args_truncation() -> None:
    """Test that server_tool_call args are truncated to 500 chars."""
    long_value = "y" * 600
    messages: list[BaseMessage] = [
        AIMessage(
            content=[
                {
                    "type": "server_tool_call",
                    "id": "call_1",
                    "name": "test_tool",
                    "args": {"data": long_value},
                },
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert "..." in result
    # The full 600-char value should not appear
    assert long_value not in result


def test_get_buffer_string_xml_server_tool_result_output_truncation() -> None:
    """Test that server_tool_result output is truncated to 500 chars."""
    long_output = "z" * 600
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {
                    "type": "server_tool_result",
                    "tool_call_id": "call_1",
                    "status": "success",
                    "output": {"result": long_output},
                },
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert "..." in result
    # The full 600-char value should not appear
    assert long_output not in result


def test_get_buffer_string_xml_no_truncation_under_limit() -> None:
    """Test that content under 500 chars is not truncated."""
    short_text = "a" * 400
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {"type": "text-plain", "text": short_text, "mime_type": "text/plain"},
            ]
        ),
    ]
    result = get_buffer_string(messages, format="xml")
    assert short_text in result
    assert "..." not in result


def test_get_buffer_string_custom_system_prefix() -> None:
    """Test `get_buffer_string` with custom `system_prefix`."""
    messages: list[BaseMessage] = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello"),
    ]
    result = get_buffer_string(messages, system_prefix="Instructions")
    assert result == "Instructions: You are a helpful assistant.\nHuman: Hello"


def test_get_buffer_string_custom_function_prefix() -> None:
    """Test `get_buffer_string` with custom `function_prefix`."""
    messages: list[BaseMessage] = [
        HumanMessage(content="Call a function"),
        FunctionMessage(name="test_func", content="Function result"),
    ]
    result = get_buffer_string(messages, function_prefix="Func")
    assert result == "Human: Call a function\nFunc: Function result"


def test_get_buffer_string_custom_tool_prefix() -> None:
    """Test `get_buffer_string` with custom `tool_prefix`."""
    messages: list[BaseMessage] = [
        HumanMessage(content="Use a tool"),
        ToolMessage(tool_call_id="call_123", content="Tool result"),
    ]
    result = get_buffer_string(messages, tool_prefix="ToolResult")
    assert result == "Human: Use a tool\nToolResult: Tool result"


def test_get_buffer_string_all_custom_prefixes() -> None:
    """Test `get_buffer_string` with all custom prefixes."""
    messages: list[BaseMessage] = [
        SystemMessage(content="System says hello"),
        HumanMessage(content="Human says hello"),
        AIMessage(content="AI says hello"),
        FunctionMessage(name="func", content="Function says hello"),
        ToolMessage(tool_call_id="call_1", content="Tool says hello"),
    ]
    result = get_buffer_string(
        messages,
        human_prefix="User",
        ai_prefix="Assistant",
        system_prefix="Sys",
        function_prefix="Fn",
        tool_prefix="T",
    )
    expected = (
        "Sys: System says hello\n"
        "User: Human says hello\n"
        "Assistant: AI says hello\n"
        "Fn: Function says hello\n"
        "T: Tool says hello"
    )
    assert result == expected


def test_get_buffer_string_xml_custom_system_prefix() -> None:
    """Test `get_buffer_string` XML format with custom `system_prefix`."""
    messages: list[BaseMessage] = [
        SystemMessage(content="You are a helpful assistant."),
    ]
    result = get_buffer_string(messages, system_prefix="Instructions", format="xml")
    assert (
        result == '<message type="instructions">You are a helpful assistant.</message>'
    )


def test_get_buffer_string_xml_custom_function_prefix() -> None:
    """Test `get_buffer_string` XML format with custom `function_prefix`."""
    messages: list[BaseMessage] = [
        FunctionMessage(name="test_func", content="Function result"),
    ]
    result = get_buffer_string(messages, function_prefix="Fn", format="xml")
    assert result == '<message type="fn">Function result</message>'


def test_get_buffer_string_xml_custom_tool_prefix() -> None:
    """Test `get_buffer_string` XML format with custom `tool_prefix`."""
    messages: list[BaseMessage] = [
        ToolMessage(tool_call_id="call_123", content="Tool result"),
    ]
    result = get_buffer_string(messages, tool_prefix="ToolOutput", format="xml")
    assert result == '<message type="tooloutput">Tool result</message>'


def test_get_buffer_string_xml_all_custom_prefixes() -> None:
    """Test `get_buffer_string` XML format with all custom prefixes."""
    messages: list[BaseMessage] = [
        SystemMessage(content="System message"),
        HumanMessage(content="Human message"),
        AIMessage(content="AI message"),
        FunctionMessage(name="func", content="Function message"),
        ToolMessage(tool_call_id="call_1", content="Tool message"),
    ]
    result = get_buffer_string(
        messages,
        human_prefix="User",
        ai_prefix="Assistant",
        system_prefix="Sys",
        function_prefix="Fn",
        tool_prefix="T",
        format="xml",
    )
    # The messages are processed in order, not by type
    assert '<message type="sys">System message</message>' in result
    assert '<message type="user">Human message</message>' in result
    assert '<message type="assistant">AI message</message>' in result
    assert '<message type="fn">Function message</message>' in result
    assert '<message type="t">Tool message</message>' in result


def test_count_tokens_approximately_with_image_content() -> None:
    """Test approximate token counting with image content blocks."""
    message_with_image = HumanMessage(
        content=[
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + "A" * 100000},
            },
        ]
    )

    token_count = count_tokens_approximately([message_with_image])

    # Should be ~85 (image) + ~5 (text) + 3 (extra) = ~93 tokens, NOT 25,000+
    assert token_count < 200, f"Expected <200 tokens, got {token_count}"
    assert token_count > 80, f"Expected >80 tokens, got {token_count}"


def test_count_tokens_approximately_with_multiple_images() -> None:
    """Test token counting with multiple images."""
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Compare these images"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBB"}},
        ]
    )

    token_count = count_tokens_approximately([message])

    # Should be ~85 * 2 (images) + ~6 (text) + 3 (extra) = ~179 tokens
    assert 170 < token_count < 190


def test_count_tokens_approximately_text_only_backward_compatible() -> None:
    """Test that text-only messages still work correctly."""
    messages = [
        HumanMessage(content="Hello world"),
        AIMessage(content="Hi there!"),
    ]

    token_count = count_tokens_approximately(messages)

    # Should be ~15 tokens
    # (11 chars + 9 chars + roles + 2*3 extra)
    assert 13 <= token_count <= 17


def test_count_tokens_approximately_with_custom_image_penalty() -> None:
    """Test custom tokens_per_image parameter."""
    message = HumanMessage(
        content=[
            {"type": "text", "text": "test"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,XYZ"}},
        ]
    )

    # Using custom image penalty (e.g., for Anthropic models)
    token_count = count_tokens_approximately([message], tokens_per_image=1600)

    # Should be ~1600 (image) + ~1 (text) + 3 (extra) = ~1604 tokens
    assert 1600 < token_count < 1610


def test_count_tokens_approximately_with_image_only_message() -> None:
    """Test token counting for a message that only contains an image."""
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,AAA"},
            }
        ]
    )

    token_count = count_tokens_approximately([message])

    # Should be roughly tokens_per_image + role + extra per message.
    # Default tokens_per_image is 85 and extra_tokens_per_message is 3,
    # so we expect something in the ~90-110 range.
    assert 80 < token_count < 120


def test_count_tokens_approximately_with_unknown_block_type() -> None:
    """Test that unknown multimodal block types still contribute to token count."""
    text_only = count_tokens_approximately([HumanMessage(content="hello")])

    message_with_unknown_block = HumanMessage(
        content=[
            {"type": "text", "text": "hello"},
            {"type": "foo", "bar": "baz"},  # unknown type, falls back to repr(block)
        ]
    )

    mixed = count_tokens_approximately([message_with_unknown_block])

    # The message with an extra unknown block should be counted as more expensive
    # than the text-only version.
    assert mixed > text_only


def test_count_tokens_approximately_ai_tool_calls_skipped_for_list_content() -> None:
    """Test that tool_calls aren't double-counted for list (Anthropic-style) content."""
    tool_calls = [
        {
            "id": "call_1",
            "name": "foo",
            "args": {"x": 1},
        }
    ]

    # Case 1: content is a string -> tool_calls should be added to the char count.
    ai_with_text_content = AIMessage(
        content="do something",
        tool_calls=tool_calls,
    )
    count_text = count_tokens_approximately([ai_with_text_content])

    # Case 2: content is a list (e.g. Anthropic-style blocks) -> tool_calls are
    # already represented in the content and should NOT be counted again.
    ai_with_list_content = AIMessage(
        content=[
            {"type": "text", "text": "do something"},
            {
                "type": "tool_use",
                "name": "foo",
                "input": {"x": 1},
                "id": "call_1",
            },
        ],
        tool_calls=tool_calls,
    )
    count_list = count_tokens_approximately([ai_with_list_content])

    assert count_text - 1 <= count_list <= count_text + 1


def test_count_tokens_approximately_respects_count_name_flag() -> None:
    """Test that the count_name flag controls whether names are included."""
    message = HumanMessage(content="hello", name="user-name")

    with_name = count_tokens_approximately([message], count_name=True)
    without_name = count_tokens_approximately([message], count_name=False)

    # When count_name is True, the name should contribute to the token count.
    assert with_name > without_name
