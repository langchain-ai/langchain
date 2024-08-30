import json
from typing import Any, Callable, Dict, Iterator, List, Type

import pytest

from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.utils import (
    _bytes_to_b64_str,
    convert_to_messages,
    filter_messages,
    format_messages_as,
    merge_message_runs,
    trim_messages,
)
from langchain_core.runnables import RunnableLambda


@pytest.mark.parametrize("msg_cls", [HumanMessage, AIMessage, SystemMessage])
def test_merge_message_runs_str(msg_cls: Type[BaseMessage]) -> None:
    messages = [msg_cls("foo"), msg_cls("bar"), msg_cls("baz")]
    messages_copy = [m.copy(deep=True) for m in messages]
    expected = [msg_cls("foo\nbar\nbaz")]
    actual = merge_message_runs(messages)
    assert actual == expected
    assert messages == messages_copy


@pytest.mark.parametrize("msg_cls", [HumanMessage, AIMessage, SystemMessage])
def test_merge_message_runs_str_with_specified_separator(
    msg_cls: Type[BaseMessage],
) -> None:
    messages = [msg_cls("foo"), msg_cls("bar"), msg_cls("baz")]
    messages_copy = [m.copy(deep=True) for m in messages]
    expected = [msg_cls("foo<sep>bar<sep>baz")]
    actual = merge_message_runs(messages, chunk_separator="<sep>")
    assert actual == expected
    assert messages == messages_copy


@pytest.mark.parametrize("msg_cls", [HumanMessage, AIMessage, SystemMessage])
def test_merge_message_runs_str_without_separator(
    msg_cls: Type[BaseMessage],
) -> None:
    messages = [msg_cls("foo"), msg_cls("bar"), msg_cls("baz")]
    messages_copy = [m.copy(deep=True) for m in messages]
    expected = [msg_cls("foobarbaz")]
    actual = merge_message_runs(messages, chunk_separator="")
    assert actual == expected
    assert messages == messages_copy


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
    messages_copy = [m.copy(deep=True) for m in messages]
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
    assert messages == messages_copy


def test_merge_messages_tool_messages() -> None:
    messages = [
        ToolMessage("foo", tool_call_id="1"),
        ToolMessage("bar", tool_call_id="2"),
    ]
    messages_copy = [m.copy(deep=True) for m in messages]
    actual = merge_message_runs(messages)
    assert actual == messages
    assert messages == messages_copy


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
def test_filter_message(filters: Dict) -> None:
    messages = [
        SystemMessage("foo", name="blah", id="1"),
        HumanMessage("bar", name="blur", id="2"),
    ]
    messages_copy = [m.copy(deep=True) for m in messages]
    expected = messages[1:2]
    actual = filter_messages(messages, **filters)
    assert expected == actual
    invoked = filter_messages(**filters).invoke(messages)
    assert invoked == actual
    assert messages == messages_copy


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
_MESSAGES_TO_TRIM_COPY = [m.copy(deep=True) for m in _MESSAGES_TO_TRIM]


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


def test_trim_messages_allow_partial_text_splitter() -> None:
    expected = [
        HumanMessage("a 4 token text.", id="third"),
        AIMessage("This is a 4 token text.", id="fourth"),
    ]

    def count_words(msgs: List[BaseMessage]) -> int:
        count = 0
        for msg in msgs:
            if isinstance(msg.content, str):
                count += len(msg.content.split(" "))
            else:
                count += len(
                    " ".join(block["text"] for block in msg.content).split(" ")  # type: ignore[index]
                )
        return count

    def _split_on_space(text: str) -> List[str]:
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
        max_tokens=10, token_counter=FakeTokenCountingModel().bind(foo="bar")
    )
    trimmer.invoke([HumanMessage("foobar")])


def test_trim_messages_bad_token_counter() -> None:
    trimmer = trim_messages(max_tokens=10, token_counter={})
    with pytest.raises(ValueError):
        trimmer.invoke([HumanMessage("foobar")])


def dummy_token_counter(messages: List[BaseMessage]) -> int:
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


class FakeTokenCountingModel(FakeChatModel):
    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        return dummy_token_counter(messages)


def test_convert_to_messages() -> None:
    message_like: List = [
        # BaseMessage
        SystemMessage("1"),
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


@pytest.mark.xfail(reason="AI message does not support refusal key yet.")
def test_convert_to_messages_openai_refusal() -> None:
    convert_to_messages([{"role": "assistant", "refusal": "9.1"}])


def create_base64_image(format: str = "jpeg") -> str:
    return f"data:image/{format};base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigD//2Q=="  # noqa: E501


def test_format_messages_as_single_message() -> None:
    message = HumanMessage(content="Hello")
    result = format_messages_as(message, format="openai", text_format="string")
    assert isinstance(result, BaseMessage)
    assert result.content == "Hello"


def test_format_messages_as_multiple_messages() -> None:
    messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="Human message"),
        AIMessage(content="AI message"),
    ]
    result = format_messages_as(messages, format="openai", text_format="string")
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(msg, BaseMessage) for msg in result)
    assert [msg.content for msg in result] == [
        "System message",
        "Human message",
        "AI message",
    ]


def test_format_messages_as_openai_string() -> None:
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
    result = format_messages_as(messages, format="openai", text_format="string")
    assert [msg.content for msg in result] == ["Hello\nWorld", "Hi\nthere"]


def test_format_messages_as_openai_block() -> None:
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
    ]
    result = format_messages_as(messages, format="openai", text_format="block")
    assert [msg.content for msg in result] == [
        [{"type": "text", "text": "Hello"}],
        [{"type": "text", "text": "Hi there"}],
    ]


def test_format_messages_as_anthropic_string() -> None:
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
    result = format_messages_as(messages, format="anthropic", text_format="string")
    assert [msg.content for msg in result] == ["Hello\nWorld", "Hi\nthere"]


def test_format_messages_as_anthropic_block() -> None:
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
    ]
    result = format_messages_as(messages, format="anthropic", text_format="block")
    assert [msg.content for msg in result] == [
        [{"type": "text", "text": "Hello"}],
        [{"type": "text", "text": "Hi there"}],
    ]


def test_format_messages_as_invalid_format() -> None:
    with pytest.raises(ValueError, match="Unrecognized format="):
        format_messages_as(
            [HumanMessage(content="Hello")], format="invalid", text_format="string"
        )


def test_format_messages_as_openai_image() -> None:
    base64_image = create_base64_image()
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Here's an image:"},
                {"type": "image_url", "image_url": {"url": base64_image}},
            ]
        )
    ]
    result = format_messages_as(messages, format="openai", text_format="block")
    assert result[0].content[1]["type"] == "image_url"
    assert result[0].content[1]["image_url"]["url"] == base64_image


def test_format_messages_as_anthropic_image() -> None:
    base64_image = create_base64_image()
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Here's an image:"},
                {"type": "image_url", "image_url": {"url": base64_image}},
            ]
        )
    ]
    result = format_messages_as(messages, format="anthropic", text_format="block")
    assert result[0].content[1]["type"] == "image"
    assert result[0].content[1]["source"]["type"] == "base64"
    assert result[0].content[1]["source"]["media_type"] == "image/jpeg"


def test_format_messages_as_tool_message() -> None:
    tool_message = ToolMessage(content="Tool result", tool_call_id="123")
    result = format_messages_as([tool_message], format="openai", text_format="block")
    assert isinstance(result[0], ToolMessage)
    assert result[0].content == [{"type": "text", "text": "Tool result"}]
    assert result[0].tool_call_id == "123"


def test_format_messages_as_tool_use() -> None:
    messages = [
        AIMessage(
            content=[
                {"type": "tool_use", "id": "123", "name": "calculator", "input": "2+2"}
            ]
        )
    ]
    result = format_messages_as(messages, format="openai", text_format="block")
    assert result[0].tool_calls[0]["id"] == "123"
    assert result[0].tool_calls[0]["name"] == "calculator"
    assert result[0].tool_calls[0]["args"] == "2+2"


def test_format_messages_as_json() -> None:
    json_data = {"key": "value"}
    messages = [HumanMessage(content=[{"type": "json", "json": json_data}])]
    result = format_messages_as(messages, format="openai", text_format="block")
    assert result[0].content[0]["type"] == "text"
    assert json.loads(result[0].content[0]["text"]) == json_data


def test_format_messages_as_guard_content() -> None:
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
    result = format_messages_as(messages, format="openai", text_format="block")
    assert result[0].content[0]["type"] == "text"
    assert result[0].content[0]["text"] == "Protected content"


def test_format_messages_as_vertexai_image() -> None:
    messages = [
        HumanMessage(
            content=[
                {"type": "media", "mime_type": "image/jpeg", "data": b"image_bytes"}
            ]
        )
    ]
    result = format_messages_as(messages, format="openai", text_format="block")
    assert result[0].content[0]["type"] == "image_url"
    assert (
        result[0].content[0]["image_url"]["url"]
        == f"data:image/jpeg;base64,{_bytes_to_b64_str(b'image_bytes')}"
    )


def test_format_messages_as_invalid_block() -> None:
    messages = [HumanMessage(content=[{"type": "invalid", "foo": "bar"}])]
    with pytest.raises(ValueError, match="Unrecognized content block"):
        format_messages_as(messages, format="openai", text_format="block")
    with pytest.raises(ValueError, match="Unrecognized content block"):
        format_messages_as(messages, format="anthropic", text_format="block")


def test_format_messages_as_empty_message() -> None:
    result = format_messages_as(
        HumanMessage(content=""), format="openai", text_format="string"
    )
    assert result.content == ""


def test_format_messages_as_empty_list() -> None:
    result = format_messages_as([], format="openai", text_format="string")
    assert result == []


def test_format_messages_as_mixed_content_types() -> None:
    messages = [
        HumanMessage(
            content=[
                "Text message",
                {"type": "text", "text": "Structured text"},
                {"type": "image_url", "image_url": create_base64_image()},
            ]
        )
    ]
    result = format_messages_as(messages, format="openai", text_format="block")
    assert len(result[0].content) == 3
    assert isinstance(result[0].content[0], dict)
    assert isinstance(result[0].content[1], dict)
    assert isinstance(result[0].content[2], dict)


def test_format_messages_as_anthropic_tool_calls() -> None:
    message = AIMessage(
        "blah",
        tool_calls=[
            {"type": "tool_call", "name": "foo", "id": "1", "args": {"bar": "baz"}}
        ],
    )
    result = format_messages_as(message, format="anthropic", text_format="string")
    assert result.content == [
        {"type": "text", "text": "blah"},
        {"type": "tool_use", "id": "1", "name": "foo", "input": {"bar": "baz"}},
    ]
    assert result.tool_calls == message.tool_calls


def test_format_messages_as_declarative() -> None:
    formatter = format_messages_as(format="openai", text_format="block")
    base64_image = create_base64_image()
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Here's an image:"},
                {"type": "image_url", "image_url": {"url": base64_image}},
            ]
        )
    ]
    result = formatter.invoke(messages)
    assert result[0].content[1]["type"] == "image_url"
    assert result[0].content[1]["image_url"]["url"] == base64_image


def _stream_oai(input_: Any) -> Iterator:
    chunks = [
        AIMessageChunk(content=""),
        AIMessageChunk(content="Certainly"),
        AIMessageChunk(content="!"),
        AIMessageChunk(
            content="",
            tool_calls=[
                {
                    "name": "multiply",
                    "args": {},
                    "id": "call_hr4yN4ZN7zv9Vmc5Cuahp4K8",
                    "type": "tool_call",
                }
            ],
            tool_call_chunks=[
                {
                    "name": "multiply",
                    "args": "",
                    "id": "call_hr4yN4ZN7zv9Vmc5Cuahp4K8",
                    "index": 0,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content="",
            tool_calls=[{"name": "", "args": {}, "id": None, "type": "tool_call"}],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": '{"',
                    "id": None,
                    "index": 0,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content="",
            invalid_tool_calls=[
                {
                    "name": None,
                    "args": 'a": 5, "b": 2',
                    "id": None,
                    "error": None,
                    "type": "invalid_tool_call",
                }
            ],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": 'a": 5, "b": 2',
                    "id": None,
                    "index": 0,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content="",
            invalid_tool_calls=[
                {
                    "name": None,
                    "args": "}",
                    "id": None,
                    "error": None,
                    "type": "invalid_tool_call",
                }
            ],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": "}",
                    "id": None,
                    "index": 0,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content="",
        ),
    ]
    yield from chunks


def _stream_anthropic(input_: Any) -> Iterator:
    chunks = [
        AIMessageChunk(content=[]),
        AIMessageChunk(content=[{"text": "Certainly", "type": "text", "index": 0}]),
        AIMessageChunk(content=[{"text": "!", "type": "text", "index": 0}]),
        AIMessageChunk(
            content=[
                {
                    "id": "call_hr4yN4ZN7zv9Vmc5Cuahp4K8",
                    "input": {},
                    "name": "multiply",
                    "type": "tool_use",
                    "index": 1,
                }
            ],
            tool_calls=[
                {
                    "name": "multiply",
                    "args": {},
                    "id": "call_hr4yN4ZN7zv9Vmc5Cuahp4K8",
                    "type": "tool_call",
                }
            ],
            tool_call_chunks=[
                {
                    "name": "multiply",
                    "args": "",
                    "id": "call_hr4yN4ZN7zv9Vmc5Cuahp4K8",
                    "index": 1,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content=[{"partial_json": '{"', "type": "tool_use", "index": 1}],
            tool_calls=[{"name": "", "args": {}, "id": None, "type": "tool_call"}],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": '{"',
                    "id": None,
                    "index": 1,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content=[{"partial_json": 'a": 5, "b": 2', "type": "tool_use", "index": 1}],
            invalid_tool_calls=[
                {
                    "name": None,
                    "args": 'a": 5, "b": 2',
                    "id": None,
                    "error": None,
                    "type": "invalid_tool_call",
                }
            ],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": 'a": 5, "b": 2',
                    "id": None,
                    "index": 1,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content=[{"partial_json": "}", "type": "tool_use", "index": 1}],
            invalid_tool_calls=[
                {
                    "name": None,
                    "args": "}",
                    "id": None,
                    "error": None,
                    "type": "invalid_tool_call",
                }
            ],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": "}",
                    "id": None,
                    "index": 1,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(content=""),
    ]
    yield from chunks


@pytest.mark.parametrize("stream", [_stream_oai, _stream_anthropic])
def test_format_messages_openai_string_stream(stream: Callable) -> None:
    formatter = format_messages_as(format="openai", text_format="string")

    chain = RunnableLambda(stream) | formatter
    tool_call_idx = 1 if stream == _stream_anthropic else 0
    expected = [
        AIMessageChunk(content=""),
        AIMessageChunk(content="Certainly"),
        AIMessageChunk(content="!"),
        AIMessageChunk(
            content="",
            tool_calls=[
                {
                    "name": "multiply",
                    "args": {},
                    "id": "call_hr4yN4ZN7zv9Vmc5Cuahp4K8",
                    "type": "tool_call",
                }
            ],
            tool_call_chunks=[
                {
                    "name": "multiply",
                    "args": "",
                    "id": "call_hr4yN4ZN7zv9Vmc5Cuahp4K8",
                    "index": tool_call_idx,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content="",
            tool_calls=[{"name": "", "args": {}, "id": None, "type": "tool_call"}],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": '{"',
                    "id": None,
                    "index": tool_call_idx,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content="",
            invalid_tool_calls=[
                {
                    "name": None,
                    "args": 'a": 5, "b": 2',
                    "id": None,
                    "error": None,
                    "type": "invalid_tool_call",
                }
            ],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": 'a": 5, "b": 2',
                    "id": None,
                    "index": tool_call_idx,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content="",
            invalid_tool_calls=[
                {
                    "name": None,
                    "args": "}",
                    "id": None,
                    "error": None,
                    "type": "invalid_tool_call",
                }
            ],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": "}",
                    "id": None,
                    "index": tool_call_idx,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content="",
        ),
    ]

    actual = list(chain.stream({}))
    assert expected == actual


@pytest.mark.parametrize("stream", [_stream_oai, _stream_anthropic])
def test_format_messages_openai_block_stream(stream: Callable) -> None:
    formatter = format_messages_as(format="openai", text_format="block")

    chain = RunnableLambda(stream) | formatter
    tool_call_idx = 1 if stream == _stream_anthropic else 0
    expected = [
        AIMessageChunk(content=[]),
        AIMessageChunk(content=[{"type": "text", "text": "Certainly"}]),
        AIMessageChunk(content=[{"type": "text", "text": "!"}]),
        AIMessageChunk(
            content=[],
            tool_calls=[
                {
                    "name": "multiply",
                    "args": {},
                    "id": "call_hr4yN4ZN7zv9Vmc5Cuahp4K8",
                    "type": "tool_call",
                }
            ],
            tool_call_chunks=[
                {
                    "name": "multiply",
                    "args": "",
                    "id": "call_hr4yN4ZN7zv9Vmc5Cuahp4K8",
                    "index": tool_call_idx,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content=[],
            tool_calls=[{"name": "", "args": {}, "id": None, "type": "tool_call"}],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": '{"',
                    "id": None,
                    "index": tool_call_idx,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content=[],
            invalid_tool_calls=[
                {
                    "name": None,
                    "args": 'a": 5, "b": 2',
                    "id": None,
                    "error": None,
                    "type": "invalid_tool_call",
                }
            ],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": 'a": 5, "b": 2',
                    "id": None,
                    "index": tool_call_idx,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content=[],
            invalid_tool_calls=[
                {
                    "name": None,
                    "args": "}",
                    "id": None,
                    "error": None,
                    "type": "invalid_tool_call",
                }
            ],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": "}",
                    "id": None,
                    "index": tool_call_idx,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content=[],
        ),
    ]
    actual = list(chain.stream({}))
    assert expected == actual


@pytest.mark.parametrize("stream", [_stream_oai, _stream_anthropic])
def test_format_messages_anthropic_block_stream(stream: Callable) -> None:
    formatter = format_messages_as(format="anthropic", text_format="block")

    chain = RunnableLambda(stream) | formatter
    expected = [
        AIMessageChunk(content=[]),
        AIMessageChunk(content=[{"text": "Certainly", "type": "text", "index": 0}]),
        AIMessageChunk(content=[{"text": "!", "type": "text", "index": 0}]),
        AIMessageChunk(
            content=[
                {
                    "id": "call_hr4yN4ZN7zv9Vmc5Cuahp4K8",
                    "name": "multiply",
                    "type": "tool_use",
                    "index": 1,
                    **(
                        {"input": {}}
                        if stream == _stream_anthropic
                        else {"partial_json": ""}
                    ),
                }
            ],
            tool_calls=[
                {
                    "name": "multiply",
                    "args": {},
                    "id": "call_hr4yN4ZN7zv9Vmc5Cuahp4K8",
                    "type": "tool_call",
                }
            ],
            tool_call_chunks=[
                {
                    "name": "multiply",
                    "args": "",
                    "id": "call_hr4yN4ZN7zv9Vmc5Cuahp4K8",
                    "index": 1,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content=[{"partial_json": '{"', "type": "tool_use", "index": 1}],
            tool_calls=[{"name": "", "args": {}, "id": None, "type": "tool_call"}],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": '{"',
                    "id": None,
                    "index": 1,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content=[{"partial_json": 'a": 5, "b": 2', "type": "tool_use", "index": 1}],
            invalid_tool_calls=[
                {
                    "name": None,
                    "args": 'a": 5, "b": 2',
                    "id": None,
                    "error": None,
                    "type": "invalid_tool_call",
                }
            ],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": 'a": 5, "b": 2',
                    "id": None,
                    "index": 1,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content=[{"partial_json": "}", "type": "tool_use", "index": 1}],
            invalid_tool_calls=[
                {
                    "name": None,
                    "args": "}",
                    "id": None,
                    "error": None,
                    "type": "invalid_tool_call",
                }
            ],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": "}",
                    "id": None,
                    "index": 1,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(content=[]),
    ]
    actual = list(chain.stream({}))
    assert expected == actual


@pytest.mark.parametrize("stream", [_stream_oai, _stream_anthropic])
def test_format_messages_anthropic_string_stream(stream: Callable) -> None:
    formatter = format_messages_as(format="anthropic", text_format="string")

    chain = RunnableLambda(stream) | formatter
    expected = [
        AIMessageChunk(content=""),
        AIMessageChunk(content="Certainly"),
        AIMessageChunk(content="!"),
        AIMessageChunk(
            content=[
                {
                    "type": "tool_use",
                    "id": "call_hr4yN4ZN7zv9Vmc5Cuahp4K8",
                    "name": "multiply",
                    "index": 1,
                    **(
                        {"input": {}}
                        if stream == _stream_anthropic
                        else {"partial_json": ""}
                    ),
                },
            ],
            tool_calls=[
                {
                    "name": "multiply",
                    "args": {},
                    "id": "call_hr4yN4ZN7zv9Vmc5Cuahp4K8",
                    "type": "tool_call",
                }
            ],
            tool_call_chunks=[
                {
                    "name": "multiply",
                    "args": "",
                    "id": "call_hr4yN4ZN7zv9Vmc5Cuahp4K8",
                    "index": 1,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content=[
                {"type": "tool_use", "partial_json": '{"', "index": 1},
            ],
            tool_calls=[{"name": "", "args": {}, "id": None, "type": "tool_call"}],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": '{"',
                    "id": None,
                    "index": 1,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content=[
                {"type": "tool_use", "partial_json": 'a": 5, "b": 2', "index": 1},
            ],
            invalid_tool_calls=[
                {
                    "name": None,
                    "args": 'a": 5, "b": 2',
                    "id": None,
                    "error": None,
                    "type": "invalid_tool_call",
                }
            ],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": 'a": 5, "b": 2',
                    "id": None,
                    "index": 1,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content=[
                {"type": "tool_use", "partial_json": "}", "index": 1},
            ],
            invalid_tool_calls=[
                {
                    "name": None,
                    "args": "}",
                    "id": None,
                    "error": None,
                    "type": "invalid_tool_call",
                }
            ],
            tool_call_chunks=[
                {
                    "name": None,
                    "args": "}",
                    "id": None,
                    "index": 1,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(content=""),
    ]
    actual = list(chain.stream({}))
    assert expected == actual
