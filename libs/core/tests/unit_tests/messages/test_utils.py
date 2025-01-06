import base64
import json
import typing
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import pytest

from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.utils import (
    convert_to_messages,
    convert_to_openai_messages,
    filter_messages,
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
def test_filter_message(filters: dict) -> None:
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
        text_splitter=lambda x: list(x),
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
        text_splitter=lambda x: list(x),
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
        max_tokens=10, token_counter=FakeTokenCountingModel().bind(foo="bar")
    )
    trimmer.invoke([HumanMessage("foobar")])


def test_trim_messages_bad_token_counter() -> None:
    trimmer = trim_messages(max_tokens=10, token_counter={})
    with pytest.raises(ValueError):
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


class FakeTokenCountingModel(FakeChatModel):
    def get_num_tokens_from_messages(
        self,
        messages: list[BaseMessage],
        tools: Optional[
            Sequence[
                Union[typing.Dict[str, Any], type, Callable, BaseTool]  # noqa: UP006
            ]
        ] = None,
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
    with pytest.raises(ValueError):
        convert_to_messages([{"role": "assistant", "refusal": "9.1"}])


def create_image_data() -> str:
    return "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigD//2Q=="  # noqa: E501


def create_base64_image(format: str = "jpeg") -> str:
    data = create_image_data()
    return f"data:image/{format};base64,{data}"  # noqa: E501


def test_convert_to_openai_messages_single_message() -> None:
    message = HumanMessage(content="Hello")
    result = convert_to_openai_messages(message)
    assert result == {"role": "user", "content": "Hello"}


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
        convert_to_openai_messages(
            [HumanMessage(content="Hello")],
            text_format="invalid",  # type: ignore[arg-type]
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
        convert_to_openai_messages(messages, text_format="block")


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
    messages: list = [
        SystemMessage("a", additional_kwargs={"__openai_role__": "developer"}),
        {"role": "developer", "content": "a"},
    ]
    result = convert_to_openai_messages(messages)
    assert result == [{"role": "developer", "content": "a"}] * 2
