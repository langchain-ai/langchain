from langchain.schema.messages import (
    AIMessageChunk,
    ChatMessageChunk,
    HumanMessageChunk,
)


def test_message_chunks() -> None:
    assert AIMessageChunk(content="I am") + AIMessageChunk(
        content=" indeed."
    ) == AIMessageChunk(
        content="I am indeed."
    ), "MessageChunk + MessageChunk should be a MessageChunk"

    assert AIMessageChunk(content="I am") + HumanMessageChunk(
        content=" indeed."
    ) == AIMessageChunk(
        content="I am indeed."
    ), "MessageChunk + MessageChunk should be a MessageChunk of same class as the left side"  # noqa: E501

    assert AIMessageChunk(
        content="", additional_kwargs={"foo": "bar"}
    ) + AIMessageChunk(content="", additional_kwargs={"baz": "foo"}) == AIMessageChunk(
        content="", additional_kwargs={"foo": "bar", "baz": "foo"}
    ), "MessageChunk + MessageChunk should be a MessageChunk with merged additional_kwargs"  # noqa: E501

    assert AIMessageChunk(
        content="", additional_kwargs={"function_call": {"name": "web_search"}}
    ) + AIMessageChunk(
        content="", additional_kwargs={"function_call": {"arguments": "{\n"}}
    ) + AIMessageChunk(
        content="",
        additional_kwargs={"function_call": {"arguments": '  "query": "turtles"\n}'}},
    ) == AIMessageChunk(
        content="",
        additional_kwargs={
            "function_call": {
                "name": "web_search",
                "arguments": '{\n  "query": "turtles"\n}',
            }
        },
    ), "MessageChunk + MessageChunk should be a MessageChunk with merged additional_kwargs"  # noqa: E501


def test_chat_message_chunks() -> None:
    assert ChatMessageChunk(role="User", content="I am") + ChatMessageChunk(
        role="User", content=" indeed."
    ) == ChatMessageChunk(
        role="User", content="I am indeed."
    ), "ChatMessageChunk + ChatMessageChunk should be a ChatMessageChunk"

    assert ChatMessageChunk(role="User", content="I am") + ChatMessageChunk(
        role="Assistant", content=" indeed."
    ) == ChatMessageChunk(
        role="User", content="I am indeed."
    ), "ChatMessageChunk + ChatMessageChunk with different role should be a ChatMessageChunk of same role as the left side"  # noqa: E501

    assert ChatMessageChunk(role="User", content="I am") + AIMessageChunk(
        content=" indeed."
    ) == ChatMessageChunk(
        role="User", content="I am indeed."
    ), "ChatMessageChunk + other MessageChunk should be a ChatMessageChunk with the left side's role"  # noqa: E501

    assert AIMessageChunk(content="I am") + ChatMessageChunk(
        role="User", content=" indeed."
    ) == AIMessageChunk(
        content="I am indeed."
    ), "Other MessageChunk + ChatMessageChunk should be a MessageChunk as the left side"  # noqa: E501
