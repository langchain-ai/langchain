from langchain.schema.messages import AIMessageChunk, HumanMessageChunk, ChatMessageChunk


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

    assert (ChatMessageChunk(content="Hello ", role="user")
            + ChatMessageChunk(content="World", role="user")
            == ChatMessageChunk(content="Hello World", role="user")), \
        "ChatMessageChunk + ChatMessageChunk should be a ChatMessageChunk"

    assert (ChatMessageChunk(content="I am ", role="ai")
            + HumanMessageChunk(content="user")
            == ChatMessageChunk(content="I am user", role="ai")), \
        "Message Chunk + Message Chunk should be a MessageChunk of same class as the left side"

    assert (ChatMessageChunk(content="", role="ai")
            + ChatMessageChunk(content="", role="user")
            == ChatMessageChunk(content="", role="ai")), \
        "ChatMessageChunk + ChatMessageChunk should be a ChatMessageChunk of same role field as left side"

    assert ChatMessageChunk(
        content="", role="user", additional_kwargs={"function_call": {"name": "web_search"}}
    ) + ChatMessageChunk(
        content="", role="user", additional_kwargs={"function_call": {"arguments": "{\n"}}
    ) + ChatMessageChunk(
        content="",
        role="user",
        additional_kwargs={"function_call": {"arguments": '  "query": "turtles"\n}'}},
    ) == ChatMessageChunk(
        content="",
        role="user",
        additional_kwargs={
            "function_call": {
                "name": "web_search",
                "arguments": '{\n  "query": "turtles"\n}',
            }
        },
    ), "MessageChunk + MessageChunk should be a MessageChunk with merged additional_kwargs"
