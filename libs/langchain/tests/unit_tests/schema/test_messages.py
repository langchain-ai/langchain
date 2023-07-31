from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    get_buffer_string,
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


def test_get_buffer_string() -> None:
    system_message = SystemMessage(content="You are a useful assistant for a human.")
    message1 = HumanMessage(content="Hi. What is the meaning of life?")
    message2 = AIMessage(
        content=(
            "It's a very tricky question, I don't know. Try be happy, "
            "passionate for something and do good stuff."
        )
    )
    message3 = HumanMessage(content="Why?")
    message4 = AIMessage(content="42")
    expected_history = (
        "System: You are a useful assistant for a human.\nHuman: Hi. "
        "What is the meaning of life?\nAI: It's a very tricky "
        "question, I don't know. Try be happy, passionate for "
        "something and do good stuff.\nHuman: Why?\nAI: 42"
    )
    assert (
        get_buffer_string([system_message, message1, message2, message3, message4])
        == expected_history
    )


def test_get_buffer_string_with_prefix() -> None:
    system_message = SystemMessage(content="You are a useful assistant for a human.")
    message1 = HumanMessage(content="Hi. What is the meaning of life?")
    message2 = AIMessage(
        content=(
            "It's a very tricky question, I don't know. Try be happy, "
            "passionate for something and do good stuff."
        )
    )
    message3 = HumanMessage(content="Why?")
    message4 = AIMessage(content="42")
    expected_history = (
        "System: You are a useful assistant for a human.\nhuman: Hi. "
        "What is the meaning of life?\nbot: It's a very tricky "
        "question, I don't know. Try be happy, passionate for "
        "something and do good stuff.\nhuman: Why?\nbot: 42"
    )
    history = get_buffer_string(
        [system_message, message1, message2, message3, message4],
        ai_prefix="bot",
        human_prefix="human",
    )
    assert history == expected_history


def test_get_buffer_string_format_system_message() -> None:
    message = (
        "You are a useful assistant for a human. Predict the next reply given "
        "the following history:\n\n{history}\nBe polite and helpful."
    )
    system_message = SystemMessage(content=message)
    message1 = HumanMessage(content="Hi. What is the meaning of life?")
    message2 = AIMessage(
        content=(
            "It's a very tricky question, I don't know. Try be happy, "
            "passionate for something and do good stuff."
        )
    )
    message3 = HumanMessage(content="Why?")
    message4 = AIMessage(content="42")
    expected_history = (
        "You are a useful assistant for a human. Predict the next reply given the "
        "following history:\n\nHuman: Hi. What is the meaning of life?\nAI: It's a "
        "very tricky question, I don't know. Try be happy, passionate for something "
        "and do good stuff.\nHuman: Why?\nAI: 42\nBe polite and helpful."
    )
    history = get_buffer_string(
        [system_message, message1, message2, message3, message4],
        system_message_format_key="history",
    )
    assert history == expected_history
