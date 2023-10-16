from pathlib import Path
from typing import Any, List, Union

import pytest

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    BaseMessagePromptTemplate,
    ChatMessage,
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    ChatPromptValue,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    _convert_to_message,
)
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)


def create_messages() -> List[BaseMessagePromptTemplate]:
    """Create messages."""
    system_message_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template="Here's some context: {context}",
            input_variables=["context"],
        )
    )
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="Hello {foo}, I'm {bar}. Thanks for the {context}",
            input_variables=["foo", "bar", "context"],
        )
    )
    ai_message_prompt = AIMessagePromptTemplate(
        prompt=PromptTemplate(
            template="I'm an AI. I'm {foo}. I'm {bar}.",
            input_variables=["foo", "bar"],
        )
    )
    chat_message_prompt = ChatMessagePromptTemplate(
        role="test",
        prompt=PromptTemplate(
            template="I'm a generic message. I'm {foo}. I'm {bar}.",
            input_variables=["foo", "bar"],
        ),
    )
    return [
        system_message_prompt,
        human_message_prompt,
        ai_message_prompt,
        chat_message_prompt,
    ]


def create_chat_prompt_template() -> ChatPromptTemplate:
    """Create a chat prompt template."""
    return ChatPromptTemplate(
        input_variables=["foo", "bar", "context"],
        messages=create_messages(),
    )


def test_create_chat_prompt_template_from_template() -> None:
    """Create a chat prompt template."""
    prompt = ChatPromptTemplate.from_template("hi {foo} {bar}")
    assert prompt.messages == [
        HumanMessagePromptTemplate.from_template("hi {foo} {bar}")
    ]


def test_create_chat_prompt_template_from_template_partial() -> None:
    """Create a chat prompt template with partials."""
    prompt = ChatPromptTemplate.from_template(
        "hi {foo} {bar}", partial_variables={"foo": "jim"}
    )
    expected_prompt = PromptTemplate(
        template="hi {foo} {bar}",
        input_variables=["bar"],
        partial_variables={"foo": "jim"},
    )
    assert len(prompt.messages) == 1
    output_prompt = prompt.messages[0]
    assert isinstance(output_prompt, HumanMessagePromptTemplate)
    assert output_prompt.prompt == expected_prompt


def test_message_prompt_template_from_template_file() -> None:
    expected = ChatMessagePromptTemplate(
        prompt=PromptTemplate(
            template="Question: {question}\nAnswer:", input_variables=["question"]
        ),
        role="human",
    )
    actual = ChatMessagePromptTemplate.from_template_file(
        Path(__file__).parent.parent / "data" / "prompt_file.txt",
        ["question"],
        role="human",
    )
    assert expected == actual


def test_chat_prompt_template() -> None:
    """Test chat prompt template."""
    prompt_template = create_chat_prompt_template()
    prompt = prompt_template.format_prompt(foo="foo", bar="bar", context="context")
    assert isinstance(prompt, ChatPromptValue)
    messages = prompt.to_messages()
    assert len(messages) == 4
    assert messages[0].content == "Here's some context: context"
    assert messages[1].content == "Hello foo, I'm bar. Thanks for the context"
    assert messages[2].content == "I'm an AI. I'm foo. I'm bar."
    assert messages[3].content == "I'm a generic message. I'm foo. I'm bar."

    string = prompt.to_string()
    expected = (
        "System: Here's some context: context\n"
        "Human: Hello foo, I'm bar. Thanks for the context\n"
        "AI: I'm an AI. I'm foo. I'm bar.\n"
        "test: I'm a generic message. I'm foo. I'm bar."
    )
    assert string == expected

    string = prompt_template.format(foo="foo", bar="bar", context="context")
    assert string == expected


def test_chat_prompt_template_from_messages() -> None:
    """Test creating a chat prompt template from messages."""
    chat_prompt_template = ChatPromptTemplate.from_messages(create_messages())
    assert sorted(chat_prompt_template.input_variables) == sorted(
        ["context", "foo", "bar"]
    )
    assert len(chat_prompt_template.messages) == 4


def test_chat_prompt_template_from_messages_using_role_strings() -> None:
    """Test creating a chat prompt template from role string messages."""
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI bot. Your name is {name}."),
            ("human", "Hello, how are you doing?"),
            ("ai", "I'm doing well, thanks!"),
            ("human", "{user_input}"),
        ]
    )

    messages = template.format_messages(name="Bob", user_input="What is your name?")

    assert messages == [
        SystemMessage(
            content="You are a helpful AI bot. Your name is Bob.", additional_kwargs={}
        ),
        HumanMessage(
            content="Hello, how are you doing?", additional_kwargs={}, example=False
        ),
        AIMessage(
            content="I'm doing well, thanks!", additional_kwargs={}, example=False
        ),
        HumanMessage(content="What is your name?", additional_kwargs={}, example=False),
    ]


def test_chat_prompt_template_with_messages() -> None:
    messages: List[
        Union[BaseMessagePromptTemplate, BaseMessage]
    ] = create_messages() + [HumanMessage(content="foo")]
    chat_prompt_template = ChatPromptTemplate.from_messages(messages)
    assert sorted(chat_prompt_template.input_variables) == sorted(
        ["context", "foo", "bar"]
    )
    assert len(chat_prompt_template.messages) == 5
    prompt_value = chat_prompt_template.format_prompt(
        context="see", foo="this", bar="magic"
    )
    prompt_value_messages = prompt_value.to_messages()
    assert prompt_value_messages[-1] == HumanMessage(content="foo")


def test_chat_invalid_input_variables_extra() -> None:
    messages = [HumanMessage(content="foo")]
    with pytest.raises(ValueError):
        ChatPromptTemplate(messages=messages, input_variables=["foo"])


def test_chat_invalid_input_variables_missing() -> None:
    messages = [HumanMessagePromptTemplate.from_template("{foo}")]
    with pytest.raises(ValueError):
        ChatPromptTemplate(messages=messages, input_variables=[])


def test_infer_variables() -> None:
    messages = [HumanMessagePromptTemplate.from_template("{foo}")]
    prompt = ChatPromptTemplate(messages=messages)
    assert prompt.input_variables == ["foo"]


def test_chat_valid_with_partial_variables() -> None:
    messages = [
        HumanMessagePromptTemplate.from_template(
            "Do something with {question} using {context} giving it like {formatins}"
        )
    ]
    prompt = ChatPromptTemplate(
        messages=messages,
        input_variables=["question", "context"],
        partial_variables={"formatins": "some structure"},
    )
    assert set(prompt.input_variables) == {"question", "context"}
    assert prompt.partial_variables == {"formatins": "some structure"}


def test_chat_valid_infer_variables() -> None:
    messages = [
        HumanMessagePromptTemplate.from_template(
            "Do something with {question} using {context} giving it like {formatins}"
        )
    ]
    prompt = ChatPromptTemplate(
        messages=messages, partial_variables={"formatins": "some structure"}
    )
    assert set(prompt.input_variables) == {"question", "context"}
    assert prompt.partial_variables == {"formatins": "some structure"}


def test_chat_from_role_strings() -> None:
    """Test instantiation of chat template from role strings."""
    template = ChatPromptTemplate.from_role_strings(
        [
            ("system", "You are a bot."),
            ("assistant", "hello!"),
            ("human", "{question}"),
            ("other", "{quack}"),
        ]
    )

    messages = template.format_messages(question="How are you?", quack="duck")
    assert messages == [
        ChatMessage(content="You are a bot.", role="system"),
        ChatMessage(content="hello!", role="assistant"),
        ChatMessage(content="How are you?", role="human"),
        ChatMessage(content="duck", role="other"),
    ]


@pytest.mark.parametrize(
    "args,expected",
    [
        (
            ("human", "{question}"),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate.from_template("{question}")
            ),
        ),
        (
            "{question}",
            HumanMessagePromptTemplate(
                prompt=PromptTemplate.from_template("{question}")
            ),
        ),
        (HumanMessage(content="question"), HumanMessage(content="question")),
        (
            HumanMessagePromptTemplate(
                prompt=PromptTemplate.from_template("{question}")
            ),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate.from_template("{question}")
            ),
        ),
    ],
)
def test_convert_to_message(
    args: Any, expected: Union[BaseMessage, BaseMessagePromptTemplate]
) -> None:
    """Test convert to message."""
    assert _convert_to_message(args) == expected


def test_chat_prompt_template_indexing() -> None:
    message1 = SystemMessage(content="foo")
    message2 = HumanMessage(content="bar")
    message3 = HumanMessage(content="baz")
    template = ChatPromptTemplate.from_messages([message1, message2, message3])
    assert template[0] == message1
    assert template[1] == message2

    # Slice starting from index 1
    slice_template = template[1:]
    assert slice_template[0] == message2
    assert len(slice_template) == 2


def test_chat_prompt_template_append_and_extend() -> None:
    """Test append and extend methods of ChatPromptTemplate."""
    message1 = SystemMessage(content="foo")
    message2 = HumanMessage(content="bar")
    message3 = HumanMessage(content="baz")
    template = ChatPromptTemplate.from_messages([message1])
    template.append(message2)
    template.append(message3)
    assert len(template) == 3
    template.extend([message2, message3])
    assert len(template) == 5
    assert template.messages == [
        message1,
        message2,
        message3,
        message2,
        message3,
    ]
    template.append(("system", "hello!"))
    assert template[-1] == SystemMessagePromptTemplate.from_template("hello!")


def test_convert_to_message_is_strict() -> None:
    """Verify that _convert_to_message is strict."""
    with pytest.raises(ValueError):
        # meow does not correspond to a valid message type.
        # this test is here to ensure that functionality to interpret `meow`
        # as a role is NOT added.
        _convert_to_message(("meow", "question"))


def test_chat_message_partial() -> None:
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI assistant named {name}."),
            ("human", "Hi I'm {user}"),
            ("ai", "Hi there, {user}, I'm {name}."),
            ("human", "{input}"),
        ]
    )
    template2 = template.partial(user="Lucy", name="R2D2")
    with pytest.raises(KeyError):
        template.format_messages(input="hello")

    res = template2.format_messages(input="hello")
    expected = [
        SystemMessage(content="You are an AI assistant named R2D2."),
        HumanMessage(content="Hi I'm Lucy"),
        AIMessage(content="Hi there, Lucy, I'm R2D2."),
        HumanMessage(content="hello"),
    ]
    assert res == expected
    assert template2.format(input="hello") == get_buffer_string(expected)
