import re
import warnings
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.load import dumpd, load
from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     SystemMessage, ToolMessage, get_buffer_string)
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import (AIMessagePromptTemplate,
                                         ChatMessagePromptTemplate, ChatPromptTemplate,
                                         HumanMessagePromptTemplate,
                                         MessagesPlaceholder,
                                         SystemMessagePromptTemplate,
                                         _convert_to_message_template)
from langchain_core.prompts.message import BaseMessagePromptTemplate
from langchain_core.prompts.string import PromptTemplateFormat
from langchain_core.utils.pydantic import PYDANTIC_VERSION
from packaging import version
from pydantic import ValidationError
from syrupy.assertion import SnapshotAssertion
from tests.unit_tests.pydantic_utils import _normalize_schema

CUR_DIR = Path(__file__).parent.absolute().resolve()


@pytest.fixture
def messages() -> list[BaseMessagePromptTemplate]:
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


@pytest.fixture
def chat_prompt_template(
    messages: list[BaseMessagePromptTemplate],
) -> ChatPromptTemplate:
    """Create a chat prompt template."""
    return ChatPromptTemplate(
        input_variables=["foo", "bar", "context"],
        messages=messages,
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


def test_create_system_message_prompt_template_from_template_partial() -> None:
    """Create a system message prompt template with partials."""
    graph_creator_content = """
    Your instructions are:
    {instructions}
    History:
    {history}
    """
    json_prompt_instructions: dict = {}
    graph_analyst_template = SystemMessagePromptTemplate.from_template(
        template=graph_creator_content,
        input_variables=["history"],
        partial_variables={"instructions": json_prompt_instructions},
    )
    assert graph_analyst_template.format(history="history") == SystemMessage(
        content="\n    Your instructions are:\n    {}\n    History:\n    history\n    "
    )


def test_create_system_message_prompt_list_template() -> None:
    graph_creator_content1 = """
    This is the prompt for the first test:
    {variables}
    """
    graph_creator_content2 = """
    This is the prompt for the second test:
        {variables}
        """
    graph_analyst_template = SystemMessagePromptTemplate.from_template(
        template=[graph_creator_content1, graph_creator_content2],
        input_variables=["variables"],
    )
    assert graph_analyst_template.format(variables="foo") == SystemMessage(
        content=[
            {
                "type": "text",
                "text": "\n    This is the prompt for the first test:\n    foo\n    ",
            },
            {
                "type": "text",
                "text": "\n    This is the prompt for "
                "the second test:\n        foo\n        ",
            },
        ]
    )


def test_create_system_message_prompt_list_template_partial_variables_not_null() -> (
    None
):
    graph_creator_content1 = """
    This is the prompt for the first test:
    {variables}
    """
    graph_creator_content2 = """
    This is the prompt for the second test:
        {variables}
        """

    with pytest.raises(
        ValueError, match="Partial variables are not supported for list of templates"
    ):
        _ = SystemMessagePromptTemplate.from_template(
            template=[graph_creator_content1, graph_creator_content2],
            input_variables=["variables"],
            partial_variables={"variables": "foo"},
        )


def test_message_prompt_template_from_template_file() -> None:
    expected = ChatMessagePromptTemplate(
        prompt=PromptTemplate(
            template="Question: {question}\nAnswer:", input_variables=["question"]
        ),
        role="human",
    )
    actual = ChatMessagePromptTemplate.from_template_file(
        Path(__file__).parent.parent / "data" / "prompt_file.txt",
        role="human",
    )
    assert expected == actual


async def test_chat_prompt_template(chat_prompt_template: ChatPromptTemplate) -> None:
    """Test chat prompt template."""
    prompt = chat_prompt_template.format_prompt(foo="foo", bar="bar", context="context")
    assert isinstance(prompt, ChatPromptValue)
    messages = prompt.to_messages()
    assert len(messages) == 4
    assert messages[0].content == "Here's some context: context"
    assert messages[1].content == "Hello foo, I'm bar. Thanks for the context"
    assert messages[2].content == "I'm an AI. I'm foo. I'm bar."
    assert messages[3].content == "I'm a generic message. I'm foo. I'm bar."

    async_prompt = await chat_prompt_template.aformat_prompt(
        foo="foo", bar="bar", context="context"
    )

    assert async_prompt.to_messages() == messages

    string = prompt.to_string()
    expected = (
        "System: Here's some context: context\n"
        "Human: Hello foo, I'm bar. Thanks for the context\n"
        "AI: I'm an AI. I'm foo. I'm bar.\n"
        "test: I'm a generic message. I'm foo. I'm bar."
    )
    assert string == expected

    string = chat_prompt_template.format(foo="foo", bar="bar", context="context")
    assert string == expected

    string = await chat_prompt_template.aformat(foo="foo", bar="bar", context="context")
    assert string == expected


def test_chat_prompt_template_from_messages(
    messages: list[BaseMessagePromptTemplate],
) -> None:
    """Test creating a chat prompt template from messages."""
    chat_prompt_template = ChatPromptTemplate.from_messages(messages)
    assert sorted(chat_prompt_template.input_variables) == sorted(
        [
            "context",
            "foo",
            "bar",
        ]
    )
    assert len(chat_prompt_template.messages) == 4


async def test_chat_prompt_template_from_messages_using_role_strings() -> None:
    """Test creating a chat prompt template from role string messages."""
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI bot. Your name is {name}."),
            ("human", "Hello, how are you doing?"),
            ("ai", "I'm doing well, thanks!"),
            ("human", "{user_input}"),
        ]
    )

    expected = [
        SystemMessage(
            content="You are a helpful AI bot. Your name is Bob.", additional_kwargs={}
        ),
        HumanMessage(content="Hello, how are you doing?", additional_kwargs={}),
        AIMessage(content="I'm doing well, thanks!", additional_kwargs={}),
        HumanMessage(content="What is your name?", additional_kwargs={}),
    ]

    messages = template.format_messages(name="Bob", user_input="What is your name?")
    assert messages == expected

    messages = await template.aformat_messages(
        name="Bob", user_input="What is your name?"
    )
    assert messages == expected


def test_chat_prompt_template_from_messages_mustache() -> None:
    """Test creating a chat prompt template from role string messages."""
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI bot. Your name is {{name}}."),
            ("human", "Hello, how are you doing?"),
            ("ai", "I'm doing well, thanks!"),
            ("human", "{{user_input}}"),
        ],
        "mustache",
    )

    messages = template.format_messages(name="Bob", user_input="What is your name?")

    assert messages == [
        SystemMessage(
            content="You are a helpful AI bot. Your name is Bob.", additional_kwargs={}
        ),
        HumanMessage(content="Hello, how are you doing?", additional_kwargs={}),
        AIMessage(content="I'm doing well, thanks!", additional_kwargs={}),
        HumanMessage(content="What is your name?", additional_kwargs={}),
    ]


@pytest.mark.requires("jinja2")
def test_chat_prompt_template_from_messages_jinja2() -> None:
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI bot. Your name is {{ name }}."),
            ("human", "Hello, how are you doing?"),
            ("ai", "I'm doing well, thanks!"),
            ("human", "{{ user_input }}"),
        ],
        "jinja2",
    )

    messages = template.format_messages(name="Bob", user_input="What is your name?")

    assert messages == [
        SystemMessage(
            content="You are a helpful AI bot. Your name is Bob.", additional_kwargs={}
        ),
        HumanMessage(content="Hello, how are you doing?", additional_kwargs={}),
        AIMessage(content="I'm doing well, thanks!", additional_kwargs={}),
        HumanMessage(content="What is your name?", additional_kwargs={}),
    ]


def test_chat_prompt_template_from_messages_using_message_classes() -> None:
    """Test creating a chat prompt template using message class tuples."""
    template = ChatPromptTemplate.from_messages(
        [
            (SystemMessage, "You are a helpful AI bot. Your name is {name}."),
            (HumanMessage, "Hello, how are you doing?"),
            (AIMessage, "I'm doing well, thanks!"),
            (HumanMessage, "{user_input}"),
        ]
    )

    expected = [
        SystemMessage(
            content="You are a helpful AI bot. Your name is Bob.", additional_kwargs={}
        ),
        HumanMessage(content="Hello, how are you doing?", additional_kwargs={}),
        AIMessage(content="I'm doing well, thanks!", additional_kwargs={}),
        HumanMessage(content="What is your name?", additional_kwargs={}),
    ]

    messages = template.format_messages(name="Bob", user_input="What is your name?")
    assert messages == expected


def test_chat_prompt_template_message_class_tuples_with_invoke() -> None:
    """Test message class tuples work with invoke() method."""
    template = ChatPromptTemplate.from_messages(
        [
            (SystemMessage, "You are {name}."),
            (HumanMessage, "{question}"),
        ]
    )

    result = template.invoke({"name": "Alice", "question": "Hello?"})
    messages = result.to_messages()

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[0].content == "You are Alice."
    assert messages[1].content == "Hello?"


def test_chat_prompt_template_message_class_tuples_mixed_syntax() -> None:
    """Test mixing message class tuples with string tuples."""
    template = ChatPromptTemplate.from_messages(
        [
            (SystemMessage, "System prompt."),  # class tuple
            ("human", "{user_input}"),  # string tuple
            (AIMessage, "AI response."),  # class tuple
        ]
    )

    messages = template.format_messages(user_input="Hello!")

    assert len(messages) == 3
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[2], AIMessage)
    assert messages[0].content == "System prompt."
    assert messages[1].content == "Hello!"
    assert messages[2].content == "AI response."


def test_chat_prompt_template_message_class_tuples_multiple_variables() -> None:
    """Test message class tuples with multiple template variables."""
    template = ChatPromptTemplate.from_messages(
        [
            (SystemMessage, "You are {name}, a {role} assistant."),
            (HumanMessage, "My question about {topic} is: {question}"),
        ]
    )

    messages = template.format_messages(
        name="Bob", role="helpful", topic="Python", question="What is a list?"
    )

    assert len(messages) == 2
    assert messages[0].content == "You are Bob, a helpful assistant."
    assert messages[1].content == "My question about Python is: What is a list?"


def test_chat_prompt_template_message_class_tuples_empty_template() -> None:
    """Test message class tuples with empty string template."""
    template = ChatPromptTemplate.from_messages(
        [
            (HumanMessage, ""),
        ]
    )

    messages = template.format_messages()

    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == ""


def test_chat_prompt_template_message_class_tuples_static_text() -> None:
    """Test message class tuples with no template variables (static text)."""
    template = ChatPromptTemplate.from_messages(
        [
            (SystemMessage, "You are a helpful assistant."),
            (HumanMessage, "Hello there!"),
            (AIMessage, "Hi! How can I help?"),
        ]
    )

    messages = template.format_messages()

    assert len(messages) == 3
    assert messages[0].content == "You are a helpful assistant."
    assert messages[1].content == "Hello there!"
    assert messages[2].content == "Hi! How can I help?"


def test_chat_prompt_template_message_class_tuples_input_variables() -> None:
    """Test that input_variables are correctly extracted from message class tuples."""
    template = ChatPromptTemplate.from_messages(
        [
            (SystemMessage, "You are {name}."),
            (HumanMessage, "{question}"),
        ]
    )

    assert sorted(template.input_variables) == ["name", "question"]


def test_chat_prompt_template_message_class_tuples_partial_variables() -> None:
    """Test message class tuples with partial variables."""
    template = ChatPromptTemplate.from_messages(
        [
            (SystemMessage, "You are {name}, a {role} assistant."),
            (HumanMessage, "{question}"),
        ]
    )

    partial_template = template.partial(name="Alice", role="helpful")
    messages = partial_template.format_messages(question="What is Python?")

    assert len(messages) == 2
    assert messages[0].content == "You are Alice, a helpful assistant."
    assert messages[1].content == "What is Python?"


def test_chat_prompt_template_message_class_tuples_with_placeholder() -> None:
    """Test message class tuples combined with MessagesPlaceholder."""
    template = ChatPromptTemplate.from_messages(
        [
            (SystemMessage, "You are a helpful assistant."),
            MessagesPlaceholder("history"),
            (HumanMessage, "{question}"),
        ]
    )

    messages = template.format_messages(
        history=[HumanMessage(content="Hi"), AIMessage(content="Hello!")],
        question="How are you?",
    )

    assert len(messages) == 4
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[2], AIMessage)
    assert isinstance(messages[3], HumanMessage)
    assert messages[3].content == "How are you?"


def test_chat_prompt_template_message_class_tuples_mustache_format() -> None:
    """Test message class tuples with mustache template format."""
    template = ChatPromptTemplate.from_messages(
        [
            (SystemMessage, "You are {{name}}."),
            (HumanMessage, "{{question}}"),
        ],
        template_format="mustache",
    )

    messages = template.format_messages(name="Bob", question="Hello?")

    assert len(messages) == 2
    assert messages[0].content == "You are Bob."
    assert messages[1].content == "Hello?"


def test_chat_prompt_template_message_class_tuples_append() -> None:
    """Test appending message class tuples to existing template."""
    template = ChatPromptTemplate.from_messages(
        [
            (SystemMessage, "You are helpful."),
        ]
    )

    template.append((HumanMessage, "{question}"))

    messages = template.format_messages(question="What is AI?")

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "What is AI?"


def test_chat_prompt_template_message_class_tuples_extend() -> None:
    """Test extending template with message class tuples."""
    template = ChatPromptTemplate.from_messages(
        [
            (SystemMessage, "System message."),
        ]
    )

    template.extend(
        [
            (HumanMessage, "{q1}"),
            (AIMessage, "Response."),
            (HumanMessage, "{q2}"),
        ]
    )

    messages = template.format_messages(q1="First?", q2="Second?")

    assert len(messages) == 4
    assert messages[1].content == "First?"
    assert messages[3].content == "Second?"


def test_chat_prompt_template_message_class_tuples_concatenation() -> None:
    """Test concatenating two templates with message class tuples."""
    template1 = ChatPromptTemplate.from_messages(
        [
            (SystemMessage, "You are {name}."),
        ]
    )

    template2 = ChatPromptTemplate.from_messages(
        [
            (HumanMessage, "{question}"),
        ]
    )

    combined = template1 + template2
    messages = combined.format_messages(name="Alice", question="Hello?")

    assert len(messages) == 2
    assert messages[0].content == "You are Alice."
    assert messages[1].content == "Hello?"


def test_chat_prompt_template_message_class_tuples_slicing() -> None:
    """Test slicing a template with message class tuples."""
    template = ChatPromptTemplate.from_messages(
        [
            (SystemMessage, "System."),
            (HumanMessage, "Human 1."),
            (AIMessage, "AI."),
            (HumanMessage, "Human 2."),
        ]
    )

    sliced = template[1:3]
    messages = sliced.format_messages()

    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)


def test_chat_prompt_template_message_class_tuples_special_characters() -> None:
    """Test message class tuples with special characters in template."""
    template = ChatPromptTemplate.from_messages(
        [
            (SystemMessage, "You are a helpful assistant! 🤖"),
            (HumanMessage, "Question: {question}? (please answer)"),
        ]
    )

    messages = template.format_messages(question="What is 2+2")

    assert len(messages) == 2
    assert messages[0].content == "You are a helpful assistant! 🤖"
    assert messages[1].content == "Question: What is 2+2? (please answer)"


@pytest.mark.requires("jinja2")
@pytest.mark.parametrize(
    ("template_format", "image_type_placeholder", "image_data_placeholder"),
    [
        ("f-string", "{image_type}", "{image_data}"),
        ("mustache", "{{image_type}}", "{{image_data}}"),
        ("jinja2", "{{ image_type }}", "{{ image_data }}"),
    ],
)
def test_chat_prompt_template_image_prompt_from_message(
    template_format: PromptTemplateFormat,
    image_type_placeholder: str,
    image_data_placeholder: str,
) -> None:
    prompt = {
        "type": "image_url",
        "image_url": {
            "url": f"data:{image_type_placeholder};base64, {image_data_placeholder}",
            "detail": "low",
        },
    }

    template = ChatPromptTemplate.from_messages(
        [("human", [prompt])], template_format=template_format
    )
    assert template.format_messages(
        image_type="image/png", image_data="base64data"
    ) == [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64, base64data",
                        "detail": "low",
                    },
                }
            ]
        )
    ]


def test_chat_prompt_template_with_messages(
    messages: list[BaseMessagePromptTemplate],
) -> None:
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            *messages,
            HumanMessage(content="foo"),
        ]
    )
    assert sorted(chat_prompt_template.input_variables) == sorted(
        [
            "context",
            "foo",
            "bar",
        ]
    )
    assert len(chat_prompt_template.messages) == 5
    prompt_value = chat_prompt_template.format_prompt(
        context="see", foo="this", bar="magic"
    )
    prompt_value_messages = prompt_value.to_messages()
    assert prompt_value_messages[-1] == HumanMessage(content="foo")


def test_chat_invalid_input_variables_extra() -> None:
    messages = [HumanMessage(content="foo")]
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Got mismatched input_variables. Expected: set(). Got: ['foo']"
        ),
    ):
        ChatPromptTemplate(
            messages=messages,
            input_variables=["foo"],
            validate_template=True,
        )
    assert (
        ChatPromptTemplate(messages=messages, input_variables=["foo"]).input_variables
        == []
    )


def test_chat_invalid_input_variables_missing() -> None:
    messages = [HumanMessagePromptTemplate.from_template("{foo}")]
    with pytest.raises(
        ValueError,
        match=re.escape("Got mismatched input_variables. Expected: {'foo'}. Got: []"),
    ):
        ChatPromptTemplate(
            messages=messages,
            input_variables=[],
            validate_template=True,
        )
    assert ChatPromptTemplate(
        messages=messages,
        input_variables=[],
    ).input_variables == ["foo"]


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
        messages=messages,
        partial_variables={"formatins": "some structure"},
    )
    assert set(prompt.input_variables) == {"question", "context"}
    assert prompt.partial_variables == {"formatins": "some structure"}


@pytest.mark.parametrize(
    ("args", "expected"),
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
    args: Any, expected: BaseMessage | BaseMessagePromptTemplate
) -> None:
    """Test convert to message."""
    assert _convert_to_message_template(args) == expected


def test_chat_prompt_template_indexing() -> None:
    message1 = SystemMessage(content="foo")
    message2 = HumanMessage(content="bar")
    message3 = HumanMessage(content="baz")
    template = ChatPromptTemplate([message1, message2, message3])
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
    template = ChatPromptTemplate([message1])
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
    with pytest.raises(ValueError, match="Unexpected message type: meow"):
        # meow does not correspond to a valid message type.
        # this test is here to ensure that functionality to interpret `meow`
        # as a role is NOT added.
        _convert_to_message_template(("meow", "question"))


def test_chat_message_partial() -> None:
    template = ChatPromptTemplate(
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


def test_chat_message_partial_composition() -> None:
    """Test composition of partially initialized messages."""
    prompt = ChatPromptTemplate.from_messages([("system", "Prompt {x} {y}")]).partial(
        x="1"
    )

    appendix = ChatPromptTemplate.from_messages([("system", "Appendix {z}")])

    res = (prompt + appendix).format_messages(y="2", z="3")
    expected = [
        SystemMessage(content="Prompt 1 2"),
        SystemMessage(content="Appendix 3"),
    ]

    assert res == expected


async def test_chat_tmpl_from_messages_multipart_text() -> None:
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI assistant named {name}."),
            (
                "human",
                [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "text", "text": "Oh nvm"},
                ],
            ),
        ]
    )
    expected = [
        SystemMessage(content="You are an AI assistant named R2D2."),
        HumanMessage(
            content=[
                {"type": "text", "text": "What's in this image?"},
                {"type": "text", "text": "Oh nvm"},
            ]
        ),
    ]
    messages = template.format_messages(name="R2D2")
    assert messages == expected

    messages = await template.aformat_messages(name="R2D2")
    assert messages == expected


async def test_chat_tmpl_from_messages_multipart_text_with_template() -> None:
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI assistant named {name}."),
            (
                "human",
                [
                    {"type": "text", "text": "What's in this {object_name}?"},
                    {"type": "text", "text": "Oh nvm"},
                ],
            ),
        ]
    )
    expected = [
        SystemMessage(content="You are an AI assistant named R2D2."),
        HumanMessage(
            content=[
                {"type": "text", "text": "What's in this image?"},
                {"type": "text", "text": "Oh nvm"},
            ]
        ),
    ]
    messages = template.format_messages(name="R2D2", object_name="image")
    assert messages == expected

    messages = await template.aformat_messages(name="R2D2", object_name="image")
    assert messages == expected


async def test_chat_tmpl_from_messages_multipart_image() -> None:
    """Test multipart image URL formatting."""
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAA"
    other_base64_image = "other_iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAA"
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI assistant named {name}."),
            (
                "human",
                [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": "data:image/jpeg;base64,{my_image}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{my_image}"},
                    },
                    {"type": "image_url", "image_url": "{my_other_image}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "{my_other_image}", "detail": "medium"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://www.langchain.com/image.png"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,foobar"},
                    },
                ],
            ),
        ]
    )
    expected = [
        SystemMessage(content="You are an AI assistant named R2D2."),
        HumanMessage(
            content=[
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"{other_base64_image}"},
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{other_base64_image}",
                        "detail": "medium",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://www.langchain.com/image.png"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,foobar"},
                },
            ]
        ),
    ]
    messages = template.format_messages(
        name="R2D2", my_image=base64_image, my_other_image=other_base64_image
    )
    assert messages == expected

    messages = await template.aformat_messages(
        name="R2D2", my_image=base64_image, my_other_image=other_base64_image
    )
    assert messages == expected


async def test_chat_tmpl_from_messages_multipart_formatting_with_path() -> None:
    """Verify that we cannot pass `path` for an image as a variable."""
    in_mem_ = "base64mem"

    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI assistant named {name}."),
            (
                "human",
                [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": "data:image/jpeg;base64,{in_mem}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"path": "{file_path}"},
                    },
                ],
            ),
        ]
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Loading images from 'path' has been removed as of 0.3.15 "
            "for security reasons."
        ),
    ):
        template.format_messages(
            name="R2D2",
            in_mem=in_mem_,
            file_path="some/path",
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Loading images from 'path' has been removed as of 0.3.15 "
            "for security reasons."
        ),
    ):
        await template.aformat_messages(
            name="R2D2",
            in_mem=in_mem_,
            file_path="some/path",
        )


def test_messages_placeholder() -> None:
    prompt = MessagesPlaceholder("history")
    with pytest.raises(KeyError):
        prompt.format_messages()
    prompt = MessagesPlaceholder("history", optional=True)
    assert prompt.format_messages() == []
    assert prompt.format_messages(
        history=[("system", "You are an AI assistant."), "Hello!"]
    ) == [
        SystemMessage(content="You are an AI assistant."),
        HumanMessage(content="Hello!"),
    ]


def test_messages_placeholder_with_max() -> None:
    history = [
        AIMessage(content="1"),
        AIMessage(content="2"),
        AIMessage(content="3"),
    ]
    prompt = MessagesPlaceholder("history")
    assert prompt.format_messages(history=history) == history
    prompt = MessagesPlaceholder("history", n_messages=2)
    assert prompt.format_messages(history=history) == [
        AIMessage(content="2"),
        AIMessage(content="3"),
    ]


def test_chat_prompt_message_placeholder_partial() -> None:
    prompt = ChatPromptTemplate([MessagesPlaceholder("history")])
    prompt = prompt.partial(history=[("system", "foo")])
    assert prompt.format_messages() == [SystemMessage(content="foo")]
    assert prompt.format_messages(history=[("system", "bar")]) == [
        SystemMessage(content="bar")
    ]

    prompt = ChatPromptTemplate(
        [
            MessagesPlaceholder("history", optional=True),
        ]
    )
    assert prompt.format_messages() == []
    prompt = prompt.partial(history=[("system", "foo")])
    assert prompt.format_messages() == [SystemMessage(content="foo")]


def test_chat_prompt_message_placeholder_tuple() -> None:
    prompt = ChatPromptTemplate([("placeholder", "{convo}")])
    assert prompt.format_messages(convo=[("user", "foo")]) == [
        HumanMessage(content="foo")
    ]

    assert prompt.format_messages() == []

    # Is optional = True
    optional_prompt = ChatPromptTemplate([("placeholder", ["{convo}", False])])
    assert optional_prompt.format_messages(convo=[("user", "foo")]) == [
        HumanMessage(content="foo")
    ]
    with pytest.raises(KeyError):
        assert optional_prompt.format_messages() == []


def test_chat_prompt_message_placeholder_dict() -> None:
    prompt = ChatPromptTemplate([{"role": "placeholder", "content": "{convo}"}])
    assert prompt.format_messages(convo=[("user", "foo")]) == [
        HumanMessage(content="foo")
    ]

    assert prompt.format_messages() == []

    # Is optional = True
    optional_prompt = ChatPromptTemplate(
        [{"role": "placeholder", "content": ["{convo}", False]}]
    )
    assert optional_prompt.format_messages(convo=[("user", "foo")]) == [
        HumanMessage(content="foo")
    ]
    with pytest.raises(KeyError):
        assert optional_prompt.format_messages() == []


def test_chat_prompt_message_dict() -> None:
    prompt = ChatPromptTemplate(
        [
            {"role": "system", "content": "foo"},
            {"role": "user", "content": "bar"},
        ]
    )
    assert prompt.format_messages() == [
        SystemMessage(content="foo"),
        HumanMessage(content="bar"),
    ]

    with pytest.raises(ValueError, match="Invalid template: False"):
        ChatPromptTemplate([{"role": "system", "content": False}])

    with pytest.raises(ValueError, match="Unexpected message type: foo"):
        ChatPromptTemplate([{"role": "foo", "content": "foo"}])


async def test_messages_prompt_accepts_list() -> None:
    prompt = ChatPromptTemplate([MessagesPlaceholder("history")])
    value = prompt.invoke([("user", "Hi there")])  # type: ignore[arg-type]
    assert value.to_messages() == [HumanMessage(content="Hi there")]

    value = await prompt.ainvoke([("user", "Hi there")])  # type: ignore[arg-type]
    assert value.to_messages() == [HumanMessage(content="Hi there")]

    # Assert still raises a nice error
    prompt = ChatPromptTemplate(
        [
            ("system", "You are a {foo}"),
            MessagesPlaceholder("history"),
        ]
    )
    with pytest.raises(TypeError):
        prompt.invoke([("user", "Hi there")])  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        await prompt.ainvoke([("user", "Hi there")])  # type: ignore[arg-type]


def test_chat_input_schema(snapshot: SnapshotAssertion) -> None:
    prompt_all_required = ChatPromptTemplate(
        messages=[MessagesPlaceholder("history", optional=False), ("user", "${input}")]
    )
    assert set(prompt_all_required.input_variables) == {"input", "history"}
    assert prompt_all_required.optional_variables == []
    with pytest.raises(ValidationError):
        prompt_all_required.input_schema(input="")

    if version.parse("2.10") <= PYDANTIC_VERSION:
        assert _normalize_schema(
            prompt_all_required.get_input_jsonschema()
        ) == snapshot(name="required")
    prompt_optional = ChatPromptTemplate(
        messages=[MessagesPlaceholder("history", optional=True), ("user", "${input}")]
    )
    # input variables only lists required variables
    assert set(prompt_optional.input_variables) == {"input"}
    prompt_optional.input_schema(input="")  # won't raise error

    if version.parse("2.10") <= PYDANTIC_VERSION:
        assert _normalize_schema(prompt_optional.get_input_jsonschema()) == snapshot(
            name="partial"
        )


def test_chat_prompt_w_msgs_placeholder_ser_des(snapshot: SnapshotAssertion) -> None:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "foo"),
            MessagesPlaceholder("bar"),
            ("human", "baz"),
        ]
    )
    assert dumpd(MessagesPlaceholder("bar")) == snapshot(name="placeholder")
    assert load(dumpd(MessagesPlaceholder("bar"))) == MessagesPlaceholder("bar")
    assert dumpd(prompt) == snapshot(name="chat_prompt")
    assert load(dumpd(prompt)) == prompt


def test_chat_tmpl_serdes(snapshot: SnapshotAssertion) -> None:
    """Test chat prompt template ser/des."""
    template = ChatPromptTemplate(
        [
            ("system", "You are an AI assistant named {name}."),
            ("system", [{"text": "You are an AI assistant named {name}."}]),
            SystemMessagePromptTemplate.from_template("you are {foo}"),
            cast(
                "tuple",
                (
                    "human",
                    [
                        "hello",
                        {"text": "What's in this image?"},
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "text",
                            "text": "What's in this image?",
                            "cache_control": {"type": "{foo}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{my_image}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,{my_image}"},
                        },
                        {"type": "image_url", "image_url": "{my_other_image}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "{my_other_image}",
                                "detail": "medium",
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://www.langchain.com/image.png"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,foobar"},
                        },
                        {"image_url": {"url": "data:image/jpeg;base64,foobar"}},
                    ],
                ),
            ),
            ("placeholder", "{chat_history}"),
            MessagesPlaceholder("more_history", optional=False),
        ]
    )
    assert dumpd(template) == snapshot()
    assert load(dumpd(template)) == template


@pytest.mark.xfail(
    reason=(
        "In a breaking release, we can update `_convert_to_message_template` to use "
        "DictPromptTemplate for all `dict` inputs, allowing for templatization "
        "of message attributes outside content blocks. That would enable the below "
        "test to pass."
    )
)
def test_chat_tmpl_dict_msg() -> None:
    template = ChatPromptTemplate(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "{text1}",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
                "name": "{name1}",
                "tool_calls": [
                    {
                        "name": "{tool_name1}",
                        "args": {"arg1": "{tool_arg1}"},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            },
            {
                "role": "tool",
                "content": "{tool_content2}",
                "tool_call_id": "1",
                "name": "{tool_name1}",
            },
        ]
    )
    expected = [
        AIMessage(
            [
                {
                    "type": "text",
                    "text": "important message",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            name="foo",
            tool_calls=[
                {
                    "name": "do_stuff",
                    "args": {"arg1": "important arg1"},
                    "id": "1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage("foo", name="do_stuff", tool_call_id="1"),
    ]

    actual = template.invoke(
        {
            "text1": "important message",
            "name1": "foo",
            "tool_arg1": "important arg1",
            "tool_name1": "do_stuff",
            "tool_content2": "foo",
        }
    ).to_messages()
    assert actual == expected

    partial_ = template.partial(text1="important message")
    actual = partial_.invoke(
        {
            "name1": "foo",
            "tool_arg1": "important arg1",
            "tool_name1": "do_stuff",
            "tool_content2": "foo",
        }
    ).to_messages()
    assert actual == expected


def test_chat_prompt_template_variable_names() -> None:
    """This test was written for an edge case that triggers a warning from Pydantic.

    Verify that no run time warnings are raised.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")  # Cause all warnings to always be triggered
        prompt = ChatPromptTemplate([("system", "{schema}")])
        prompt.get_input_schema()

    if record:
        error_msg = [
            f"Warning type: {warning.category.__name__}, "
            f"Warning message: {warning.message}, "
            f"Warning location: {warning.filename}:{warning.lineno}"
            for warning in record
        ]
        msg = "\n".join(error_msg)
    else:
        msg = ""

    assert list(record) == [], msg

    # Verify value errors raised from illegal names
    assert ChatPromptTemplate(
        [("system", "{_private}")]
    ).get_input_schema().model_json_schema() == {
        "properties": {"_private": {"title": "Private", "type": "string"}},
        "required": ["_private"],
        "title": "PromptInput",
        "type": "object",
    }

    assert ChatPromptTemplate(
        [("system", "{model_json_schema}")]
    ).get_input_schema().model_json_schema() == {
        "properties": {
            "model_json_schema": {"title": "Model Json Schema", "type": "string"}
        },
        "required": ["model_json_schema"],
        "title": "PromptInput",
        "type": "object",
    }


def test_data_prompt_template_deserializable() -> None:
    """Test that the image prompt template is serializable."""
    load(
        dumpd(
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        [{"type": "image", "source_type": "url", "url": "{url}"}],
                    )
                ]
            )
        )
    )


@pytest.mark.requires("jinja2")
@pytest.mark.parametrize(
    ("template_format", "cache_control_placeholder", "source_data_placeholder"),
    [
        ("f-string", "{cache_type}", "{source_data}"),
        ("mustache", "{{cache_type}}", "{{source_data}}"),
    ],
)
def test_chat_prompt_template_data_prompt_from_message(
    template_format: PromptTemplateFormat,
    cache_control_placeholder: str,
    source_data_placeholder: str,
) -> None:
    prompt: dict = {
        "type": "image",
        "source_type": "base64",
        "data": f"{source_data_placeholder}",
    }

    template = ChatPromptTemplate.from_messages(
        [("human", [prompt])], template_format=template_format
    )
    assert template.format_messages(source_data="base64data") == [
        HumanMessage(
            content=[
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": "base64data",
                }
            ]
        )
    ]

    # metadata
    prompt["metadata"] = {"cache_control": {"type": f"{cache_control_placeholder}"}}
    template = ChatPromptTemplate.from_messages(
        [("human", [prompt])], template_format=template_format
    )
    assert template.format_messages(
        cache_type="ephemeral", source_data="base64data"
    ) == [
        HumanMessage(
            content=[
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": "base64data",
                    "metadata": {"cache_control": {"type": "ephemeral"}},
                }
            ]
        )
    ]


def test_dict_message_prompt_template_errors_on_jinja2() -> None:
    prompt = {
        "type": "image",
        "source_type": "base64",
        "data": "{source_data}",
    }

    with pytest.raises(ValueError, match="jinja2"):
        _ = ChatPromptTemplate.from_messages(
            [("human", [prompt])], template_format="jinja2"
        )


def test_rendering_prompt_with_conditionals_no_empty_text_blocks() -> None:
    manifest = {
        "lc": 1,
        "type": "constructor",
        "id": ["langchain_core", "prompts", "chat", "ChatPromptTemplate"],
        "kwargs": {
            "messages": [
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": [
                        "langchain_core",
                        "prompts",
                        "chat",
                        "SystemMessagePromptTemplate",
                    ],
                    "kwargs": {
                        "prompt": {
                            "lc": 1,
                            "type": "constructor",
                            "id": [
                                "langchain_core",
                                "prompts",
                                "prompt",
                                "PromptTemplate",
                            ],
                            "kwargs": {
                                "input_variables": [],
                                "template_format": "mustache",
                                "template": "Always echo back whatever I send you.",
                            },
                        },
                    },
                },
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": [
                        "langchain_core",
                        "prompts",
                        "chat",
                        "HumanMessagePromptTemplate",
                    ],
                    "kwargs": {
                        "prompt": [
                            {
                                "lc": 1,
                                "type": "constructor",
                                "id": [
                                    "langchain_core",
                                    "prompts",
                                    "prompt",
                                    "PromptTemplate",
                                ],
                                "kwargs": {
                                    "input_variables": [],
                                    "template_format": "mustache",
                                    "template": "Here is the teacher's prompt:",
                                    "additional_content_fields": {
                                        "text": "Here is the teacher's prompt:",
                                    },
                                },
                            },
                            {
                                "lc": 1,
                                "type": "constructor",
                                "id": [
                                    "langchain_core",
                                    "prompts",
                                    "prompt",
                                    "PromptTemplate",
                                ],
                                "kwargs": {
                                    "input_variables": ["promptDescription"],
                                    "template_format": "mustache",
                                    "template": '"{{promptDescription}}"\n',
                                    "additional_content_fields": {
                                        "text": '"{{promptDescription}}"\n',
                                    },
                                },
                            },
                            {
                                "lc": 1,
                                "type": "constructor",
                                "id": [
                                    "langchain_core",
                                    "prompts",
                                    "prompt",
                                    "PromptTemplate",
                                ],
                                "kwargs": {
                                    "input_variables": [],
                                    "template_format": "mustache",
                                    "template": "Here is the expected answer or success criteria given by the teacher:",  # noqa: E501
                                    "additional_content_fields": {
                                        "text": "Here is the expected answer or success criteria given by the teacher:",  # noqa: E501
                                    },
                                },
                            },
                            {
                                "lc": 1,
                                "type": "constructor",
                                "id": [
                                    "langchain_core",
                                    "prompts",
                                    "prompt",
                                    "PromptTemplate",
                                ],
                                "kwargs": {
                                    "input_variables": ["expectedResponse"],
                                    "template_format": "mustache",
                                    "template": '"{{expectedResponse}}"\n',
                                    "additional_content_fields": {
                                        "text": '"{{expectedResponse}}"\n',
                                    },
                                },
                            },
                            {
                                "lc": 1,
                                "type": "constructor",
                                "id": [
                                    "langchain_core",
                                    "prompts",
                                    "prompt",
                                    "PromptTemplate",
                                ],
                                "kwargs": {
                                    "input_variables": [],
                                    "template_format": "mustache",
                                    "template": "Note: This may be just one example of many possible correct ways for the student to respond.\n",  # noqa: E501
                                    "additional_content_fields": {
                                        "text": "Note: This may be just one example of many possible correct ways for the student to respond.\n",  # noqa: E501
                                    },
                                },
                            },
                            {
                                "lc": 1,
                                "type": "constructor",
                                "id": [
                                    "langchain_core",
                                    "prompts",
                                    "prompt",
                                    "PromptTemplate",
                                ],
                                "kwargs": {
                                    "input_variables": [],
                                    "template_format": "mustache",
                                    "template": "For your evaluation of the student's response:\n",  # noqa: E501
                                    "additional_content_fields": {
                                        "text": "For your evaluation of the student's response:\n",  # noqa: E501
                                    },
                                },
                            },
                            {
                                "lc": 1,
                                "type": "constructor",
                                "id": [
                                    "langchain_core",
                                    "prompts",
                                    "prompt",
                                    "PromptTemplate",
                                ],
                                "kwargs": {
                                    "input_variables": [],
                                    "template_format": "mustache",
                                    "template": "Here is a transcript of the student's explanation:",  # noqa: E501
                                    "additional_content_fields": {
                                        "text": "Here is a transcript of the student's explanation:",  # noqa: E501
                                    },
                                },
                            },
                            {
                                "lc": 1,
                                "type": "constructor",
                                "id": [
                                    "langchain_core",
                                    "prompts",
                                    "prompt",
                                    "PromptTemplate",
                                ],
                                "kwargs": {
                                    "input_variables": ["responseTranscript"],
                                    "template_format": "mustache",
                                    "template": '"{{responseTranscript}}"\n',
                                    "additional_content_fields": {
                                        "text": '"{{responseTranscript}}"\n',
                                    },
                                },
                            },
                            {
                                "lc": 1,
                                "type": "constructor",
                                "id": [
                                    "langchain_core",
                                    "prompts",
                                    "prompt",
                                    "PromptTemplate",
                                ],
                                "kwargs": {
                                    "input_variables": ["readingFluencyAnalysis"],
                                    "template_format": "mustache",
                                    "template": "{{#readingFluencyAnalysis}} For this task, the student's reading pronunciation and fluency were important. Here is analysis of the student's oral response: \"{{readingFluencyAnalysis}}\" {{/readingFluencyAnalysis}}",  # noqa: E501
                                    "additional_content_fields": {
                                        "text": "{{#readingFluencyAnalysis}} For this task, the student's reading pronunciation and fluency were important. Here is analysis of the student's oral response: \"{{readingFluencyAnalysis}}\" {{/readingFluencyAnalysis}}",  # noqa: E501
                                    },
                                },
                            },
                            {
                                "lc": 1,
                                "type": "constructor",
                                "id": [
                                    "langchain_core",
                                    "prompts",
                                    "prompt",
                                    "PromptTemplate",
                                ],
                                "kwargs": {
                                    "input_variables": ["readingFluencyAnalysis"],
                                    "template_format": "mustache",
                                    "template": "{{#readingFluencyAnalysis}}Root analysis of the student's response (step 3) in this oral analysis rather than inconsistencies in the transcript.{{/readingFluencyAnalysis}}",  # noqa: E501
                                    "additional_content_fields": {
                                        "text": "{{#readingFluencyAnalysis}}Root analysis of the student's response (step 3) in this oral analysis rather than inconsistencies in the transcript.{{/readingFluencyAnalysis}}",  # noqa: E501
                                    },
                                },
                            },
                            {
                                "lc": 1,
                                "type": "constructor",
                                "id": [
                                    "langchain_core",
                                    "prompts",
                                    "prompt",
                                    "PromptTemplate",
                                ],
                                "kwargs": {
                                    "input_variables": ["readingFluencyAnalysis"],
                                    "template_format": "mustache",
                                    "template": "{{#readingFluencyAnalysis}}Remember this is a student, so we care about general fluency - not voice acting. {{/readingFluencyAnalysis}}\n",  # noqa: E501
                                    "additional_content_fields": {
                                        "text": "{{#readingFluencyAnalysis}}Remember this is a student, so we care about general fluency - not voice acting. {{/readingFluencyAnalysis}}\n",  # noqa: E501
                                    },
                                },
                            },
                            {
                                "lc": 1,
                                "type": "constructor",
                                "id": [
                                    "langchain_core",
                                    "prompts",
                                    "prompt",
                                    "PromptTemplate",
                                ],
                                "kwargs": {
                                    "input_variables": ["multipleChoiceAnalysis"],
                                    "template_format": "mustache",
                                    "template": "{{#multipleChoiceAnalysis}}Here is an analysis of the student's multiple choice response: {{multipleChoiceAnalysis}}{{/multipleChoiceAnalysis}}\n",  # noqa: E501
                                    "additional_content_fields": {
                                        "text": "{{#multipleChoiceAnalysis}}Here is an analysis of the student's multiple choice response: {{multipleChoiceAnalysis}}{{/multipleChoiceAnalysis}}\n",  # noqa: E501
                                    },
                                },
                            },
                            {
                                "lc": 1,
                                "type": "constructor",
                                "id": [
                                    "langchain_core",
                                    "prompts",
                                    "prompt",
                                    "PromptTemplate",
                                ],
                                "kwargs": {
                                    "input_variables": [],
                                    "template_format": "mustache",
                                    "template": "Here is the student's whiteboard:\n",
                                    "additional_content_fields": {
                                        "text": "Here is the student's whiteboard:\n",
                                    },
                                },
                            },
                            {
                                "lc": 1,
                                "type": "constructor",
                                "id": [
                                    "langchain_core",
                                    "prompts",
                                    "image",
                                    "ImagePromptTemplate",
                                ],
                                "kwargs": {
                                    "template": {
                                        "url": "{{whiteboard}}",
                                    },
                                    "input_variables": ["whiteboard"],
                                    "template_format": "mustache",
                                    "additional_content_fields": {
                                        "image_url": {
                                            "url": "{{whiteboard}}",
                                        },
                                    },
                                },
                            },
                        ],
                        "additional_options": {},
                    },
                },
            ],
            "input_variables": [
                "promptDescription",
                "expectedResponse",
                "responseTranscript",
                "readingFluencyAnalysis",
                "readingFluencyAnalysis",
                "readingFluencyAnalysis",
                "multipleChoiceAnalysis",
                "whiteboard",
            ],
            "template_format": "mustache",
            "metadata": {
                "lc_hub_owner": "jacob",
                "lc_hub_repo": "mustache-conditionals",
                "lc_hub_commit_hash": "836ad82d512409ea6024fb760b76a27ba58fc68b1179656c0ba2789778686d46",  # noqa: E501
            },
        },
    }

    # Load the ChatPromptTemplate from the manifest
    template = load(manifest)

    # Format with conditional data - rules is empty, so mustache conditionals
    # should not render
    result = template.invoke(
        {
            "promptDescription": "What is the capital of the USA?",
            "expectedResponse": "Washington, D.C.",
            "responseTranscript": "Washington, D.C.",
            "readingFluencyAnalysis": None,
            "multipleChoiceAnalysis": "testing2",
            "whiteboard": "https://foo.com/bar.png",
        }
    )
    content = result.messages[1].content
    assert isinstance(content, list)
    assert not [
        block for block in content if block["type"] == "text" and block["text"] == ""
    ]


def test_fstring_rejects_invalid_identifier_variable_names() -> None:
    """Test that f-string templates block attribute access, indexing.

    This validation prevents template injection attacks by blocking:
    - Attribute access like {msg.__class__}
    - Indexing like {msg[0]}
    - All-digit variable names like {0} or {100} (interpreted as positional args)

    While allowing any other field names that Python's Formatter accepts.
    """
    # Test that attribute access and indexing are blocked (security issue)
    invalid_templates = [
        "{msg.__class__}",  # Attribute access with dunder
        "{msg.__class__.__name__}",  # Multiple dunders
        "{msg.content}",  # Attribute access
        "{msg[0]}",  # Item access
        "{0}",  # All-digit variable name (positional argument)
        "{100}",  # All-digit variable name (positional argument)
        "{42}",  # All-digit variable name (positional argument)
    ]

    for template_str in invalid_templates:
        with pytest.raises(ValueError, match="Invalid variable name") as exc_info:
            ChatPromptTemplate.from_messages(
                [("human", template_str)],
                template_format="f-string",
            )

        error_msg = str(exc_info.value)
        assert "Invalid variable name" in error_msg
        # Check for any of the expected error message parts
        assert (
            "attribute access" in error_msg
            or "indexing" in error_msg
            or "positional arguments" in error_msg
        )

    # Valid templates - Python's Formatter accepts non-identifier field names
    valid_templates = [
        (
            "Hello {name} and {user_id}",
            {"name": "Alice", "user_id": "123"},
            "Hello Alice and 123",
        ),
        ("User: {user-name}", {"user-name": "Bob"}, "User: Bob"),  # Hyphen allowed
        (
            "Value: {2fast}",
            {"2fast": "Charlie"},
            "Value: Charlie",
        ),  # Starts with digit allowed
        ("Data: {my var}", {"my var": "Dave"}, "Data: Dave"),  # Space allowed
    ]

    for template_str, kwargs, expected in valid_templates:
        template = ChatPromptTemplate.from_messages(
            [("human", template_str)],
            template_format="f-string",
        )
        result = template.invoke(kwargs)
        assert result.messages[0].content == expected  # type: ignore[attr-defined]


def test_mustache_template_attribute_access_vulnerability() -> None:
    """Test that Mustache template injection is blocked.

    Verify the fix for security vulnerability GHSA-6qv9-48xg-fc7f

    Previously, Mustache used getattr() as a fallback, allowing access to
    dangerous attributes like __class__, __globals__, etc.

    The fix adds isinstance checks that reject non-dict/list types.
    When templates try to traverse Python objects, they get empty string
    per Mustache spec (better than the previous behavior of exposing internals).
    """
    msg = HumanMessage("howdy")

    # Template tries to access attributes on a Python object
    prompt = ChatPromptTemplate.from_messages(
        [("human", "{{question.__class__.__name__}}")],
        template_format="mustache",
    )

    # After the fix: returns empty string (attack blocked!)
    # Previously would return "HumanMessage" via getattr()
    result = prompt.invoke({"question": msg})
    assert result.messages[0].content == ""  # type: ignore[attr-defined]

    # Mustache still works correctly with actual dicts
    prompt_dict = ChatPromptTemplate.from_messages(
        [("human", "{{person.name}}")],
        template_format="mustache",
    )
    result_dict = prompt_dict.invoke({"person": {"name": "Alice"}})
    assert result_dict.messages[0].content == "Alice"  # type: ignore[attr-defined]
