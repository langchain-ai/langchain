from typing import List

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    BaseMessagePromptTemplate,
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    ChatPromptValue,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
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
        '[SystemMessage(content="Here\'s some context: context", '
        'additional_kwargs={}), HumanMessage(content="Hello foo, '
        "I'm bar. Thanks for the context\", additional_kwargs={}), "
        "AIMessage(content=\"I'm an AI. I'm foo. I'm bar.\", additional_kwargs={}), "
        "ChatMessage(content=\"I'm a generic message. I'm foo. I'm bar.\","
        " additional_kwargs={}, role='test')]"
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
