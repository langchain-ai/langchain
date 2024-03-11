"""Test AzureML Chat Endpoint wrapper."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models.azureml_endpoint import (
    AzureMLChatOnlineEndpoint,
    CustomOpenAIChatContentFormatter,
)


def test_llama_call() -> None:
    """Test valid call to Open Source Foundation Model."""
    chat = AzureMLChatOnlineEndpoint(
        content_formatter=CustomOpenAIChatContentFormatter()
    )
    response = chat.invoke([HumanMessage(content="Foo")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_temperature_kwargs() -> None:
    """Test that timeout kwarg works."""
    chat = AzureMLChatOnlineEndpoint(
        content_formatter=CustomOpenAIChatContentFormatter()
    )
    response = chat.invoke([HumanMessage(content="FOO")], temperature=0.8)
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_message_history() -> None:
    """Test that multiple messages works."""
    chat = AzureMLChatOnlineEndpoint(
        content_formatter=CustomOpenAIChatContentFormatter()
    )
    response = chat.invoke(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_multiple_messages() -> None:
    chat = AzureMLChatOnlineEndpoint(
        content_formatter=CustomOpenAIChatContentFormatter()
    )
    message = HumanMessage(content="Hi!")
    response = chat.generate([[message], [message]])

    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content
