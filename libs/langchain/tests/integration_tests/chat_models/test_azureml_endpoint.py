"""Test AzureML Chat Endpoint wrapper."""

from langchain.chat_models.azureml_endpoint import (
    AzureMLChatOnlineEndpoint,
    LlamaContentFormatter,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    HumanMessage,
    LLMResult,
)


def test_llama_call() -> None:
    """Test valid call to Open Source Foundation Model."""
    chat = AzureMLChatOnlineEndpoint(content_formatter=LlamaContentFormatter())
    response = chat(messages=[HumanMessage(content="Foo")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_timeout_kwargs() -> None:
    """Test that timeout kwarg works."""
    chat = AzureMLChatOnlineEndpoint(content_formatter=LlamaContentFormatter())
    response = chat(messages=[HumanMessage(content="FOO")], timeout=60)
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_message_history() -> None:
    """Test that multiple messages works."""
    chat = AzureMLChatOnlineEndpoint(content_formatter=LlamaContentFormatter())
    response = chat(
        messages=[
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_multiple_messages() -> None:
    chat = AzureMLChatOnlineEndpoint(content_formatter=LlamaContentFormatter())
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
