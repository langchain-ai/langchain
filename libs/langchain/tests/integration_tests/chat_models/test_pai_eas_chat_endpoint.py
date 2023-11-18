"""Test AliCloud Pai Eas Chat Model."""
import os

from langchain.callbacks.manager import CallbackManager
from langchain.chat_models.pai_eas_endpoint import PaiEasChatEndpoint
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    HumanMessage,
    LLMResult,
)
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_pai_eas_call() -> None:
    chat = PaiEasChatEndpoint(
        eas_service_url=os.getenv("EAS_SERVICE_URL"),
        eas_service_token=os.getenv("EAS_SERVICE_TOKEN"),
    )
    response = chat(messages=[HumanMessage(content="Say foo:")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_multiple_history() -> None:
    """Tests multiple history works."""
    chat = PaiEasChatEndpoint(
        eas_service_url=os.getenv("EAS_SERVICE_URL"),
        eas_service_token=os.getenv("EAS_SERVICE_TOKEN"),
    )

    response = chat(
        messages=[
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_stream() -> None:
    """Test that stream works."""
    chat = PaiEasChatEndpoint(
        eas_service_url=os.getenv("EAS_SERVICE_URL"),
        eas_service_token=os.getenv("EAS_SERVICE_TOKEN"),
        streaming=True,
    )
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    response = chat(
        messages=[
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="Who are you?"),
        ],
        stream=True,
        callbacks=callback_manager,
    )
    assert callback_handler.llm_streams > 0
    assert isinstance(response.content, str)


def test_multiple_messages() -> None:
    """Tests multiple messages works."""
    chat = PaiEasChatEndpoint(
        eas_service_url=os.getenv("EAS_SERVICE_URL"),
        eas_service_token=os.getenv("EAS_SERVICE_TOKEN"),
    )
    message = HumanMessage(content="Hi, how are you.")
    response = chat.generate([[message], [message]])

    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content
