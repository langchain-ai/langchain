"""Test AliCloud Pai Eas Chat Model."""

import os

from langchain_core.callbacks import CallbackManager
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models.pai_eas_endpoint import PaiEasChatEndpoint
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_pai_eas_call() -> None:
    chat = PaiEasChatEndpoint(
        eas_service_url=os.getenv("EAS_SERVICE_URL"),  # type: ignore[arg-type]
        eas_service_token=os.getenv("EAS_SERVICE_TOKEN"),  # type: ignore[arg-type]
    )
    response = chat.invoke([HumanMessage(content="Say foo:")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_multiple_history() -> None:
    """Tests multiple history works."""
    chat = PaiEasChatEndpoint(
        eas_service_url=os.getenv("EAS_SERVICE_URL"),  # type: ignore[arg-type]
        eas_service_token=os.getenv("EAS_SERVICE_TOKEN"),  # type: ignore[arg-type]
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


def test_stream() -> None:
    """Test that stream works."""
    chat = PaiEasChatEndpoint(
        eas_service_url=os.getenv("EAS_SERVICE_URL"),  # type: ignore[arg-type]
        eas_service_token=os.getenv("EAS_SERVICE_TOKEN"),  # type: ignore[arg-type]
        streaming=True,
    )
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    response = chat.invoke(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="Who are you?"),
        ],
        stream=True,
        config={"callbacks": callback_manager},
    )
    assert callback_handler.llm_streams > 0
    assert isinstance(response.content, str)


def test_multiple_messages() -> None:
    """Tests multiple messages works."""
    chat = PaiEasChatEndpoint(
        eas_service_url=os.getenv("EAS_SERVICE_URL"),  # type: ignore[arg-type]
        eas_service_token=os.getenv("EAS_SERVICE_TOKEN"),  # type: ignore[arg-type]
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
