"""Test volc engine maas chat model."""

from langchain_core.callbacks import CallbackManager
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models.volcengine_maas import VolcEngineMaasChat
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_default_call() -> None:
    """Test valid chat call to volc engine."""
    chat = VolcEngineMaasChat()  # type: ignore[call-arg]
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_multiple_history() -> None:
    """Tests multiple history works."""
    chat = VolcEngineMaasChat()  # type: ignore[call-arg]

    response = chat.invoke(
        [
            HumanMessage(content="Hello"),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_stream() -> None:
    """Test that stream works."""
    chat = VolcEngineMaasChat(streaming=True)  # type: ignore[call-arg]
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    response = chat.invoke(
        [
            HumanMessage(content="Hello"),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you?"),
        ],
        stream=True,
        config={"callbacks": callback_manager},
    )
    assert callback_handler.llm_streams > 0
    assert isinstance(response.content, str)


def test_stop() -> None:
    """Test that stop works."""
    chat = VolcEngineMaasChat(  # type: ignore[call-arg]
        model="skylark2-pro-4k", model_version="1.2", streaming=True
    )
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    response = chat.invoke(
        [
            HumanMessage(content="repeat: hello world"),
            AIMessage(content="hello world"),
            HumanMessage(content="repeat: hello world"),
        ],
        stream=True,
        config={"callbacks": callback_manager},
        stop=["world"],
    )
    assert callback_handler.llm_streams > 0
    assert isinstance(response.content, str)
    assert response.content.rstrip() == "hello"


def test_multiple_messages() -> None:
    """Tests multiple messages works."""
    chat = VolcEngineMaasChat()  # type: ignore[call-arg]
    message = HumanMessage(content="Hi, how are you?")
    response = chat.generate([[message], [message]])

    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content
