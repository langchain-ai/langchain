"""test ChatSenseNova wrapper."""
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage
from langchain_community.chat_models.sense_nova import ChatSenseNova
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler
from langchain_core.callbacks import CallbackManager


def test_chat_sense_nova() -> None:
    chat = ChatSenseNova()
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_sense_nova_with_model() -> None:
    chat = ChatSenseNova(model="SenseChat")
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_sense_nova_with_temperature() -> None:
    chat = ChatSenseNova(model="SenseChat", temperature=1.0)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_sense_nova_with_kwargs() -> None:
    chat = ChatSenseNova(temperature=0.88, top_p=0.7)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_sense_nova_multiple_history() -> None:
    """Tests multiple history works."""
    chat = ChatSenseNova()
    response = chat(
        messages=[
            SystemMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_sense_nova_stream() -> None:
    """Test that stream works."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatSenseNova(streaming=True, callback_manager=callback_manager, temperature=0.8)
    response = chat(
        messages=[
            SystemMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="Who are you?"),
        ],
    )
    assert callback_handler.llm_streams > 0
    assert isinstance(response.content, str)
