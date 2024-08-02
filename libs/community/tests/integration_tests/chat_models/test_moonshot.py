"""Test Moonshot Chat Model."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langchain_community.chat_models.moonshot import MoonshotChat


def test_default_call() -> None:
    """Test default model call."""
    chat = MoonshotChat()  # type: ignore[call-arg]
    response = chat.invoke([HumanMessage(content="How are you?")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_model() -> None:
    """Test model kwarg works."""
    chat = MoonshotChat(model="moonshot-v1-32k")  # type: ignore[call-arg]
    response = chat.invoke([HumanMessage(content="How are you?")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_multiple_history() -> None:
    """Tests multiple history works."""
    chat = MoonshotChat()  # type: ignore[call-arg]

    response = chat.invoke(
        [
            HumanMessage(content="How are you?"),
            AIMessage(content="I'm fine, and you?"),
            HumanMessage(content="Not bad!"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)
