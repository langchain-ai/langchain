"""Test LangChain+ Client Utils."""

from typing import List

from langchain.client.utils import parse_chat_messages
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)


def test_parse_chat_messages() -> None:
    """Test that chat messages are parsed correctly."""
    input_text = (
        "Human: I am human roar\nAI: I am AI beep boop\nSystem: I am a system message"
    )
    expected = [
        HumanMessage(content="I am human roar"),
        AIMessage(content="I am AI beep boop"),
        SystemMessage(content="I am a system message"),
    ]
    assert parse_chat_messages(input_text) == expected


def test_parse_chat_messages_empty_input() -> None:
    """Test that an empty input string returns an empty list."""
    input_text = ""
    expected: List[BaseMessage] = []
    assert parse_chat_messages(input_text) == expected


def test_parse_chat_messages_multiline_messages() -> None:
    """Test that multiline messages are parsed correctly."""
    input_text = (
        "Human: I am a human\nand I roar\nAI: I am an AI\nand I"
        " beep boop\nSystem: I am a system\nand a message"
    )
    expected = [
        HumanMessage(content="I am a human\nand I roar"),
        AIMessage(content="I am an AI\nand I beep boop"),
        SystemMessage(content="I am a system\nand a message"),
    ]
    assert parse_chat_messages(input_text) == expected


def test_parse_chat_messages_custom_roles() -> None:
    """Test that custom roles are parsed correctly."""
    input_text = "Client: I need help\nAgent: I'm here to help\nClient: Thank you"
    expected = [
        ChatMessage(role="Client", content="I need help"),
        ChatMessage(role="Agent", content="I'm here to help"),
        ChatMessage(role="Client", content="Thank you"),
    ]
    assert parse_chat_messages(input_text, roles=["Client", "Agent"]) == expected


def test_parse_chat_messages_embedded_roles() -> None:
    """Test that messages with embedded role references are parsed correctly."""
    input_text = (
        "Human: Oh ai what if you said AI: foo bar?"
        "\nAI: Well, that would be interesting!"
    )
    expected = [
        HumanMessage(content="Oh ai what if you said AI: foo bar?"),
        AIMessage(content="Well, that would be interesting!"),
    ]
    assert parse_chat_messages(input_text) == expected
