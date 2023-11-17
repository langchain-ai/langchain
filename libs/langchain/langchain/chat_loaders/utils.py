"""Utilities for chat loaders."""
from copy import deepcopy
from typing import Iterable, Iterator, List

from langchain.schema.chat import ChatSession
from langchain.schema.messages import AIMessage, BaseMessage


def merge_chat_runs_in_session(
    chat_session: ChatSession, delimiter: str = "\n\n"
) -> ChatSession:
    """Merge chat runs together in a chat session.

    A chat run is a sequence of messages from the same sender.

    Args:
        chat_session: A chat session.

    Returns:
        A chat session with merged chat runs.
    """
    messages: List[BaseMessage] = []
    for message in chat_session["messages"]:
        if not isinstance(message.content, str):
            raise ValueError(
                "Chat Loaders only support messages with content type string, "
                f"got {message.content}"
            )
        if not messages:
            messages.append(deepcopy(message))
        elif (
            isinstance(message, type(messages[-1]))
            and messages[-1].additional_kwargs.get("sender") is not None
            and messages[-1].additional_kwargs["sender"]
            == message.additional_kwargs.get("sender")
        ):
            if not isinstance(messages[-1].content, str):
                raise ValueError(
                    "Chat Loaders only support messages with content type string, "
                    f"got {messages[-1].content}"
                )
            messages[-1].content = (
                messages[-1].content + delimiter + message.content
            ).strip()
            messages[-1].additional_kwargs.get("events", []).extend(
                message.additional_kwargs.get("events") or []
            )
        else:
            messages.append(deepcopy(message))
    return ChatSession(messages=messages)


def merge_chat_runs(chat_sessions: Iterable[ChatSession]) -> Iterator[ChatSession]:
    """Merge chat runs together.

    A chat run is a sequence of messages from the same sender.

    Args:
        chat_sessions: A list of chat sessions.

    Returns:
        A list of chat sessions with merged chat runs.
    """
    for chat_session in chat_sessions:
        yield merge_chat_runs_in_session(chat_session)


def map_ai_messages_in_session(chat_sessions: ChatSession, sender: str) -> ChatSession:
    """Convert messages from the specified 'sender' to AI messages.

    This is useful for fine-tuning the AI to adapt to your voice.
    """
    messages = []
    num_converted = 0
    for message in chat_sessions["messages"]:
        if message.additional_kwargs.get("sender") == sender:
            message = AIMessage(
                content=message.content,
                additional_kwargs=message.additional_kwargs.copy(),
                example=getattr(message, "example", None),
            )
            num_converted += 1
        messages.append(message)
    return ChatSession(messages=messages)


def map_ai_messages(
    chat_sessions: Iterable[ChatSession], sender: str
) -> Iterator[ChatSession]:
    """Convert messages from the specified 'sender' to AI messages.

    This is useful for fine-tuning the AI to adapt to your voice.
    """
    for chat_session in chat_sessions:
        yield map_ai_messages_in_session(chat_session, sender)
