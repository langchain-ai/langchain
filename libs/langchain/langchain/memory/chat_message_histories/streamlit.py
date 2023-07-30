from typing import List
from langchain.schema import (
    BaseChatMessageHistory,
)
from langchain.schema.messages import BaseMessage


class StreamlitChatMessageHistory(BaseChatMessageHistory):
    """
    Chat message history that stores messages in Streamlit session state.

    Args:
        message_key: session state key to use for storing messages.
    """

    def __init__(self, message_key: str = "messages"):
        import streamlit as st

        if message_key not in st.session_state:
            st.session_state[message_key] = []
        self._message_key = message_key
        self._state = st.session_state

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the current list of messages"""
        return self._state[self._message_key]

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the session memory"""
        self._state[self._message_key].append(message)

    def clear(self) -> None:
        """Clear session memory"""
        self._state[self._message_key] = []
