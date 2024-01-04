from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage


class StreamlitChatMessageHistory(BaseChatMessageHistory):
    """
    Chat message history that stores messages in Streamlit session state.

    Args:
        key: The key to use in Streamlit session state for storing messages.
    """

    def __init__(self, key: str = "langchain_messages"):
        try:
            import streamlit as st
        except ImportError as e:
            raise ImportError(
                "Unable to import streamlit, please run `pip install streamlit`."
            ) from e

        if key not in st.session_state:
            st.session_state[key] = []
        self._messages = st.session_state[key]

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the current list of messages"""
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the session memory"""
        self._messages.append(message)

    def clear(self) -> None:
        """Clear session memory"""
        self._messages.clear()
