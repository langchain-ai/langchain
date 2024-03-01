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
        self._key = key

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve the current list of messages"""
        import streamlit as st

        return st.session_state[self._key]

    @messages.setter
    def messages(self, value: List[BaseMessage]) -> None:
        """Set the messages list with a new value"""
        import streamlit as st

        st.session_state[self._key] = value

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the session memory"""
        self.messages.append(message)

    def clear(self) -> None:
        """Clear session memory"""
        self.messages.clear()
