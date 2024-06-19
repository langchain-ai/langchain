from typing import List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage


class StreamlitChatMessageHistory(BaseChatMessageHistory):
    """
    Chat message history that stores messages in Streamlit session state.

    Args:
        key: The key to use in Streamlit session state for storing messages.
        history_size: Maximum number fo messages to retrieve. If None then
            there is no limit. If not None then only the latest `history_size`
            messages are retrieved.
    """

    def __init__(
        self, key: str = "langchain_messages", *, history_size: Optional[int] = None
    ):
        try:
            import streamlit as st
        except ImportError as e:
            raise ImportError(
                "Unable to import streamlit, please run `pip install streamlit`."
            ) from e

        if key not in st.session_state:
            st.session_state[key] = []
        self._messages = st.session_state[key]
        self._key = key
        self._history_size = history_size

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve the current list of messages"""
        messages = self._messages
        if self._history_size:
            messages = messages[-self._history_size :]
        return messages

    @messages.setter
    def messages(self, value: List[BaseMessage]) -> None:
        """Set the messages list with a new value"""
        import streamlit as st

        st.session_state[self._key] = value
        self._messages = st.session_state[self._key]

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the session memory"""
        self.messages.append(message)

    def clear(self) -> None:
        """Clear session memory"""
        self.messages.clear()
