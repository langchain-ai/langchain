from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
)

if TYPE_CHECKING:
    from zep_python import Memory, MemorySearchResult, Message, NotFoundError

logger = logging.getLogger(__name__)


class ZepChatMessageHistory(BaseChatMessageHistory):
    """A ChatMessageHistory implementation that uses Zep as a backend.

    Recommended usage::

        # Set up Zep Chat History
        zep_chat_history = ZepChatMessageHistory(
            session_id=session_id,
            url=ZEP_API_URL,
        )

        # Use a standard ConversationBufferMemory to encapsulate the Zep chat history
        memory = ConversationBufferMemory(
            memory_key="chat_history", chat_memory=zep_chat_history
        )


    Zep provides long-term conversation storage for LLM apps. The server stores,
    summarizes, embeds, indexes, and enriches conversational AI chat
    histories, and exposes them via simple, low-latency APIs.

    For server installation instructions and more, see: https://getzep.github.io/

    This class is a thin wrapper around the zep-python package. Additional
    Zep functionality is exposed via the `zep_summary` and `zep_messages`
    properties.

    For more information on the zep-python package, see:
    https://github.com/getzep/zep-python
    """

    def __init__(
        self,
        session_id: str,
        url: str = "http://localhost:8000",
    ) -> None:
        try:
            from zep_python import ZepClient
        except ImportError:
            raise ValueError(
                "Could not import zep-python package. "
                "Please install it with `pip install zep-python`."
            )

        self.zep_client = ZepClient(base_url=url)
        self.session_id = session_id

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve messages from Zep memory"""
        zep_memory: Optional[Memory] = self._get_memory()
        if not zep_memory:
            return []

        messages: List[BaseMessage] = []
        # Extract summary, if present, and messages
        if zep_memory.summary:
            if len(zep_memory.summary.content) > 0:
                messages.append(HumanMessage(content=zep_memory.summary.content))
        if zep_memory.messages:
            msg: Message
            for msg in zep_memory.messages:
                if msg.role == "ai":
                    messages.append(AIMessage(content=msg.content))
                else:
                    messages.append(HumanMessage(content=msg.content))

        return messages

    @property
    def zep_messages(self) -> List[Message]:
        """Retrieve summary from Zep memory"""
        zep_memory: Optional[Memory] = self._get_memory()
        if not zep_memory:
            return []

        return zep_memory.messages

    @property
    def zep_summary(self) -> Optional[str]:
        """Retrieve summary from Zep memory"""
        zep_memory: Optional[Memory] = self._get_memory()
        if not zep_memory or not zep_memory.summary:
            return None

        return zep_memory.summary.content

    def _get_memory(self) -> Optional[Memory]:
        """Retrieve memory from Zep"""
        from zep_python import NotFoundError

        try:
            zep_memory: Memory = self.zep_client.get_memory(self.session_id)
        except NotFoundError:
            logger.warning(
                f"Session {self.session_id} not found in Zep. Returning None"
            )
            return None
        return zep_memory

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the Zep memory history"""
        from zep_python import Memory, Message

        zep_message: Message
        if isinstance(message, HumanMessage):
            zep_message = Message(content=message.content, role="human")
        else:
            zep_message = Message(content=message.content, role="ai")

        zep_memory = Memory(messages=[zep_message])

        self.zep_client.add_memory(self.session_id, zep_memory)

    def search(
        self, query: str, metadata: Optional[Dict] = None, limit: Optional[int] = None
    ) -> List[MemorySearchResult]:
        """Search Zep memory for messages matching the query"""
        from zep_python import MemorySearchPayload

        payload: MemorySearchPayload = MemorySearchPayload(
            text=query, metadata=metadata
        )

        return self.zep_client.search_memory(self.session_id, payload, limit=limit)

    def clear(self) -> None:
        """Clear session memory from Zep. Note that Zep is long-term storage for memory
        and this is not advised unless you have specific data retention requirements.
        """
        try:
            self.zep_client.delete_memory(self.session_id)
        except NotFoundError:
            logger.warning(
                f"Session {self.session_id} not found in Zep. Skipping delete."
            )
