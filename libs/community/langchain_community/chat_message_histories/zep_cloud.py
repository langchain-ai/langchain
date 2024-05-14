from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)

if TYPE_CHECKING:
    from zep_cloud import (
        Memory,
        MemoryGetRequestMemoryType,
        MemorySearchResult,
        Message,
        NotFoundError,
        RoleType,
        SearchScope,
        SearchType,
    )

logger = logging.getLogger(__name__)


def condense_zep_memory_into_human_message(zep_memory: Memory) -> BaseMessage:
    prompt = ""
    if zep_memory.facts:
        prompt = "\n".join(zep_memory.facts)
    if zep_memory.summary and zep_memory.summary.content:
        prompt += "\n" + zep_memory.summary.content
    for msg in zep_memory.messages or []:
        prompt += f"\n{msg.role or msg.role_type}: {msg.content}"
    return HumanMessage(content=prompt)


def get_zep_message_role_type(role: str) -> RoleType:
    if role == "human":
        return "user"
    elif role == "ai":
        return "assistant"
    elif role == "system":
        return "system"
    elif role == "function":
        return "function"
    elif role == "tool":
        return "tool"
    else:
        return "system"


class ZepCloudChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that uses Zep Cloud as a backend.

    Recommended usage::

        # Set up Zep Chat History
        zep_chat_history = ZepChatMessageHistory(
            session_id=session_id,
            api_key=<your_api_key>,
        )

        # Use a standard ConversationBufferMemory to encapsulate the Zep chat history
        memory = ConversationBufferMemory(
            memory_key="chat_history", chat_memory=zep_chat_history
        )

    Zep - Recall, understand, and extract data from chat histories.
    Power personalized AI experiences.

    Zep is a long-term memory service for AI Assistant apps.
    With Zep, you can provide AI assistants with the
    ability to recall past conversations,
    no matter how distant,
    while also reducing hallucinations, latency, and cost.

    see Zep Cloud Docs: https://help.getzep.com

    This class is a thin wrapper around the zep-python package. Additional
    Zep functionality is exposed via the `zep_summary`, `zep_messages` and `zep_facts`
    properties.

    For more information on the zep-python package, see:
    https://github.com/getzep/zep-python
    """

    def __init__(
        self,
        session_id: str,
        api_key: str,
        *,
        memory_type: Optional[MemoryGetRequestMemoryType] = None,
        lastn: Optional[int] = None,
        ai_prefix: Optional[str] = None,
        human_prefix: Optional[str] = None,
        summary_instruction: Optional[str] = None,
    ) -> None:
        try:
            from zep_cloud.client import AsyncZep, Zep
        except ImportError:
            raise ImportError(
                "Could not import zep-cloud package. "
                "Please install it with `pip install zep-cloud`."
            )

        self.zep_client = Zep(api_key=api_key)
        self.zep_client_async = AsyncZep(api_key=api_key)
        self.session_id = session_id

        self.memory_type = memory_type or "perpetual"
        self.lastn = lastn
        self.ai_prefix = ai_prefix or "ai"
        self.human_prefix = human_prefix or "human"
        self.summary_instruction = summary_instruction

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve messages from Zep memory"""
        zep_memory: Optional[Memory] = self._get_memory()
        if not zep_memory:
            return []

        return [condense_zep_memory_into_human_message(zep_memory)]

    @property
    def zep_messages(self) -> List[Message]:
        """Retrieve summary from Zep memory"""
        zep_memory: Optional[Memory] = self._get_memory()
        if not zep_memory:
            return []

        return zep_memory.messages or []

    @property
    def zep_summary(self) -> Optional[str]:
        """Retrieve summary from Zep memory"""
        zep_memory: Optional[Memory] = self._get_memory()
        if not zep_memory or not zep_memory.summary:
            return None

        return zep_memory.summary.content

    @property
    def zep_facts(self) -> Optional[List[str]]:
        """Retrieve conversation facts from Zep memory"""
        if self.memory_type != "perpetual":
            return None
        zep_memory: Optional[Memory] = self._get_memory()
        if not zep_memory or not zep_memory.facts:
            return None

        return zep_memory.facts

    def _get_memory(self) -> Optional[Memory]:
        """Retrieve memory from Zep"""
        from zep_cloud import NotFoundError

        try:
            zep_memory: Memory = self.zep_client.memory.get(
                self.session_id, memory_type=self.memory_type, lastn=self.lastn
            )
        except NotFoundError:
            logger.warning(
                f"Session {self.session_id} not found in Zep. Returning None"
            )
            return None
        return zep_memory

    def add_user_message(  # type: ignore[override]
        self, message: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Convenience method for adding a human message string to the store.

        Args:
            message: The string contents of a human message.
            metadata: Optional metadata to attach to the message.
        """
        self.add_message(HumanMessage(content=message), metadata=metadata)

    def add_ai_message(  # type: ignore[override]
        self, message: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Convenience method for adding an AI message string to the store.

        Args:
            message: The string contents of an AI message.
            metadata: Optional metadata to attach to the message.
        """
        self.add_message(AIMessage(content=message), metadata=metadata)

    def add_message(
        self, message: BaseMessage, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Append the message to the Zep memory history"""
        from zep_cloud import Message

        self.zep_client.memory.add(
            self.session_id,
            messages=[
                Message(
                    content=str(message.content),
                    role=message.type,
                    role_type=get_zep_message_role_type(message.type),
                    metadata=metadata,
                )
            ],
        )

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Append the messages to the Zep memory history"""
        from zep_cloud import Message

        zep_messages = [
            Message(
                content=str(message.content),
                role=message.type,
                role_type=get_zep_message_role_type(message.type),
                metadata=message.additional_kwargs.get("metadata", None),
            )
            for message in messages
        ]

        self.zep_client.memory.add(self.session_id, messages=zep_messages)

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Append the messages to the Zep memory history asynchronously"""
        from zep_cloud import Message

        zep_messages = [
            Message(
                content=str(message.content),
                role=message.type,
                role_type=get_zep_message_role_type(message.type),
                metadata=message.additional_kwargs.get("metadata", None),
            )
            for message in messages
        ]

        await self.zep_client_async.memory.add(self.session_id, messages=zep_messages)

    def search(
        self,
        query: str,
        metadata: Optional[Dict] = None,
        search_scope: SearchScope = "messages",
        search_type: SearchType = "similarity",
        mmr_lambda: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[MemorySearchResult]:
        """Search Zep memory for messages matching the query"""

        return self.zep_client.memory.search(
            self.session_id,
            text=query,
            metadata=metadata,
            search_scope=search_scope,
            search_type=search_type,
            mmr_lambda=mmr_lambda,
            limit=limit,
        )

    def clear(self) -> None:
        """Clear session memory from Zep. Note that Zep is long-term storage for memory
        and this is not advised unless you have specific data retention requirements.
        """
        try:
            self.zep_client.memory.delete(self.session_id)
        except NotFoundError:
            logger.warning(
                f"Session {self.session_id} not found in Zep. Skipping delete."
            )

    async def aclear(self) -> None:
        """Clear session memory from Zep asynchronously.
        Note that Zep is long-term storage for memory and this is not advised
        unless you have specific data retention requirements.
        """
        try:
            await self.zep_client_async.memory.delete(self.session_id)
        except NotFoundError:
            logger.warning(
                f"Session {self.session_id} not found in Zep. Skipping delete."
            )
