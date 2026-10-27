"""InfinityDB Agent Memory implementation for LangChain."""

from __future__ import annotations

from typing import List, Optional

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )
except ImportError:
    class BaseChatMessageHistory:  # type: ignore
        pass

    class BaseMessage:  # type: ignore
        content: str
    class HumanMessage(BaseMessage): pass
    class AIMessage(BaseMessage): pass
    class SystemMessage(BaseMessage): pass

from infinitydb import InfinityDB


class InfinityDBChatMessageHistory(BaseChatMessageHistory):
    """Chat message history backed by InfinityDB's neurosymbolic L0-L6 engine.

    Stores conversation turns with zero external LLM extraction cost ($0),
    providing sub-millisecond retrieval on SQLite WAL mode.
    """

    def __init__(
        self,
        session_id: str,
        db_path: str = "infinity_langchain.db",
        user_id: Optional[str] = None,
    ) -> None:
        self.session_id = session_id
        self.user_id = user_id or session_id
        self.db = InfinityDB(db_path=db_path)

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve all episodic facts and conversation turns for this session."""
        recalled = self.db.recall(query="", n=50, user_id=self.user_id)
        results = recalled.get("results", [])
        
        messages_list: List[BaseMessage] = []
        for item in results:
            fact = item.get("fact", "")
            if fact.startswith("User:") or "user" in fact.lower():
                messages_list.append(HumanMessage(content=fact))
            else:
                messages_list.append(AIMessage(content=fact))
        return messages_list

    def add_message(self, message: BaseMessage) -> None:
        """Append a message to the persistent InfinityDB session."""
        role_prefix = "User: " if isinstance(message, HumanMessage) else "Assistant: "
        text = f"{role_prefix}{message.content}"
        self.db.add(
            conversation=text,
            user_id=self.user_id,
            session_id=self.session_id,
            importance=0.9,
        )

    def add_user_message(self, message: str) -> None:
        """Convenience method to add a user message."""
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Convenience method to add an assistant message."""
        self.add_message(AIMessage(content=message))

    def clear(self) -> None:
        """Wipe session memories in compliance with GDPR Article 17."""
        self.db.delete_user(user_id=self.user_id)
