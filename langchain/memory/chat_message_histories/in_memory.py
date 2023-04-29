from typing import List

from pydantic import BaseModel

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    MessageLog
)

class ChatMessageHistory(BaseChatMessageHistory):
    def load_message_logs(self) -> List[MessageLog]:
        # No database to load from
        return []
    
    def save_message_log(self, message_log: MessageLog) -> None:
        # No database to save to
        return None

    def _clear(self) -> None:
        # We have no database
        return None
