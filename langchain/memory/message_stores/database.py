from typing import List

from langchain.schema import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    MessageDB,
    MessageStore,
)


class DBMessageStore(MessageStore):
    """Using a database with the message store"""

    def __init__(self, message_db: MessageDB, session_id: str):
        self.message_db = message_db
        self.session_id = session_id

    def read(self) -> List[BaseMessage]:
        return self.message_db.read(self.session_id)

    def add_user_message(self, message: HumanMessage) -> None:
        self.message_db.append(self.session_id, message)

    def add_ai_message(self, message: AIMessage) -> None:
        self.message_db.append(self.session_id, message)

    def clear(self) -> None:
        self.message_db.clear(self.session_id)
