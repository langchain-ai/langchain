import json
import logging
from typing import List

from sqlalchemy import Column, Integer, Text, create_engine
from sqlalchemy.orm import sessionmaker

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    _message_to_dict,
    messages_from_dict,
)

logger = logging.getLogger(__name__)


def create_message_model(table_name, DynamicBase):  # type: ignore
    # Model decleared inside a function to have a dynamic table name
    class Message(DynamicBase):
        __tablename__ = table_name
        id = Column(Integer, primary_key=True)
        session_id = Column(Text)
        message = Column(Text)

    return Message


class SQLChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        session_id: str,
        connection_string: str,
        table_name: str = "message_store",
    ):
        self.table_name = table_name
        self.connection_string = connection_string
        self.engine = create_engine(connection_string, echo=False)
        self._create_table_if_not_exists()

        self.session_id = session_id
        self.Session = sessionmaker(self.engine)

    def _create_table_if_not_exists(self) -> None:
        DynamicBase = declarative_base()
        self.Message = create_message_model(self.table_name, DynamicBase)
        # Create all does the check for us in case the table exists.
        DynamicBase.metadata.create_all(self.engine)

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve all messages from db"""
        session = self.Session()
        try:
            result = session.query(self.Message).where(
                self.Message.session_id == self.session_id
            )
            items = [json.loads(record.message) for record in result]
            messages = messages_from_dict(items)
            return messages
        finally:
            session.close()

    def add_user_message(self, message: str) -> None:
        self.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.append(AIMessage(content=message))

    def append(self, message: BaseMessage) -> None:
        """Append the message to the record in db"""
        session = self.Session()
        try:
            jsonstr = json.dumps(_message_to_dict(message))
            session.add(self.Message(session_id=self.session_id, message=jsonstr))
            session.commit()
        finally:
            session.close()

    def clear(self) -> None:
        """Clear session memory from db"""
        session = self.Session()
        try:
            session.query(self.Message).filter(
                self.Message.session_id == self.session_id
            ).delete()
            session.commit()
        finally:
            session.close()
