from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import Column, Integer, Text, create_engine,Sequence
from typing import Any, List, Optional
try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base
import json
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)
from abc import ABC, abstractmethod
from datetime import datetime

SF_SEQUENCE_NAME = "message_id_sequence" #"dev_message_sequence"
SF_TABLE_NAME = "message_store" #"dev_message_store"

class BaseMessageConverter(ABC):
    """The class responsible for converting BaseMessage to your SQLAlchemy model."""

    @abstractmethod
    def from_sql_model(self, sql_message: Any) -> BaseMessage:
        """Convert a SQLAlchemy model to a BaseMessage instance."""
        raise NotImplementedError

    @abstractmethod
    def to_sql_model(self, message: BaseMessage, session_id: str) -> Any:
        """Convert a BaseMessage instance to a SQLAlchemy model."""
        raise NotImplementedError

    @abstractmethod
    def get_sql_model_class(self) -> Any:
        """Get the SQLAlchemy model class."""
        raise NotImplementedError

def create_snowflake_message_model(table_name: str, DynamicBase: Any) -> Any:
    """
    Create a message model for a given table name.

    Args:
        table_name: The name of the table to use.
        DynamicBase: The base class to use for the model.

    Returns:
        The model class.

    """

    # Model declared inside a function to have a dynamic table name.
    class Message(DynamicBase):  # type: ignore[valid-type, misc]
        __tablename__ = table_name
        id = Column(Integer, Sequence(SF_SEQUENCE_NAME), primary_key=True,autoincrement=True)
        # id = Column(Integer, primary_key=True)
        session_id = Column(Text)
        message = Column(Text)

    return Message



class SnowflakeMessageConverter(BaseMessageConverter):
    def __init__(self, table_name: str):
        self.model_class = create_snowflake_message_model(table_name, declarative_base())

    def from_sql_model(self, sql_message: Any) -> BaseMessage:
        return messages_from_dict([json.loads(sql_message.message)])[0]

    def to_sql_model(self, message: BaseMessage, session_id: str) -> Any:
        return self.model_class(
            session_id=session_id, message=json.dumps(message_to_dict(message))
        )

    def get_sql_model_class(self) -> Any:
        return self.model_class



class SnowflakeChatMessageHistory(SQLChatMessageHistory):
    def __init__(
        self,
        session_id,
        connection_string,
        table_name:str = SF_TABLE_NAME,
        session_id_field_name:str = "session_id",
        custom_message_converter: Optional[BaseMessageConverter] = None,
    ):
        super().__init__(session_id, connection_string, table_name, session_id_field_name, custom_message_converter)

        self.converter = custom_message_converter or SnowflakeMessageConverter(table_name)
        self._create_sequence_if_not_exists()
        
    def _create_sequence_if_not_exists(self) -> None:
        """Make sure the Snowflake user/role has create sequence privilege."""
        connection = self.engine.connect()
        query = f"create sequence if not exists {SF_SEQUENCE_NAME} start with 1 increment by 1 noorder;"
        connection.execute(query)  
