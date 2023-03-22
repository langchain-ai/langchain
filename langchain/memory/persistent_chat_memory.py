from abc import ABC, abstractmethod
from typing import Any, Dict, List

import boto3
from botocore.exceptions import ClientError

from langchain.memory.utils import get_buffer_string
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage, ChatMessage
from langchain.memory.chat_memory import BaseChatMemory, ChatMessageHistory

from pydantic import BaseModel

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class MessageStore(ABC):
    session_id: str

    @abstractmethod
    def read(self) -> List[BaseMessage]:
        ...

    @abstractmethod
    def add_user_message(self, message: HumanMessage) -> None:
        ...

    @abstractmethod
    def add_ai_message(self, message: AIMessage) -> None:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

class DynamoDBMessageStore(MessageStore):
    def __init__(self, table_name, session_id: str, region_name: str = "us-east-1"):
        client = boto3.resource("dynamodb", region_name=region_name)
        self.table = client.Table(table_name)
        self.session_id = session_id

    def _read(self) -> List[Dict]:
        try: 
            response = self.table.get_item(
                Key={'SessionId': self.session_id}
            )
            if response and 'Item' in response:
                return response['Item']['History']
        except ClientError as error:
            if error.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.warn("No record found with session id: %s", self.session_id)
            else:
                logger.error(error)
        
        return []

    def read(self) -> List[BaseMessage]:
        items = self._read()
        messages = []
        for item in items:
            role = item["role"]
            content = item["content"]
            if role == 'human':
                messages.append(HumanMessage(content=content))
            elif role == 'ai':
                messages.append(AIMessage(content=content))
            elif role == 'system':
                messages.append(SystemMessage(content=content))
            else:
                messages.append(ChatMessage(content=content, role=role))

        return messages
    
    def add_user_message(self, message: HumanMessage) -> None:
        self._add_message(message)

    def add_ai_message(self, message: AIMessage) -> None:
       self._add_message(message)

    def _add_message(self, message: BaseMessage) -> None:
        messages = self._read()
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            role = "ai"
        elif isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, ChatMessage):
            role = message.role
        else:
            raise ValueError(f"Got unsupported message type: {message}")
        messages.append({"role": role, "content": message.content})
        self._write(self.session_id, messages)

    def _write(self, session_id: str, messages: List[Dict]):
        try:
            self.table.put_item(
                Item={
                    'SessionId': session_id,
                    'History': messages
                }
            )
        except ClientError as err:
            logger.error(err)
    
    def clear(self):
        try:
            self.table.delete_item(
                Key={'SessionId': self.session_id}
            )
        except ClientError as err:
            logger.error(err)

class PersistentChatMessageHistory(ChatMessageHistory):
    store: MessageStore

    class Config:
        arbitrary_types_allowed = True
    
    @property
    def messages(self) -> List[BaseMessage]:
        self.store.read()

    def add_user_message(self, message: str) -> None:
        self.store.add_user_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.store.add_ai_message(AIMessage(content=message))

    def clear(self) -> None:
        self.store.clear()


class ConversationBufferPersistentStoreMemory(BaseChatMemory, BaseModel):
    """Buffer for storing conversation memory."""

    chat_memory: PersistentChatMessageHistory
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"  #: :meta private:

    @property
    def buffer(self) -> Any:
        """String buffer of memory."""
        if self.return_messages:
            return self.chat_memory.messages
        else:
            return get_buffer_string(
                self.chat_memory.messages,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]
   
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}