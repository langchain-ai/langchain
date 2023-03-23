from typing import Dict, Type

from langchain.memory.buffer import (
    ConversationBufferMemory,
    ConversationStringBufferMemory,
)
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain.memory.chat_memory import ChatMessageHistory
from langchain.memory.combined import CombinedMemory
from langchain.memory.entity import ConversationEntityMemory
from langchain.memory.kg import ConversationKGMemory
from langchain.memory.readonly import ReadOnlySharedMemory
from langchain.memory.simple import SimpleMemory
from langchain.memory.summary import ConversationSummaryMemory
from langchain.memory.summary_buffer import ConversationSummaryBufferMemory
from langchain.memory.token_buffer import ConversationTokenBufferMemory
from langchain.schema import BaseMemory

__all__ = [
    "CombinedMemory",
    "ConversationBufferWindowMemory",
    "ConversationBufferMemory",
    "SimpleMemory",
    "ConversationSummaryBufferMemory",
    "ConversationKGMemory",
    "ConversationEntityMemory",
    "ConversationSummaryMemory",
    "ChatMessageHistory",
    "ConversationStringBufferMemory",
    "ReadOnlySharedMemory",
    "ConversationTokenBufferMemory",
]

type_to_cls_dict: Dict[str, Type[BaseMemory]] = {
    "conversation_buffer": ConversationBufferMemory,
    "conversation_string_buffer": ConversationStringBufferMemory,
    "conversation_buffer_window": ConversationBufferWindowMemory,
    "conversation_entity": ConversationEntityMemory,
    "simple": SimpleMemory,
    "conversation_summary_buffer": ConversationSummaryBufferMemory,
    "conversation_summary": ConversationSummaryMemory,
    "conversation_token_buffer": ConversationTokenBufferMemory,
}
