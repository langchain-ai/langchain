"""Memory modules for conversation prompts."""

from langchain.chains.memory.entity import ConversationEntityMemory
from langchain.chains.memory.kg import ConversationKGMemory
from langchain.chains.memory.summary import ConversationSummaryMemory
from langchain.chains.memory.summary_buffer import ConversationSummaryBufferMemory
from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain.memory.combined import CombinedMemory

# This is only for backwards compatibility.

__all__ = [
    "ConversationSummaryBufferMemory",
    "ConversationSummaryMemory",
    "ConversationKGMemory",
    "ConversationBufferWindowMemory",
    "ConversationEntityMemory",
    "ConversationBufferMemory",
    "CombinedMemory",
]
