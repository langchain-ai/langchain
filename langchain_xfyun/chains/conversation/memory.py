"""Memory modules for conversation prompts."""

from langchain_xfyun.memory.buffer import (
    ConversationBufferMemory,
    ConversationStringBufferMemory,
)
from langchain_xfyun.memory.buffer_window import ConversationBufferWindowMemory
from langchain_xfyun.memory.combined import CombinedMemory
from langchain_xfyun.memory.entity import ConversationEntityMemory
from langchain_xfyun.memory.kg import ConversationKGMemory
from langchain_xfyun.memory.summary import ConversationSummaryMemory
from langchain_xfyun.memory.summary_buffer import ConversationSummaryBufferMemory

# This is only for backwards compatibility.

__all__ = [
    "ConversationSummaryBufferMemory",
    "ConversationSummaryMemory",
    "ConversationKGMemory",
    "ConversationBufferWindowMemory",
    "ConversationEntityMemory",
    "ConversationBufferMemory",
    "CombinedMemory",
    "ConversationStringBufferMemory",
]
