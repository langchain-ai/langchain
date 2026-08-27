"""Memory modules for conversation prompts."""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer
from langchain_classic.memory.buffer import (
    ConversationBufferMemory,
    ConversationStringBufferMemory,
)
from langchain_classic.memory.buffer_window import ConversationBufferWindowMemory
from langchain_classic.memory.combined import CombinedMemory
from langchain_classic.memory.entity import ConversationEntityMemory
from langchain_classic.memory.summary import ConversationSummaryMemory
from langchain_classic.memory.summary_buffer import ConversationSummaryBufferMemory

if TYPE_CHECKING:
    from langchain_community.memory.kg import ConversationKGMemory

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "ConversationKGMemory": "langchain_community.memory.kg",
}

_importer = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _importer(name)


# This is only for backwards compatibility.

__all__ = [
    "CombinedMemory",
    "ConversationBufferMemory",
    "ConversationBufferWindowMemory",
    "ConversationEntityMemory",
    "ConversationKGMemory",
    "ConversationStringBufferMemory",
    "ConversationSummaryBufferMemory",
    "ConversationSummaryMemory",
]
