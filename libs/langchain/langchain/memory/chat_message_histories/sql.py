from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.chat_message_histories import SQLChatMessageHistory
    from langchain_community.chat_message_histories.sql import (
        BaseMessageConverter,
        DefaultMessageConverter,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "BaseMessageConverter": "langchain_community.chat_message_histories.sql",
    "DefaultMessageConverter": "langchain_community.chat_message_histories.sql",
    "SQLChatMessageHistory": "langchain_community.chat_message_histories",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "BaseMessageConverter",
    "DefaultMessageConverter",
    "SQLChatMessageHistory",
]
