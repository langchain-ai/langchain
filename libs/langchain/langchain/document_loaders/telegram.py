from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders import (
        TelegramChatApiLoader,
        TelegramChatFileLoader,
    )
    from langchain_community.document_loaders.telegram import (
        concatenate_rows,
        text_to_docs,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "concatenate_rows": "langchain_community.document_loaders.telegram",
    "TelegramChatFileLoader": "langchain_community.document_loaders",
    "text_to_docs": "langchain_community.document_loaders.telegram",
    "TelegramChatApiLoader": "langchain_community.document_loaders",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "concatenate_rows",
    "TelegramChatFileLoader",
    "text_to_docs",
    "TelegramChatApiLoader",
]
