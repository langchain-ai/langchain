from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.chat_loaders.utils import (
        map_ai_messages,
        map_ai_messages_in_session,
        merge_chat_runs,
        merge_chat_runs_in_session,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "merge_chat_runs_in_session": "langchain_community.chat_loaders.utils",
    "merge_chat_runs": "langchain_community.chat_loaders.utils",
    "map_ai_messages_in_session": "langchain_community.chat_loaders.utils",
    "map_ai_messages": "langchain_community.chat_loaders.utils",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "map_ai_messages",
    "map_ai_messages_in_session",
    "merge_chat_runs",
    "merge_chat_runs_in_session",
]
