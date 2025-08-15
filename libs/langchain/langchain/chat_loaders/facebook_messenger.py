from typing import TYPE_CHECKING, Any

from langchain._api.module_import import create_importer

if TYPE_CHECKING:
    from langchain_community.chat_loaders.facebook_messenger import (
        FolderFacebookMessengerChatLoader,
        SingleFileFacebookMessengerChatLoader,
    )

module_lookup = {
    "SingleFileFacebookMessengerChatLoader": (
        "langchain_community.chat_loaders.facebook_messenger"
    ),
    "FolderFacebookMessengerChatLoader": (
        "langchain_community.chat_loaders.facebook_messenger"
    ),
}

# Temporary code for backwards compatibility for deprecated imports.
# This will eventually be removed.
import_lookup = create_importer(
    __package__,
    deprecated_lookups=module_lookup,
)


def __getattr__(name: str) -> Any:
    return import_lookup(name)


__all__ = ["FolderFacebookMessengerChatLoader", "SingleFileFacebookMessengerChatLoader"]
