from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders.parsers.audio import (
        OpenAIWhisperParser,
        OpenAIWhisperParserLocal,
        YandexSTTParser,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "OpenAIWhisperParser": "langchain_community.document_loaders.parsers.audio",
    "OpenAIWhisperParserLocal": "langchain_community.document_loaders.parsers.audio",
    "YandexSTTParser": "langchain_community.document_loaders.parsers.audio",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "OpenAIWhisperParser",
    "OpenAIWhisperParserLocal",
    "YandexSTTParser",
]
