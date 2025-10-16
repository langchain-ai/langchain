from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.utilities import ArceeWrapper
    from langchain_community.utilities.arcee import (
        ArceeDocument,
        ArceeDocumentAdapter,
        ArceeDocumentSource,
        ArceeRoute,
        DALMFilter,
        DALMFilterType,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "ArceeRoute": "langchain_community.utilities.arcee",
    "DALMFilterType": "langchain_community.utilities.arcee",
    "DALMFilter": "langchain_community.utilities.arcee",
    "ArceeDocumentSource": "langchain_community.utilities.arcee",
    "ArceeDocument": "langchain_community.utilities.arcee",
    "ArceeDocumentAdapter": "langchain_community.utilities.arcee",
    "ArceeWrapper": "langchain_community.utilities",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "ArceeDocument",
    "ArceeDocumentAdapter",
    "ArceeDocumentSource",
    "ArceeRoute",
    "ArceeWrapper",
    "DALMFilter",
    "DALMFilterType",
]
