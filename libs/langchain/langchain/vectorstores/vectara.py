from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.vectorstores import Vectara
    from langchain_community.vectorstores.vectara import VectaraRetriever

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "Vectara": "langchain_community.vectorstores",
    "VectaraRetriever": "langchain_community.vectorstores.vectara",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "Vectara",
    "VectaraRetriever",
]
