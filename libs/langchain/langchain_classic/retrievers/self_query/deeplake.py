from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.query_constructors.deeplake import (
        DeepLakeTranslator,
        can_cast_to_float,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "DeepLakeTranslator": "langchain_community.query_constructors.deeplake",
    "can_cast_to_float": "langchain_community.query_constructors.deeplake",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = ["DeepLakeTranslator", "can_cast_to_float"]
