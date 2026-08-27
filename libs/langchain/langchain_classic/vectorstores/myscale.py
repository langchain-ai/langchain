from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.vectorstores import MyScale, MyScaleSettings
    from langchain_community.vectorstores.myscale import MyScaleWithoutJSON

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "MyScaleSettings": "langchain_community.vectorstores",
    "MyScale": "langchain_community.vectorstores",
    "MyScaleWithoutJSON": "langchain_community.vectorstores.myscale",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "MyScale",
    "MyScaleSettings",
    "MyScaleWithoutJSON",
]
