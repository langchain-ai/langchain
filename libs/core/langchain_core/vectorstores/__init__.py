"""Vector stores."""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core.vectorstores.base import VST, VectorStore, VectorStoreRetriever
    from langchain_core.vectorstores.in_memory import InMemoryVectorStore

__all__ = (
    "VST",
    "InMemoryVectorStore",
    "VectorStore",
    "VectorStoreRetriever",
)

_dynamic_imports = {
    "VectorStore": "base",
    "VST": "base",
    "VectorStoreRetriever": "base",
    "InMemoryVectorStore": "in_memory",
}


def __getattr__(attr_name: str) -> object:
    """Dynamically import and return an attribute from a submodule.

    This function enables lazy loading of vectorstore classes from submodules, reducing
    initial import time and circular dependency issues.

    Args:
        attr_name: Name of the attribute to import.

    Returns:
        The imported attribute object.

    Raises:
        AttributeError: If the attribute is not found in `_dynamic_imports`.
    """
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    """Return a list of available attributes for this module.

    Returns:
        List of attribute names that can be imported from this module.
    """
    return list(__all__)
