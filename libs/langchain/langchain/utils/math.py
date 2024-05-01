from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.utils.math import (
        cosine_similarity,
        cosine_similarity_top_k,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
# Not marked as deprecated since we may want to move the functionality
# into langchain as long as we're OK with numpy as the dependency.
_MODULE_LOOKUP = {
    "cosine_similarity": "langchain_community.utils.math",
    "cosine_similarity_top_k": "langchain_community.utils.math",
}

_import_attribute = create_importer(__package__, module_lookup=_MODULE_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "cosine_similarity",
    "cosine_similarity_top_k",
]
