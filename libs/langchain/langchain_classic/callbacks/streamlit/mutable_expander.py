from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.callbacks.streamlit.mutable_expander import (
        ChildRecord,
        ChildType,
        MutableExpander,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "ChildType": "langchain_community.callbacks.streamlit.mutable_expander",
    "ChildRecord": "langchain_community.callbacks.streamlit.mutable_expander",
    "MutableExpander": "langchain_community.callbacks.streamlit.mutable_expander",
}

_import_attribute = create_importer(__file__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "ChildRecord",
    "ChildType",
    "MutableExpander",
]
