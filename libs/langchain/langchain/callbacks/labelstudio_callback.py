from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.callbacks.labelstudio_callback import (
        LabelStudioCallbackHandler,
        LabelStudioMode,
        get_default_label_configs,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "LabelStudioMode": "langchain_community.callbacks.labelstudio_callback",
    "get_default_label_configs": "langchain_community.callbacks.labelstudio_callback",
    "LabelStudioCallbackHandler": "langchain_community.callbacks.labelstudio_callback",
}

_import_attribute = create_importer(__file__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "LabelStudioMode",
    "get_default_label_configs",
    "LabelStudioCallbackHandler",
]
