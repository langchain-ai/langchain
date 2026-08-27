from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.callbacks.tracers.comet import (
        CometTracer,
        import_comet_llm_api,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "import_comet_llm_api": "langchain_community.callbacks.tracers.comet",
    "CometTracer": "langchain_community.callbacks.tracers.comet",
}

_import_attribute = create_importer(__file__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "CometTracer",
    "import_comet_llm_api",
]
