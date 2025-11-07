from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.agent_toolkits.openapi.spec import (
        ReducedOpenAPISpec,
        reduce_openapi_spec,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "ReducedOpenAPISpec": "langchain_community.agent_toolkits.openapi.spec",
    "reduce_openapi_spec": "langchain_community.agent_toolkits.openapi.spec",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "ReducedOpenAPISpec",
    "reduce_openapi_spec",
]
