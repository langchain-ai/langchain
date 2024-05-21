from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.chains.graph_qa.falkordb import (
        INTERMEDIATE_STEPS_KEY,
        FalkorDBQAChain,
        extract_cypher,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "FalkorDBQAChain": "langchain_community.chains.graph_qa.falkordb",
    "INTERMEDIATE_STEPS_KEY": "langchain_community.chains.graph_qa.falkordb",
    "extract_cypher": "langchain_community.chains.graph_qa.falkordb",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = ["FalkorDBQAChain", "INTERMEDIATE_STEPS_KEY", "extract_cypher"]
