from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.chains.graph_qa.gremlin import (
        GRAPHDB_SPARQL_FIX_TEMPLATE,
        INTERMEDIATE_STEPS_KEY,
        GremlinQAChain,
        extract_gremlin,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "GRAPHDB_SPARQL_FIX_TEMPLATE": "langchain_community.chains.graph_qa.gremlin",
    "GremlinQAChain": "langchain_community.chains.graph_qa.gremlin",
    "INTERMEDIATE_STEPS_KEY": "langchain_community.chains.graph_qa.gremlin",
    "extract_gremlin": "langchain_community.chains.graph_qa.gremlin",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "GRAPHDB_SPARQL_FIX_TEMPLATE",
    "INTERMEDIATE_STEPS_KEY",
    "GremlinQAChain",
    "extract_gremlin",
]
