from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.chains.graph_qa.cypher import (
        CYPHER_GENERATION_PROMPT,
        INTERMEDIATE_STEPS_KEY,
        GraphCypherQAChain,
        construct_schema,
        extract_cypher,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "GraphCypherQAChain": "langchain_community.chains.graph_qa.cypher",
    "INTERMEDIATE_STEPS_KEY": "langchain_community.chains.graph_qa.cypher",
    "construct_schema": "langchain_community.chains.graph_qa.cypher",
    "extract_cypher": "langchain_community.chains.graph_qa.cypher",
    "CYPHER_GENERATION_PROMPT": "langchain_community.chains.graph_qa.cypher",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "GraphCypherQAChain",
    "INTERMEDIATE_STEPS_KEY",
    "construct_schema",
    "extract_cypher",
    "CYPHER_GENERATION_PROMPT",
]
