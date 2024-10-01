from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.chains.graph_qa.memgraph import (
        MEMGRAPH_GENERATION_PROMPT,
        MEMGRAPH_QA_PROMPT,
        INTERMEDIATE_STEPS_KEY,
        MemgraphQAChain,
        construct_schema,
        extract_cypher,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "MemgraphQAChain": "langchain_community.chains.graph_qa.memgraph",
    "INTERMEDIATE_STEPS_KEY": "langchain_community.chains.graph_qa.memgraph",
    "construct_schema": "langchain_community.chains.graph_qa.memgraph",
    "extract_cypher": "langchain_community.chains.graph_qa.memgraph",
    "MEMGRAPH_GENERATION_PROMPT": "langchain_community.chains.graph_qa.memgraph",
    "MEMGRAPH_QA_PROMPT": "langchain_community.chains.graph_qa.memgraph",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "MemgraphQAChain",
    "INTERMEDIATE_STEPS_KEY",
    "construct_schema",
    "extract_cypher",
    "MEMGRAPH_GENERATION_PROMPT",
    "MEMGRAPH_QA_PROMPT",
]
