from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.chains.graph_qa.neptune_cypher import (
        INTERMEDIATE_STEPS_KEY,
        NeptuneOpenCypherQAChain,
        extract_cypher,
        trim_query,
        use_simple_prompt,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "INTERMEDIATE_STEPS_KEY": "langchain_community.chains.graph_qa.neptune_cypher",
    "NeptuneOpenCypherQAChain": "langchain_community.chains.graph_qa.neptune_cypher",
    "extract_cypher": "langchain_community.chains.graph_qa.neptune_cypher",
    "trim_query": "langchain_community.chains.graph_qa.neptune_cypher",
    "use_simple_prompt": "langchain_community.chains.graph_qa.neptune_cypher",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "INTERMEDIATE_STEPS_KEY",
    "NeptuneOpenCypherQAChain",
    "extract_cypher",
    "trim_query",
    "use_simple_prompt",
]
