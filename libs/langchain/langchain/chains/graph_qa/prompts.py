from typing import Any

DEPRECATED_IMPORTS = [
    "ENTITY_EXTRACTION_PROMPT",
    "GRAPH_QA_PROMPT",
    "CYPHER_GENERATION_TEMPLATE",
    "CYPHER_GENERATION_PROMPT",
    "NEBULAGRAPH_EXTRA_INSTRUCTIONS",
    "NGQL_GENERATION_TEMPLATE",
    "NGQL_GENERATION_PROMPT",
    "KUZU_EXTRA_INSTRUCTIONS",
    "KUZU_GENERATION_TEMPLATE",
    "KUZU_GENERATION_PROMPT",
    "GREMLIN_GENERATION_TEMPLATE",
    "GREMLIN_GENERATION_PROMPT",
    "CYPHER_QA_TEMPLATE",
    "CYPHER_QA_PROMPT",
    "SPARQL_INTENT_TEMPLATE",
    "SPARQL_INTENT_PROMPT",
    "SPARQL_GENERATION_SELECT_TEMPLATE",
    "SPARQL_GENERATION_SELECT_PROMPT",
    "SPARQL_GENERATION_UPDATE_TEMPLATE",
    "SPARQL_GENERATION_UPDATE_PROMPT",
    "SPARQL_QA_TEMPLATE",
    "SPARQL_QA_PROMPT",
    "GRAPHDB_SPARQL_GENERATION_TEMPLATE",
    "GRAPHDB_SPARQL_GENERATION_PROMPT",
    "GRAPHDB_SPARQL_FIX_TEMPLATE",
    "GRAPHDB_SPARQL_FIX_PROMPT",
    "GRAPHDB_QA_TEMPLATE",
    "GRAPHDB_QA_PROMPT",
    "AQL_GENERATION_TEMPLATE",
    "AQL_GENERATION_PROMPT",
    "AQL_FIX_TEMPLATE",
    "AQL_FIX_PROMPT",
    "AQL_QA_TEMPLATE",
    "AQL_QA_PROMPT",
    "NEPTUNE_OPENCYPHER_EXTRA_INSTRUCTIONS",
    "NEPTUNE_OPENCYPHER_GENERATION_TEMPLATE",
    "NEPTUNE_OPENCYPHER_GENERATION_PROMPT",
    "NEPTUNE_OPENCYPHER_GENERATION_SIMPLE_TEMPLATE",
    "NEPTUNE_OPENCYPHER_GENERATION_SIMPLE_PROMPT",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.chains.graph_qa.prompts import {name}`"  # noqa: #E501
        )

    raise AttributeError()
