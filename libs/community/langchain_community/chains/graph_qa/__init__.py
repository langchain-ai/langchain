from typing import Any

_LANGCHAIN_DEPENDENT = [
    "ArangoGraphQAChain",
    "GraphQAChain",
    "GraphCypherQAChain",
    "FalkorDBQAChain",
    "GremlinQAChain",
    "HugeGraphQAChain",
    "KuzuQAChain",
    "NebulaGraphQAChain",
    "NeptuneOpenCypherQAChain",
    "NeptuneSparqlQAChain",
    "OntotextGraphDBQAChain",
    "GraphSparqlQAChain",
]

try:
    from langchain.chains.base import Chain
except ImportError:
    __all__ = []

    def __getattr__(name: str) -> Any:
        if name in _LANGCHAIN_DEPENDENT:
            raise ImportError(
                f"Must have `langchain` installed to use {name}. Please install it "
                f"`pip install -U langchain` and re-run your {name} import."
            )
        raise AttributeError()
else:
    from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
    from langchain_community.chains.graph_qa.base import GraphQAChain
    from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
    from langchain_community.chains.graph_qa.falkordb import FalkorDBQAChain
    from langchain_community.chains.graph_qa.gremlin import GremlinQAChain
    from langchain_community.chains.graph_qa.hugegraph import HugeGraphQAChain
    from langchain_community.chains.graph_qa.kuzu import KuzuQAChain
    from langchain_community.chains.graph_qa.nebulagraph import NebulaGraphQAChain
    from langchain_community.chains.graph_qa.neptune_cypher import (
        NeptuneOpenCypherQAChain,
    )
    from langchain_community.chains.graph_qa.neptune_sparql import NeptuneSparqlQAChain
    from langchain_community.chains.graph_qa.ontotext_graphdb import (
        OntotextGraphDBQAChain,
    )
    from langchain_community.chains.graph_qa.sparql import GraphSparqlQAChain

    __all__ = _LANGCHAIN_DEPENDENT
