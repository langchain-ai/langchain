"""Networkx wrapper for graph operations."""
from __future__ import annotations

from typing import Any, List, NamedTuple, Optional, Tuple

KG_TRIPLE_DELIMITER = "<|>"


class KnowledgeTriple(NamedTuple):
    """A triple in the graph."""

    subject: str
    predicate: str
    object_: str

    @classmethod
    def from_string(cls, triple_string: str) -> "KnowledgeTriple":
        """Create a KnowledgeTriple from a string."""
        subject, predicate, object_ = triple_string.strip().split(", ")
        subject = subject[1:]
        object_ = object_[:-1]
        return cls(subject, predicate, object_)


def parse_triples(knowledge_str: str) -> List[KnowledgeTriple]:
    """Parse knowledge triples from the knowledge string."""
    knowledge_str = knowledge_str.strip()
    if not knowledge_str or knowledge_str == "NONE":
        return []
    triple_strs = knowledge_str.split(KG_TRIPLE_DELIMITER)
    results = []
    for triple_str in triple_strs:
        try:
            kg_triple = KnowledgeTriple.from_string(triple_str)
        except ValueError:
            continue
        results.append(kg_triple)
    return results


def get_entities(entity_str: str) -> List[str]:
    """Extract entities from entity string."""
    if entity_str.strip() == "NONE":
        return []
    else:
        return [w.strip() for w in entity_str.split(",")]


class NetworkxEntityGraph:
    """Networkx wrapper for entity graph operations."""

    def __init__(self, graph: Optional[Any] = None) -> None:
        """Create a new graph."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "Could not import networkx python package. "
                "Please install it with `pip install networkx`."
            )
        if graph is not None:
            if not isinstance(graph, nx.DiGraph):
                raise ValueError("Passed in graph is not of correct shape")
            self._graph = graph
        else:
            self._graph = nx.DiGraph()

    @classmethod
    def from_gml(cls, gml_path: str) -> NetworkxEntityGraph:
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "Could not import networkx python package. "
                "Please install it with `pip install networkx`."
            )
        graph = nx.read_gml(gml_path)
        return cls(graph)

    def add_triple(self, knowledge_triple: KnowledgeTriple) -> None:
        """Add a triple to the graph."""
        # Creates nodes if they don't exist
        # Overwrites existing edges
        if not self._graph.has_node(knowledge_triple.subject):
            self._graph.add_node(knowledge_triple.subject)
        if not self._graph.has_node(knowledge_triple.object_):
            self._graph.add_node(knowledge_triple.object_)
        self._graph.add_edge(
            knowledge_triple.subject,
            knowledge_triple.object_,
            relation=knowledge_triple.predicate,
        )

    def delete_triple(self, knowledge_triple: KnowledgeTriple) -> None:
        """Delete a triple from the graph."""
        if self._graph.has_edge(knowledge_triple.subject, knowledge_triple.object_):
            self._graph.remove_edge(knowledge_triple.subject, knowledge_triple.object_)

    def get_triples(self) -> List[Tuple[str, str, str]]:
        """Get all triples in the graph."""
        return [(u, v, d["relation"]) for u, v, d in self._graph.edges(data=True)]

    def get_entity_knowledge(self, entity: str, depth: int = 1) -> List[str]:
        """Get information about an entity."""
        import networkx as nx

        # TODO: Have more information-specific retrieval methods
        if not self._graph.has_node(entity):
            return []

        results = []
        for src, sink in nx.dfs_edges(self._graph, entity, depth_limit=depth):
            relation = self._graph[src][sink]["relation"]
            results.append(f"{src} {relation} {sink}")
        return results

    def write_to_gml(self, path: str) -> None:
        import networkx as nx

        nx.write_gml(self._graph, path)

    def clear(self) -> None:
        """Clear the graph."""
        self._graph.clear()
