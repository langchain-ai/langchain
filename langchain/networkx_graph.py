"""Networkx wrapper for graph operations."""

from typing import List, NamedTuple, Tuple
import networkx  as nx

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

class NetworkxEntityGraph:
    """Networkx wrapper for entity graph operations."""

    def __init__(self):
        """Create a new graph."""
        self._graph = nx.DiGraph()

    def add_triple(self, knowledge_triple: KnowledgeTriple) -> None:
        """Add a triple to the graph."""
        # Creates nodes if they don't exist
        # Overwrites existing edges
        if not self._graph.has_node(knowledge_triple.subject):
            self._graph.add_node(knowledge_triple.subject)
        if not self._graph.has_node(knowledge_triple.object_):
            self._graph.add_node(knowledge_triple.object_)
        self._graph.add_edge(knowledge_triple.subject, knowledge_triple.object_, relation=knowledge_triple.predicate)
    
    def delete_triple(self, knowledge_triple: KnowledgeTriple) -> None:
        """Delete a triple from the graph."""
        if self._graph.has_edge(knowledge_triple.subject, knowledge_triple.object_):
            self._graph.remove_edge(knowledge_triple.subject, knowledge_triple.object_)
    
    def get_triples(self) -> List[Tuple[str, str, str]]:
        """Get all triples in the graph."""
        return [(u, v, d["relation"]) for u, v, d in self._graph.edges(data=True)]

    def get_entity_knowledge(self, entity: str, depth: int = 1) -> List[str]:
        """Get information about an entity."""
        # TODO: Have more information-specific retrieval methods
        if not self._graph.has_node(entity):
            return []

        results= []
        for src, sink in nx.dfs_edges(self._graph, entity, depth_limit=depth):
            relation = self._graph[src][sink]["relation"]
            results.append(f"{src} {relation} {sink}")
        return results


    def clear(self) -> None:
        """Clear the graph."""
        self._graph.clear()
    