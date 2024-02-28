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
    """Networkx wrapper for entity graph operations.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

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

    def clear_edges(self) -> None:
        """Clear the graph edges."""
        self._graph.clear_edges()

    def add_node(self, node: str) -> None:
        """Add node in the graph."""
        self._graph.add_node(node)

    def remove_node(self, node: str) -> None:
        """Remove node from the graph."""
        if self._graph.has_node(node):
            self._graph.remove_node(node)

    def has_node(self, node: str) -> bool:
        """Return if graph has the given node."""
        return self._graph.has_node(node)

    def remove_edge(self, source_node: str, destination_node: str) -> None:
        """Remove edge from the graph."""
        self._graph.remove_edge(source_node, destination_node)

    def has_edge(self, source_node: str, destination_node: str) -> bool:
        """Return if graph has an edge between the given nodes."""
        if self._graph.has_node(source_node) and self._graph.has_node(destination_node):
            return self._graph.has_edge(source_node, destination_node)
        else:
            return False

    def get_neighbors(self, node: str) -> List[str]:
        """Return the neighbor nodes of the given node."""
        return self._graph.neighbors(node)

    def get_number_of_nodes(self) -> int:
        """Get number of nodes in the graph."""
        return self._graph.number_of_nodes()

    def get_topological_sort(self) -> List[str]:
        """Get a list of entity names in the graph sorted by causal dependence."""
        import networkx as nx

        return list(nx.topological_sort(self._graph))

    def draw_graphviz(self, **kwargs: Any) -> None:
        """
        Provides better drawing

        Usage in a jupyter notebook:

            >>> from IPython.display import SVG
            >>> self.draw_graphviz_svg(layout="dot", filename="web.svg")
            >>> SVG('web.svg')
        """
        from networkx.drawing.nx_agraph import to_agraph

        try:
            import pygraphviz  # noqa: F401

        except ImportError as e:
            if e.name == "_graphviz":
                """
                >>> e.msg  # pygraphviz throws this error
                ImportError: libcgraph.so.6: cannot open shared object file
                """
                raise ImportError(
                    "Could not import graphviz debian package. "
                    "Please install it with:"
                    "`sudo apt-get update`"
                    "`sudo apt-get install graphviz graphviz-dev`"
                )
            else:
                raise ImportError(
                    "Could not import pygraphviz python package. "
                    "Please install it with:"
                    "`pip install pygraphviz`."
                )

        graph = to_agraph(self._graph)  # --> pygraphviz.agraph.AGraph
        # pygraphviz.github.io/documentation/stable/tutorial.html#layout-and-drawing
        graph.layout(prog=kwargs.get("prog", "dot"))
        graph.draw(kwargs.get("path", "graph.svg"))
