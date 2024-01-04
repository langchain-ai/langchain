from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Union

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.load.serializable import Serializable
    from langchain_core.pydantic_v1 import Field


class Node(Serializable):
    """Represents a node in a graph with associated properties.

    Attributes:
        id (Union[str, int]): A unique identifier for the node.
        type (str): The type or label of the node, default is "Node".
        properties (dict): Additional properties and metadata associated with the node.
    """

    id: Union[str, int]
    type: str = "Node"
    properties: dict = Field(default_factory=dict)


class Relationship(Serializable):
    """Represents a directed relationship between two nodes in a graph.

    Attributes:
        source (Node): The source node of the relationship.
        target (Node): The target node of the relationship.
        type (str): The type of the relationship.
        properties (dict): Additional properties associated with the relationship.
    """

    source: Node
    target: Node
    type: str
    properties: dict = Field(default_factory=dict)


class GraphDocument(Serializable):
    """Represents a graph document consisting of nodes and relationships.

    Attributes:
        nodes (List[Node]): A list of nodes in the graph.
        relationships (List[Relationship]): A list of relationships in the graph.
        source (Document): The document from which the graph information is derived.
    """

    nodes: List[Node]
    relationships: List[Relationship]
    source: Document


class GraphStoreInterface(ABC):
    """Interface for graph store."""

    @property
    @abstractmethod
    def get_schema(self) -> str:
        """Returns the schema of the Graph database"""
        pass

    @property
    @abstractmethod
    def get_structured_schema(self) -> Dict[str, Any]:
        """Returns the schema of the Graph database"""
        pass

    @abstractmethod
    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query the graph."""
        pass

    @abstractmethod
    def refresh_schema(self) -> None:
        """Refreshes the graph schema information."""
        pass

    @abstractmethod
    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        """Take GraphDocument as input as uses it to construct a graph."""
        pass
