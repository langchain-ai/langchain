from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from langchain_community.graphs.graph_document import GraphDocument


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
