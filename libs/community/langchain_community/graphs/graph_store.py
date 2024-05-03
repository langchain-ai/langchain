from abc import abstractmethod
from typing import Any, Dict, List

from langchain_community.graphs.graph_document import GraphDocument


class GraphStore:
    """Abstract class for graph operations."""

    @property
    @abstractmethod
    def get_schema(self) -> str:
        """Return the schema of the Graph database"""
        pass

    @property
    @abstractmethod
    def get_structured_schema(self) -> Dict[str, Any]:
        """Return the schema of the Graph database"""
        pass

    @abstractmethod
    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query the graph."""
        pass

    @abstractmethod
    def refresh_schema(self) -> None:
        """Refresh the graph schema information."""
        pass

    @abstractmethod
    def add_graph_documents(
        self, graph_documents: List[GraphDocument], include_source: bool = False
    ) -> None:
        """Take GraphDocument as input as uses it to construct a graph."""
        pass
