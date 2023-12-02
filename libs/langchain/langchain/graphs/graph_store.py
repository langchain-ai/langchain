from abc import abstractmethod
from typing import Any, Dict, List

from langchain.graphs.graph_document import GraphDocument


class GraphStore:
    """An abstract class wrapper for graph operations."""

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
