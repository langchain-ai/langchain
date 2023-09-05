from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Sequence, Union

from langchain.load.serializable import Serializable
from langchain.pydantic_v1 import Field
from langchain.schema import Document


class Node(Serializable):
    """
    Represents a node in a graph with associated properties.

    Attributes:
        id (Union[str, int]): A unique identifier for the node.
        type (str): The type or label of the node, default is "Node".
        properties (dict): Additional properties and metadata associated with the node.
    """

    id: Union[str, int]
    type: str = "Node"
    properties: dict = Field(default_factory=dict)


class Relationship(Serializable):
    """
    Represents a directed relationship between two nodes in a graph.

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
    """
    Represents a graph document consisting of nodes and relationships.

    Attributes:
        nodes (List[Node]): A list of nodes in the graph.
        relationships (List[Relationship]): A list of relationships in the graph.
        source (Document): The document from which the graph information is derived.
    """

    nodes: List[Node]
    relationships: List[Relationship]
    source: Document


class BaseGraphDocumentTransformer(ABC):
    """Abstract base class for graph document transformation systems.

    A graph document transformation system takes a sequence of Documents and returns a
    sequence of Graph Documents.

    Example:
        .. code-block:: python

            class DiffbotGraphTransformer(BaseGraphDocumentTransformer):

                def transform_documents(
                    self, documents: Sequence[Document], **kwargs: Any
                ) -> Sequence[GraphDocument]:
                    results = []

                    for document in documents:
                        raw_results = self.nlp_request(document.page_content)
                        graph_document = self.process_response(raw_results, document)
                        results.append(graph_document)
                    return results

                async def atransform_documents(
                    self, documents: Sequence[Document], **kwargs: Any
                ) -> Sequence[Document]:
                    raise NotImplementedError

    """  # noqa: E501

    @abstractmethod
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[GraphDocument]:
        """Transform a list of documents to graph documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A list of generated Graph Documents.
        """

    @abstractmethod
    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[GraphDocument]:
        """Asynchronously transform a list of documents to graph documents.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A list of generated Graph Documents.
        """
