import logging
from typing import Any, Dict, List, Sequence

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

DEFAULT_NODE_TYPE = "Node"


class RelikGraphTransformer:
    """
    A transformer class for converting documents into graph structures
    using the Relik library and models.

    This class leverages relik models for extracting relationships
    and nodes from text documents and converting them into a graph format.
    The relationships are filtered based on a specified confidence threshold.

    For more details on the Relik library, visit their GitHub repository:
      https://github.com/SapienzaNLP/relik

    Args:
        model (str): The name of the pretrained Relik model to use.
          Default is "relik-ie/relik-relation-extraction-small-wikipedia".
        relationship_confidence_threshold (float): The confidence threshold for
          filtering relationships. Default is 0.1.
        model_config (Dict[str, any]): Additional configuration options for the
          Relik model. Default is an empty dictionary.
        ignore_self_loops (bool): Whether to ignore relationships where the
          source and target nodes are the same. Default is True.
    """

    def __init__(
        self,
        model: str = "relik-ie/relik-relation-extraction-small",
        relationship_confidence_threshold: float = 0.1,
        model_config: Dict[str, Any] = {},
        ignore_self_loops: bool = True,
    ) -> None:
        try:
            import relik  # type: ignore

            # Remove default INFO logging
            logging.getLogger("relik").setLevel(logging.WARNING)
        except ImportError:
            raise ImportError(
                "Could not import relik python package. "
                "Please install it with `pip install relik`."
            )
        self.relik_model = relik.Relik.from_pretrained(model, **model_config)
        self.relationship_confidence_threshold = relationship_confidence_threshold
        self.ignore_self_loops = ignore_self_loops

    def process_document(self, document: Document) -> GraphDocument:
        relik_out = self.relik_model(document.page_content)
        nodes = []
        # Extract nodes
        for node in relik_out.spans:
            nodes.append(
                Node(
                    id=node.text,
                    type=DEFAULT_NODE_TYPE
                    if node.label.strip() == "--NME--"
                    else node.label.strip(),
                )
            )

        relationships = []
        # Extract relationships
        for triple in relik_out.triplets:
            # Ignore relationship if below confidence threshold
            if triple.confidence < self.relationship_confidence_threshold:
                continue
            # Ignore self loops
            if self.ignore_self_loops and triple.subject.text == triple.object.text:
                continue
            source_node = Node(
                id=triple.subject.text,
                type=DEFAULT_NODE_TYPE
                if triple.subject.label.strip() == "--NME--"
                else triple.subject.label.strip(),
            )
            target_node = Node(
                id=triple.object.text,
                type=DEFAULT_NODE_TYPE
                if triple.object.label.strip() == "--NME--"
                else triple.object.label.strip(),
            )

            relationship = Relationship(
                source=source_node,
                target=target_node,
                type=triple.label.replace(" ", "_").upper(),
            )
            relationships.append(relationship)

        return GraphDocument(nodes=nodes, relationships=relationships, source=document)

    def convert_to_graph_documents(
        self, documents: Sequence[Document]
    ) -> List[GraphDocument]:
        """Convert a sequence of documents into graph documents.

        Args:
            documents (Sequence[Document]): The original documents.
            kwargs: Additional keyword arguments.

        Returns:
            Sequence[GraphDocument]: The transformed documents as graphs.
        """
        results = []
        for document in documents:
            graph_document = self.process_document(document)
            results.append(graph_document)
        return results
