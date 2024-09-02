from typing import Any, Dict, List, Sequence, Union

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

DEFAULT_NODE_TYPE = "Node"


class GlinerGraphTransformer:
    """
    A transformer class for converting documents into graph structures
    using the GLiNER and GLiREL models.

    This class leverages GLiNER for named entity recognition and GLiREL for
    relationship extraction from text documents, converting them into a graph format.
    The extracted entities and relationships are filtered based on specified
    confidence thresholds and allowed types.

    For more details on GLiNER and GLiREL, visit their respective repositories:
      GLiNER: https://github.com/urchade/GLiNER
      GLiREL: https://github.com/jackboyla/GLiREL/tree/main

    Args:
        allowed_nodes (List[str]): A list of allowed node types for entity extraction.
        allowed_relationships (Union[List[str], Dict[str, Any]]): A list of allowed
          relationship types or a dictionary with additional configuration for
          relationship extraction.
        gliner_model (str): The name of the pretrained GLiNER model to use.
          Default is "urchade/gliner_mediumv2.1".
        glirel_model (str): The name of the pretrained GLiREL model to use.
          Default is "jackboyla/glirel_beta".
        entity_confidence_threshold (float): The confidence threshold for
          filtering extracted entities. Default is 0.1.
        relationship_confidence_threshold (float): The confidence threshold for
          filtering extracted relationships. Default is 0.1.
        device (str): The device to use for model inference ('cpu' or 'cuda').
          Default is "cpu".
        ignore_self_loops (bool): Whether to ignore relationships where the
          source and target nodes are the same. Default is True.
    """

    def __init__(
        self,
        allowed_nodes: List[str],
        allowed_relationships: Union[List[str], Dict[str, Any]],
        gliner_model: str = "urchade/gliner_mediumv2.1",
        glirel_model: str = "jackboyla/glirel_beta",
        entity_confidence_threshold: float = 0.1,
        relationship_confidence_threshold: float = 0.1,
        device: str = "cpu",
        ignore_self_loops: bool = True,
    ) -> None:
        try:
            import gliner_spacy  # type: ignore # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import relik python package. "
                "Please install it with `pip install gliner-spacy`."
            )
        try:
            import spacy  # type: ignore
        except ImportError:
            raise ImportError(
                "Could not import relik python package. "
                "Please install it with `pip install spacy`."
            )
        try:
            import glirel  # type: ignore # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import relik python package. "
                "Please install it with `pip install gliner`."
            )

        gliner_config = {
            "gliner_model": gliner_model,
            "chunk_size": 250,
            "labels": allowed_nodes,
            "style": "ent",
            "threshold": entity_confidence_threshold,
            "map_location": device,
        }
        glirel_config = {"model": glirel_model, "device": device}
        self.nlp = spacy.blank("en")
        # Add the GliNER component to the pipeline
        self.nlp.add_pipe("gliner_spacy", config=gliner_config)
        # Add the GLiREL component to the pipeline
        self.nlp.add_pipe("glirel", after="gliner_spacy", config=glirel_config)
        self.allowed_relationships = (
            {"glirel_labels": allowed_relationships}
            if isinstance(allowed_relationships, list)
            else allowed_relationships
        )
        self.relationship_confidence_threshold = relationship_confidence_threshold
        self.ignore_self_loops = ignore_self_loops

    def process_document(self, document: Document) -> GraphDocument:
        # Extraction as SpaCy pipeline
        docs = list(
            self.nlp.pipe(
                [(document.page_content, self.allowed_relationships)], as_tuples=True
            )
        )
        # Convert nodes
        nodes = []
        for node in docs[0][0].ents:
            nodes.append(
                Node(
                    id=node.text,
                    type=node.label_,
                )
            )
        # Convert relationships
        relationships = []
        relations = docs[0][0]._.relations
        # Deduplicate based on label, head text, and tail text
        # Use a list comprehension with max() function
        deduplicated_rels = []
        seen = set()

        for item in relations:
            key = (tuple(item["head_text"]), tuple(item["tail_text"]), item["label"])

            if key not in seen:
                seen.add(key)

                # Find all items matching the current key
                matching_items = [
                    rel
                    for rel in relations
                    if (tuple(rel["head_text"]), tuple(rel["tail_text"]), rel["label"])
                    == key
                ]

                # Find the item with the maximum score
                max_item = max(matching_items, key=lambda x: x["score"])
                deduplicated_rels.append(max_item)
        for rel in deduplicated_rels:
            # Relationship confidence threshold
            if rel["score"] < self.relationship_confidence_threshold:
                continue
            source_id = docs[0][0][rel["head_pos"][0] : rel["head_pos"][1]].text
            target_id = docs[0][0][rel["tail_pos"][0] : rel["tail_pos"][1]].text
            # Ignore self loops
            if self.ignore_self_loops and source_id == target_id:
                continue
            source_node = [n for n in nodes if n.id == source_id][0]
            target_node = [n for n in nodes if n.id == target_id][0]
            relationships.append(
                Relationship(
                    source=source_node,
                    target=target_node,
                    type=rel["label"].replace(" ", "_").upper(),
                )
            )

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
