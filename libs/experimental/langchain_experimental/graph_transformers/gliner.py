import logging
from typing import Any, Dict, List, Sequence, Union

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

DEFAULT_NODE_TYPE = "Node"


class GlinerGraphTransformer:
    """ """

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
            import gliner_spacy  # type: ignore
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
            import glirel  # type: ignore
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
        seen = set()
        deduplicated_rels = [
            max(
                (
                    item
                    for item in relations
                    if (
                        tuple(item["head_text"]),
                        tuple(item["tail_text"]),
                        item["label"],
                    )
                    == key
                ),
                key=lambda x: x["score"],
            )
            for key in {
                (tuple(item["head_text"]), tuple(item["tail_text"]), item["label"])
                for item in relations
            }
            if not (key in seen or seen.add(key))
        ]
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
