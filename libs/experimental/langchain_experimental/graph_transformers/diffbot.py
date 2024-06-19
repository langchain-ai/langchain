from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import requests
from langchain.utils import get_from_env
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document


class TypeOption(str, Enum):
    FACTS = "facts"
    ENTITIES = "entities"
    SENTIMENT = "sentiment"


def format_property_key(s: str) -> str:
    """Formats a string to be used as a property key."""

    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)


class NodesList:
    """List of nodes with associated properties.

    Attributes:
        nodes (Dict[Tuple, Any]): Stores nodes as keys and their properties as values.
            Each key is a tuple where the first element is the
            node ID and the second is the node type.
    """

    def __init__(self) -> None:
        self.nodes: Dict[Tuple[Union[str, int], str], Any] = dict()

    def add_node_property(
        self, node: Tuple[Union[str, int], str], properties: Dict[str, Any]
    ) -> None:
        """
        Adds or updates node properties.

        If the node does not exist in the list, it's added along with its properties.
        If the node already exists, its properties are updated with the new values.

        Args:
            node (Tuple): A tuple containing the node ID and node type.
            properties (Dict): A dictionary of properties to add or update for the node.
        """
        if node not in self.nodes:
            self.nodes[node] = properties
        else:
            self.nodes[node].update(properties)

    def return_node_list(self) -> List[Node]:
        """
        Returns the nodes as a list of Node objects.

        Each Node object will have its ID, type, and properties populated.

        Returns:
            List[Node]: A list of Node objects.
        """
        nodes = [
            Node(id=key[0], type=key[1], properties=self.nodes[key])
            for key in self.nodes
        ]
        return nodes


# Properties that should be treated as node properties instead of relationships
FACT_TO_PROPERTY_TYPE = [
    "Date",
    "Number",
    "Job title",
    "Cause of death",
    "Organization type",
    "Academic title",
]


schema_mapping = [
    ("HEADQUARTERS", "ORGANIZATION_LOCATIONS"),
    ("RESIDENCE", "PERSON_LOCATION"),
    ("ALL_PERSON_LOCATIONS", "PERSON_LOCATION"),
    ("CHILD", "HAS_CHILD"),
    ("PARENT", "HAS_PARENT"),
    ("CUSTOMERS", "HAS_CUSTOMER"),
    ("SKILLED_AT", "INTERESTED_IN"),
]


class SimplifiedSchema:
    """Simplified schema mapping.

    Attributes:
        schema (Dict): A dictionary containing the mapping to simplified schema types.
    """

    def __init__(self) -> None:
        """Initializes the schema dictionary based on the predefined list."""
        self.schema = dict()
        for row in schema_mapping:
            self.schema[row[0]] = row[1]

    def get_type(self, type: str) -> str:
        """
        Retrieves the simplified schema type for a given original type.

        Args:
            type (str): The original schema type to find the simplified type for.

        Returns:
            str: The simplified schema type if it exists;
                 otherwise, returns the original type.
        """
        try:
            return self.schema[type]
        except KeyError:
            return type


class DiffbotGraphTransformer:
    """Transform documents into graph documents using Diffbot NLP API.

    A graph document transformation system takes a sequence of Documents and returns a
    sequence of Graph Documents.

    Example:
        .. code-block:: python
          from langchain_experimental.graph_transformers import DiffbotGraphTransformer
          from langchain_core.documents import Document

          diffbot_api_key = "DIFFBOT_API_KEY"
          diffbot_nlp = DiffbotGraphTransformer(diffbot_api_key=diffbot_api_key)

          document = Document(page_content="Mike Tunge is the CEO of Diffbot.")
          graph_documents = diffbot_nlp.convert_to_graph_documents([document])

    """

    def __init__(
        self,
        diffbot_api_key: Optional[str] = None,
        fact_confidence_threshold: float = 0.7,
        include_qualifiers: bool = True,
        include_evidence: bool = True,
        simplified_schema: bool = True,
        extract_types: List[TypeOption] = [TypeOption.FACTS],
        *,
        include_confidence: bool = False,
    ) -> None:
        """
        Initialize the graph transformer with various options.

        Args:
            diffbot_api_key (str):
               The API key for Diffbot's NLP services.

            fact_confidence_threshold (float):
                Minimum confidence level for facts to be included.
            include_qualifiers (bool):
                Whether to include qualifiers in the relationships.
            include_evidence (bool):
                Whether to include evidence for the relationships.
            simplified_schema (bool):
                Whether to use a simplified schema for relationships.
            extract_types (List[TypeOption]):
                A list of data types to extract. Facts, entities, and
                sentiment are supported. By default, the option is
                set to facts. A fact represents a combination of
                source and target nodes with a relationship type.
            include_confidence (bool):
                Whether to include confidence scores on nodes and rels
        """
        self.diffbot_api_key = diffbot_api_key or get_from_env(
            "diffbot_api_key", "DIFFBOT_API_KEY"
        )
        self.fact_threshold_confidence = fact_confidence_threshold
        self.include_qualifiers = include_qualifiers
        self.include_evidence = include_evidence
        self.include_confidence = include_confidence
        self.simplified_schema = None
        if simplified_schema:
            self.simplified_schema = SimplifiedSchema()
        if not extract_types:
            raise ValueError(
                "`extract_types` cannot be an empty array. "
                "Allowed values are 'facts', 'entities', or both."
            )

        self.extract_types = extract_types

    def nlp_request(self, text: str) -> Dict[str, Any]:
        """
        Make an API request to the Diffbot NLP endpoint.

        Args:
            text (str): The text to be processed.

        Returns:
            Dict[str, Any]: The JSON response from the API.
        """

        # Relationship extraction only works for English
        payload = {
            "content": text,
            "lang": "en",
        }

        FIELDS = ",".join(self.extract_types)
        HOST = "nl.diffbot.com"
        url = (
            f"https://{HOST}/v1/?fields={FIELDS}&"
            f"token={self.diffbot_api_key}&language=en"
        )
        result = requests.post(url, data=payload)
        return result.json()

    def process_response(
        self, payload: Dict[str, Any], document: Document
    ) -> GraphDocument:
        """
        Transform the Diffbot NLP response into a GraphDocument.

        Args:
            payload (Dict[str, Any]): The JSON response from Diffbot's NLP API.
            document (Document): The original document.

        Returns:
            GraphDocument: The transformed document as a graph.
        """

        # Return empty result if there are no facts
        if ("facts" not in payload or not payload["facts"]) and (
            "entities" not in payload or not payload["entities"]
        ):
            return GraphDocument(nodes=[], relationships=[], source=document)

        # Nodes are a custom class because we need to deduplicate
        nodes_list = NodesList()
        if "entities" in payload and payload["entities"]:
            for record in payload["entities"]:
                # Ignore if it doesn't have a type
                if not record["allTypes"]:
                    continue

                # Define source node
                source_id = (
                    record["allUris"][0] if record["allUris"] else record["name"]
                )
                source_label = record["allTypes"][0]["name"].capitalize()
                source_name = record["name"]
                nodes_list.add_node_property(
                    (source_id, source_label), {"name": source_name}
                )
                if record.get("sentiment") is not None:
                    nodes_list.add_node_property(
                        (source_id, source_label),
                        {"sentiment": record.get("sentiment")},
                    )
                if self.include_confidence:
                    nodes_list.add_node_property(
                        (source_id, source_label),
                        {"confidence": record.get("confidence")},
                    )

        relationships = list()
        # Relationships are a list because we don't deduplicate nor anything else
        if "facts" in payload and payload["facts"]:
            for record in payload["facts"]:
                # Skip if the fact is below the threshold confidence
                if record["confidence"] < self.fact_threshold_confidence:
                    continue

                # TODO: It should probably be treated as a node property
                if not record["value"]["allTypes"]:
                    continue

                # Define source node
                source_id = (
                    record["entity"]["allUris"][0]
                    if record["entity"]["allUris"]
                    else record["entity"]["name"]
                )
                source_label = record["entity"]["allTypes"][0]["name"].capitalize()
                source_name = record["entity"]["name"]
                source_node = Node(id=source_id, type=source_label)
                nodes_list.add_node_property(
                    (source_id, source_label), {"name": source_name}
                )

                # Define target node
                target_id = (
                    record["value"]["allUris"][0]
                    if record["value"]["allUris"]
                    else record["value"]["name"]
                )
                target_label = record["value"]["allTypes"][0]["name"].capitalize()
                target_name = record["value"]["name"]
                # Some facts are better suited as node properties
                if target_label in FACT_TO_PROPERTY_TYPE:
                    nodes_list.add_node_property(
                        (source_id, source_label),
                        {format_property_key(record["property"]["name"]): target_name},
                    )
                else:  # Define relationship
                    # Define target node object
                    target_node = Node(id=target_id, type=target_label)
                    nodes_list.add_node_property(
                        (target_id, target_label), {"name": target_name}
                    )
                    # Define relationship type
                    rel_type = record["property"]["name"].replace(" ", "_").upper()
                    if self.simplified_schema:
                        rel_type = self.simplified_schema.get_type(rel_type)

                    # Relationship qualifiers/properties
                    rel_properties = dict()
                    relationship_evidence = [
                        el["passage"] for el in record["evidence"]
                    ][0]
                    if self.include_evidence:
                        rel_properties.update({"evidence": relationship_evidence})
                    if self.include_confidence:
                        rel_properties.update({"confidence": record["confidence"]})
                    if self.include_qualifiers and record.get("qualifiers"):
                        for property in record["qualifiers"]:
                            prop_key = format_property_key(property["property"]["name"])
                            rel_properties[prop_key] = property["value"]["name"]

                    relationship = Relationship(
                        source=source_node,
                        target=target_node,
                        type=rel_type,
                        properties=rel_properties,
                    )
                    relationships.append(relationship)

        return GraphDocument(
            nodes=nodes_list.return_node_list(),
            relationships=relationships,
            source=document,
        )

    def convert_to_graph_documents(
        self, documents: Sequence[Document]
    ) -> List[GraphDocument]:
        """Convert a sequence of documents into graph documents.

        Args:
            documents (Sequence[Document]): The original documents.
            **kwargs: Additional keyword arguments.

        Returns:
            Sequence[GraphDocument]: The transformed documents as graphs.
        """
        results = []
        for document in documents:
            raw_results = self.nlp_request(document.page_content)
            graph_document = self.process_response(raw_results, document)
            results.append(graph_document)
        return results
