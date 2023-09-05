from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import requests

from langchain.schema import Document
from langchain.schema.graph_document import (
    BaseGraphDocumentTransformer,
    GraphDocument,
    Node,
    Relationship,
)
from langchain.utils import get_from_env


def format_property_key(s) -> str:
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)


class NodesList:
    def __init__(self) -> None:
        self.nodes = dict()

    def add_node_property(
        self, node: Tuple[Union[str, int], str], properties: Dict[str, Any]
    ) -> None:
        if node not in self.nodes:
            self.nodes[node] = properties
        else:
            self.nodes[node].update(properties)

    def return_node_list(self) -> List[Node]:
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
    def __init__(self) -> None:
        self.schema = dict()
        for row in schema_mapping:
            self.schema[row[0]] = row[1]

    def get_type(self, type):
        try:
            return self.schema[type]
        except KeyError:
            return type


class DiffbotGraphTransformer(BaseGraphDocumentTransformer):
    def __init__(
        self,
        diffbot_api_key: Optional[str] = None,
        fact_confidence_threshold: float = 0.7,
        include_qualifiers: bool = True,
        include_evidence: bool = True,
        simplified_schema: bool = True,
    ) -> None:
        self.diffbot_api_key = diffbot_api_key or get_from_env(
            "diffbot_api_key", "DIFFBOT_API_KEY"
        )
        self.fact_threshold_confidence = fact_confidence_threshold
        self.include_qualifiers = include_qualifiers
        self.include_evidence = include_evidence
        self.simplified_schema = None
        if simplified_schema:
            self.simplified_schema = SimplifiedSchema()

    def nlp_request(self, text) -> Dict[str, Any]:
        """Make an API request to Diffbot NLP endpoint"""

        # Relationship extraction only works for English
        payload = {
            "content": text,
            "lang": "en",
        }

        FIELDS = "facts"
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
        """Transform the Diffbot NLP response into a list of graph documents"""

        # Return empty result if there are no facts
        if "facts" not in payload or not payload["facts"]:
            return GraphDocument(nodes=[], relationships=[], source=document)

        # Nodes are a dictionary because we need to deduplicate
        nodes_list = NodesList()
        # Relationships are a list because we don't deduplicate nor anything else
        relationships = list()
        for record in payload["facts"]:
            # Skip if the fact is below the threshold confidence
            if record["confidence"] < self.fact_threshold_confidence:
                continue

            # TODO: It should probably be treated as a property
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
                # Define target node
                target_node = Node(id=target_id, type=target_label)
                nodes_list.add_node_property(
                    (target_id, target_label), {"name": target_name}
                )
                # Define relationship type
                rel_type = record["property"]["name"].replace(" ", "_").upper()
                if self.simplified_schema:
                    rel_type = self.simplified_schema.get_type(rel_type)

                # Relationship properties
                rel_properties = dict()
                relationship_evidence = [el["passage"] for el in record["evidence"]][0]
                if self.include_evidence:
                    rel_properties.update({"evidence": relationship_evidence})
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

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[GraphDocument]:

        results = []

        for document in documents:
            raw_results = self.nlp_request(document.page_content)
            graph_document = self.process_response(raw_results, document)
            results.append(graph_document)
        return results

    def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[GraphDocument]:
        raise NotImplementedError()
