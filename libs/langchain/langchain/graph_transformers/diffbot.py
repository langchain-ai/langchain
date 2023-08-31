from typing import Any, Dict, List, Optional, Sequence, Union, Tuple

import requests

from langchain.schema import Document
from langchain.schema.graph_document import (
    BaseGraphDocumentTransformer,
    GraphDocument,
    Node,
    Relationship,
)
from langchain.utils import get_from_env

class NodesList():
    def __init__(self) -> None:
        self.nodes = dict()
    
    def add_node_property(self, node: Tuple[Union[str, int], str], properties:Dict[str,Any]) -> None:
        if not node in self.nodes:
            self.nodes[node] = properties
        else:
            self.nodes[node].update(properties)
    
    def return_node_list(self):
        nodes = [
            Node(id=key[0], type=key[1], properties=self.nodes[key])
            for key in self.nodes
        ]
        return nodes




class DiffbotNLPGraphTransformer(BaseGraphDocumentTransformer):
    def __init__(
        self, diffbot_api_key: Optional[str] = None, threshold_confidence: float = 0.7
    ) -> None:
        self.diffbot_api_key = diffbot_api_key or get_from_env(
            "diffbot_api_key", "DIFFBOT_API_KEY"
        )
        self.threshold_confidence = threshold_confidence

    def nlp_request(self, text) -> Dict[str, Any]:
        """Make an API request to Diffbot NLP endpoint"""

        # Relationship extraction only works for English
        payload = {
            "content": text,
            "lang": "en",
        }

        FIELDS = "facts"
        HOST = "nl.diffbot.com"
        url = f"https://{HOST}/v1/?fields={FIELDS}&token={self.diffbot_api_key}&language=en"
        result = requests.post(url, data=payload)
        return result.json()

    def process_response(
        self, payload: Dict[str, Any], document: Document
    ) -> GraphDocument:
        """Transform the Diffbot NLP response into a list of graph documents"""
        result = []

        # Return empty result if there are no facts
        if not "facts" in payload or not payload["facts"]:
            return GraphDocument(nodes=[], relationships=[], source=document)

        # Nodes are a dictionary because we need to deduplicate
        nodes_list = NodesList()
        # Relationships are a list because we don't deduplicate nor anything else
        relationships = list()
        for record in payload["facts"]:

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
            nodes_list.add_node_property((source_id, source_label), {"name": source_name})

            # Define target node
            target_id = (
                record["value"]["allUris"][0]
                if record["value"]["allUris"]
                else record["value"]["name"]
            )
            target_label = record["value"]["allTypes"][0]["name"].capitalize()
            target_name = record["value"]["name"]
            target_node = Node(id=target_id, type=target_label)
            nodes_list.add_node_property((target_id, target_label), {"name": target_name})
            #Define relationship and its type
            rel_type = record["property"]["name"].replace(" ", "_").upper()
            relationship = Relationship(
                source=source_node, target=target_node, type=rel_type
            )
            relationships.append(relationship)

        return GraphDocument(nodes=nodes_list.return_node_list(), relationships=relationships, source=document)

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
