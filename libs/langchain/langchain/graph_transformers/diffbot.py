from typing import Any, List, Optional, Sequence, Union, Dict

from langchain.schema import Document
from langchain.schema.graph_document import GraphDocument, BaseGraphDocumentTransformer, Node, Relationship
from langchain.utils import get_from_env

import requests




class DiffbotNLPGraphTransformer(BaseGraphDocumentTransformer):

    def __init__(
    self,
    diffbot_api_key: Optional[str] = None,
    threshold_confidence: float = 0.7
) -> None:
        self.diffbot_api_key = diffbot_api_key or get_from_env(
            "diffbot_api_key", "DIFFBOT_API_KEY"
        )
        self.threshold_confidence = threshold_confidence
    
    def nlp_request(self, text) -> Dict[str, Any]:
        """Make an API request to Diffbot NLP endpoint"""

        # Relationship extraction only works for English
        payload = {"content": text, "lang": "en", }

        FIELDS = "entities,facts"
        HOST = "nl.diffbot.com"
        url = f"https://{HOST}/v1/?fields={FIELDS}&token={self.diffbot_api_key}&language=en"
        result = requests.post(url, data=payload)
        return result.json()
    
    def process_response(self, payload:Dict[str, Any]) -> GraphDocument:
        """Transform the Diffbot NLP response into a list of graph documents"""
        result = []

        # Return empty result if there are no facts
        if not 'facts' in payload or not payload['facts']:
            return GraphDocument(
                nodes=[],
                relationships=[],
                source=Document(page_content="foo")
            )

        # Nodes are a dictionary because we can append other metadata from facts data later
        nodes_dict = dict()
        for record in payload['entities']:
            
            # If the entity doesn't have a type, it's a weird entity
            if not record['allTypes']:
                continue

            name = record['name']
            label = record['allTypes'][0]['name'].capitalize()
            id = record['allUris'][0] if record['allUris'] else record['name']
            nodes_dict[(id, label)] = {"name":name}

        # Relationships are a list because we don't deduplicate nor anything else
        relationships = list()
        for record in payload['facts']:
            
            # TODO: It should probably be treated as a property
            if not record['value']['allTypes']:
                continue
            
            source_id = record['entity']['allUris'][0] if record['entity']['allUris'] else record['entity']['name']
            source_label = record['entity']['allTypes'][0]['name'].capitalize()

            source_node = Node(id=source_id, type=source_label)
            
            target_id = record['value']['allUris'][0] if record['value']['allUris'] else record['value']['name']
            target_label = record['value']['allTypes'][0]['name'].capitalize()

            target_node = Node(id=target_id, type=target_label)

            rel_type = record['property']['name']

            relationship = Relationship(source=source_node, target=target_node, type=rel_type)
            relationships.append(relationship)

        # We construct nodes property only here as we want the option to add additional metadata
        nodes = [Node(id=key[0], type=key[1], properties=nodes_dict[key]) for key in nodes_dict]

        return GraphDocument(
            nodes=nodes,
            relationships=relationships,
            source=Document(page_content="Foo")
        )




        

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[GraphDocument]:
        
        results = []

        for document in documents:
            raw_results = self.nlp_request(document.page_content)
            graph_document = self.process_response(raw_results)
            results.append(graph_document)
        return results

    def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[GraphDocument]:
        raise NotImplementedError()

        
