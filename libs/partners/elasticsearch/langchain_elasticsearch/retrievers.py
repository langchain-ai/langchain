import logging
from typing import Any, Callable, Dict, List, Optional

from elasticsearch import Elasticsearch
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_elasticsearch._utilities import with_user_agent_header
from langchain_elasticsearch.client import create_elasticsearch_client

logger = logging.getLogger(__name__)


class ElasticsearchRetriever(BaseRetriever):
    """
    Elasticsearch retriever

    Args:
        es_client: Elasticsearch client connection. Alternatively you can use the
            `from_es_params` method with parameters to initialize the client.
        index_name: The name of the index to query.
        body_func: Function to create an Elasticsearch DSL query body from a search
            string. The returned query body must fit what you would normally send in a
            POST request the the _search endpoint. If applicable, it also includes
            parameters the `size` parameter etc.
        content_field: The document field name that contains the page content.
        document_mapper: Function to map Elasticsearch hits to LangChain Documents.
    """

    es_client: Elasticsearch
    index_name: str
    body_func: Callable[[str], Dict]
    content_field: Optional[str] = None
    document_mapper: Optional[Callable[[Dict], Document]] = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if self.content_field is None and self.document_mapper is None:
            raise ValueError("One of content_field or document_mapper must be defined.")
        if self.content_field is not None and self.document_mapper is not None:
            raise ValueError(
                "Both content_field and document_mapper are defined. "
                "Please provide only one."
            )

        self.document_mapper = self.document_mapper or self._field_mapper
        self.es_client = with_user_agent_header(self.es_client, "langchain-py-r")

    @staticmethod
    def from_es_params(
        index_name: str,
        body_func: Callable[[str], Dict],
        content_field: Optional[str] = None,
        document_mapper: Optional[Callable[[Dict], Document]] = None,
        url: Optional[str] = None,
        cloud_id: Optional[str] = None,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> "ElasticsearchRetriever":
        client = None
        try:
            client = create_elasticsearch_client(
                url=url,
                cloud_id=cloud_id,
                api_key=api_key,
                username=username,
                password=password,
                params=params,
            )
        except Exception as err:
            logger.error(f"Error connecting to Elasticsearch: {err}")
            raise err

        return ElasticsearchRetriever(
            es_client=client,
            index_name=index_name,
            body_func=body_func,
            content_field=content_field,
            document_mapper=document_mapper,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if not self.es_client or not self.document_mapper:
            raise ValueError("faulty configuration")  # should not happen

        body = self.body_func(query)
        results = self.es_client.search(index=self.index_name, body=body)
        return [self.document_mapper(hit) for hit in results["hits"]["hits"]]

    def _field_mapper(self, hit: Dict[str, Any]) -> Document:
        content = hit["_source"].pop(self.content_field)
        return Document(page_content=content, metadata=hit)
