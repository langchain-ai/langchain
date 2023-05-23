"""Wrapper around Azure Cognitive Search."""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    PrioritizedFields,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
)
from azure.search.documents.models import Vector
from pydantic import BaseModel, root_validator

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever
from langchain.vectorstores.base import VectorStore

logger = logging.getLogger()

AZURESEARCH_DIMENSIONS = int(
    os.environ.get("AZURESEARCH_DIMENSIONS", 1536)
)  # Default to OpenAI's ada-002 embedding model vector size

# Allow overriding field names for Azure Search
FIELDS_ID = os.environ.get("AZURESEARCH_FIELDS_ID", "id")
FIELDS_TITLE = os.environ.get("AZURESEARCH_FIELDS_TITLE", "title")
FIELDS_CONTENT = os.environ.get("AZURESEARCH_FIELDS_CONTENT", "content")
FIELDS_CONTENT_VECTOR = os.environ.get(
    "AZURESEARCH_FIELDS_CONTENT_VECTOR", "content_vector"
)
FIELDS_TAG = os.environ.get("AZURESEARCH_FIELDS_TAG", "tag")
FIELDS_METADATA = os.environ.get("AZURESEARCH_FIELDS_TAG", "metadata")

MAX_UPLOAD_BATCH_SIZE = 1000


def get_search_client(
    endpoint: str,
    key: str,
    index_name: str,
    semantic_configuration_name: Optional[str] = None,
) -> SearchClient:
    if key is None:
        credential = DefaultAzureCredential()
    else:
        credential = AzureKeyCredential(key)
    index_client: SearchIndexClient = SearchIndexClient(
        endpoint=endpoint, credential=credential
    )
    try:
        index_client.get_index(name=index_name)
    except ResourceNotFoundError:
        # Fields configuration
        fields = [
            SimpleField(
                name=FIELDS_ID,
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchableField(
                name=FIELDS_TITLE,
                type=SearchFieldDataType.String,
                searchable=True,
                retrievable=True,
            ),
            SearchableField(
                name=FIELDS_CONTENT,
                type=SearchFieldDataType.String,
                searchable=True,
                retrievable=True,
            ),
            SearchField(
                name=FIELDS_CONTENT_VECTOR,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                dimensions=AZURESEARCH_DIMENSIONS,
                vector_search_configuration="default",
            ),
            SearchableField(
                name=FIELDS_TAG,
                type=SearchFieldDataType.String,
                filterable=True,
                searchable=True,
                retrievable=True,
            ),
            SearchableField(
                name=FIELDS_METADATA,
                type=SearchFieldDataType.String,
                searchable=True,
                retrievable=True,
            ),
        ]
        # Vector search configuration
        vector_search = VectorSearch(
            algorithm_configurations=[
                VectorSearchAlgorithmConfiguration(
                    name="default",
                    kind="hnsw",
                    hnsw_parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine",
                    },
                )
            ]
        )
        # Create the semantic settings with the configuration
        semantic_settings = (
            None
            if semantic_configuration_name is None
            else SemanticSettings(
                configurations=[
                    SemanticConfiguration(
                        name=semantic_configuration_name,
                        prioritized_fields=PrioritizedFields(
                            title_field=SemanticField(field_name=FIELDS_TITLE),
                            prioritized_keywords_fields=[
                                SemanticField(field_name=FIELDS_TAG)
                            ],
                            prioritized_content_fields=[
                                SemanticField(field_name=FIELDS_CONTENT)
                            ],
                        ),
                    )
                ]
            )
        )
        # Create the search index with the semantic settings and vector search
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_settings=semantic_settings,
        )
        index_client.create_index(index)
    # Create the search client
    return SearchClient(
        endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(key)
    )


class AzureSearch(VectorStore):
    def __init__(
        self,
        azure_cognitive_search_name: str,
        azure_cognitive_search_key: str,
        index_name: str,
        embedding_function: Callable,
        semantic_configuration_name: Optional[str] = None,
        semantic_query_language: str = "en-us",
        **kwargs: Any,
    ):
        """Initialize with necessary components."""
        # Initialize base class
        self.embedding_function = embedding_function
        self.client = get_search_client(
            azure_cognitive_search_name,
            azure_cognitive_search_key,
            index_name,
            semantic_configuration_name,
        )
        self.semantic_configuration_name = semantic_configuration_name
        self.semantic_query_language = semantic_query_language

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts data to an existing index."""
        keys = kwargs.get("keys")
        ids = []
        # Write data to index
        data = []
        for i, text in enumerate(texts):
            # Use provided key otherwise use default key
            key = keys[i] if keys else str(uuid.uuid4())
            metadata = metadatas[i] if metadatas else {}
            # Add data to index
            data.append(
                {
                    "@search.action": "upload",
                    FIELDS_ID: key,
                    FIELDS_TITLE: metadata.get(FIELDS_TITLE, ""),
                    FIELDS_TAG: metadata.get(FIELDS_TAG, ""),
                    FIELDS_CONTENT: text,
                    FIELDS_CONTENT_VECTOR: np.array(
                        self.embedding_function(text), dtype=np.float32
                    ).tolist(),
                    FIELDS_METADATA: json.dumps(metadata),
                }
            )
            ids.append(key)
            # Upload data in batches
            if len(data) == MAX_UPLOAD_BATCH_SIZE:
                response = self.client.upload_documents(documents=data)
                # Check if all documents were successfully uploaded
                if not all([r.succeeded for r in response]):
                    raise Exception(response)
                # Reset data
                data = []
        # Upload data to index
        response = self.client.upload_documents(documents=data)
        # Check if all documents were successfully uploaded
        if all([r.succeeded for r in response]):
            return ids
        else:
            raise Exception(response)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = self.similarity_search_with_score(
            query, k=k, filters=kwargs.get("filters", None)
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filters: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        results = self.client.search(
            search_text="",
            vector=Vector(
                value=np.array(
                    self.embedding_function(query), dtype=np.float32
                ).tolist(),
                k=k,
                fields=FIELDS_CONTENT_VECTOR,
            ),
            select=[f"{FIELDS_TITLE},{FIELDS_CONTENT},{FIELDS_METADATA}"],
            filter=filters,
        )
        # Convert results to Document objects
        docs = [
            (
                Document(
                    page_content=result[FIELDS_CONTENT],
                    metadata=json.loads(result[FIELDS_METADATA]),
                ),
                1 - float(result["@search.score"]),
            )
            for result in results
        ]
        return docs

    def hybrid_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = self.hybrid_search_with_score(
            query, k=k, filters=kwargs.get("filters", None)
        )
        return [doc for doc, _ in docs_and_scores]

    def hybrid_search_with_score(
        self, query: str, k: int = 4, filters: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query with an hybrid query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        results = self.client.search(
            search_text=query,
            vector=Vector(
                value=np.array(
                    self.embedding_function(query), dtype=np.float32
                ).tolist(),
                k=k,
                fields=FIELDS_CONTENT_VECTOR,
            ),
            select=[f"{FIELDS_TITLE},{FIELDS_CONTENT},{FIELDS_METADATA}"],
            filter=filters,
            top=k,
        )
        # Convert results to Document objects
        docs = [
            (
                Document(
                    page_content=result[FIELDS_CONTENT],
                    metadata=json.loads(result[FIELDS_METADATA]),
                ),
                1 - float(result["@search.score"]),
            )
            for result in results
        ]
        return docs

    def semantic_hybrid_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = self.semantic_hybrid_search_with_score(
            query, k=k, filters=kwargs.get("filters", None)
        )
        return [doc for doc, _ in docs_and_scores]

    def semantic_hybrid_search_with_score(
        self, query: str, k: int = 4, filters: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query with an hybrid query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        results = self.client.search(
            search_text=query,
            vector=Vector(
                value=np.array(
                    self.embedding_function(query), dtype=np.float32
                ).tolist(),
                k=k,
                fields=FIELDS_CONTENT_VECTOR,
            ),
            select=[f"{FIELDS_TITLE},{FIELDS_CONTENT},{FIELDS_METADATA}"],
            filter=filters,
            query_type="semantic",
            query_language=self.semantic_query_language,
            semantic_configuration_name=self.semantic_configuration_name,
            query_caption="extractive",
            query_answer="extractive",
            top=k,
        )
        # Get Semantic Answers
        semantic_answers = results.get_answers()
        semantic_answers_dict = {}
        for semantic_answer in semantic_answers:
            semantic_answers_dict[semantic_answer.key] = {
                "text": semantic_answer.text,
                "highlights": semantic_answer.highlights,
            }
        # Convert results to Document objects
        docs = [
            (
                Document(
                    page_content=result["content"],
                    metadata={
                        **json.loads(result["metadata"]),
                        **{
                            "captions": {
                                "text": result.get("@search.captions", [{}])[0].text,
                                "highlights": result.get("@search.captions", [{}])[
                                    0
                                ].highlights,
                            }
                            if result.get("@search.captions")
                            else {},
                            "answers": semantic_answers_dict.get(
                                json.loads(result["metadata"]).get("key"), ""
                            ),
                        },
                    },
                ),
                1 - float(result["@search.score"]),
            )
            for result in results
        ]
        return docs

    @classmethod
    def from_texts(
        cls: Type[AzureSearch],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        azure_cognitive_search_name: str = os.getenv("AZURE_SEARCH_ENDPOINT", ""),
        azure_cognitive_search_key: str = os.getenv("AZURE_SEARCH_ADMIN_KEY", ""),
        index_name: Optional[str] = None,
        **kwargs: Any,
    ) -> AzureSearch:
        # Name of the search index if not given
        if not index_name:
            index_name = uuid.uuid4().hex
        # Creating a new Azure Search instance
        azure_search = cls(
            azure_cognitive_search_name,
            azure_cognitive_search_key,
            index_name,
            embedding.embed_query,
        )
        azure_search.add_texts(texts, metadatas, **kwargs)
        return azure_search


class AzureSearchVectorStoreRetriever(BaseRetriever, BaseModel):
    vectorstore: AzureSearch
    search_type: str = "similarity"
    k: int = 4
    score_threshold: float = 0.4

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator()
    def validate_search_type(cls, values: Dict) -> Dict:
        """Validate search type."""
        if "search_type" in values:
            search_type = values["search_type"]
            if search_type not in ("similarity", "hybrid", "semantic_hybrid"):
                raise ValueError(f"search_type of {search_type} not allowed.")
        return values

    def get_relevant_documents(self, query: str) -> List[Document]:
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, k=self.k)
        elif self.search_type == "hybrid":
            docs = self.vectorstore.hybrid_search(query, k=self.k)
        elif self.search_type == "semantic_hybrid":
            docs = self.vectorstore.semantic_hybrid_search(query, k=self.k)
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError(
            "AzureSearchVectorStoreRetriever does not support async"
        )
