"""Wrapper around Azure Cognitive Search."""
from __future__ import annotations

import json
import logging
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from pydantic import BaseModel, root_validator

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_env
from langchain.vectorstores.base import VectorStore, VectorStoreRetriever

if TYPE_CHECKING:
    from azure.core.credentials import AzureKeyCredential
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient, SearchItemPaged
    from azure.search.documents.indexes import SearchIndexClient


logger = logging.getLogger()

AZURESEARCH_DIMENSIONS = 1536  # Default to OpenAI's ada-002 embedding model vector size
MAX_UPLOAD_BATCH_SIZE = 1000


def _check_index(
    service_name: str,
    credential: Union[AzureKeyCredential, DefaultAzureCredential],
    index_name: str,
    id_field: str,
    content_field: str,
    content_vector_field: str,
    metadata_field: str,
    title_field: str,
    tag_field: str,
    embeddings_dim: int,
    semantic_configuration_name: Optional[str],
) -> None:
    try:
        from azure.core.exceptions import ResourceNotFoundError
        from azure.search.documents.indexes import SearchIndexClient
    except ImportError:
        raise ImportError

    index_client = SearchIndexClient(
        endpoint="https://" + service_name, credential=credential
    )
    try:
        index_client.get_index(name=index_name)
    except ResourceNotFoundError:
        logger.info(f"Index {index_name} not found, creating new index.")
        _create_index(
            index_name,
            index_client,
            id_field,
            content_field,
            content_vector_field,
            metadata_field,
            title_field,
            tag_field,
            embeddings_dim,
            semantic_configuration_name,
        )
    except:
        pass


def _create_index(
    index_name: str,
    index_client: SearchIndexClient,
    id_field: str,
    content_field: str,
    content_vector_field: str,
    metadata_field: str,
    title_field: str,
    tag_field: str,
    embeddings_dim: int,
    semantic_configuration_name: Optional[str],
) -> None:
    try:
        from azure.search.documents.indexes._generated.models import HnswParameters
        from azure.search.documents.indexes.models import (
            PrioritizedFields,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SemanticConfiguration,
            SemanticField,
            SimpleField,
            VectorSearch,
            VectorSearchAlgorithmConfiguration,
        )
    except ImportError:
        raise ImportError

    # Fields configuration
    fields = [
        SimpleField(
            name=id_field,
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name=title_field,
            type=SearchFieldDataType.String,
            searchable=True,
            retrievable=True,
        ),
        SearchableField(
            name=content_field,
            type=SearchFieldDataType.String,
            searchable=True,
            retrievable=True,
        ),
        SearchField(
            name=content_vector_field,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            dimensions=embeddings_dim,
            vector_search_configuration="default",
        ),
        SearchableField(
            name=tag_field,
            type=SearchFieldDataType.String,
            filterable=True,
            searchable=True,
            retrievable=True,
        ),
        SearchableField(
            name=metadata_field,
            type=SearchFieldDataType.String,
            searchable=True,
            retrievable=True,
        ),
    ]
    # Vector search configuration
    algorithm_configuration = VectorSearchAlgorithmConfiguration(
        name="default",
        kind="hnsw",
        hnsw_parameters=HnswParameters(metric="cosine"),
    )
    vector_search = VectorSearch(algorithm_configurations=[algorithm_configuration])
    # Create the semantic settings with the configuration
    if semantic_configuration_name:
        semantic_settings = SemanticConfiguration(
            name=semantic_configuration_name,
            prioritized_fields=PrioritizedFields(
                title_field=SemanticField(field_name=title_field),
                prioritized_keywords_fields=[SemanticField(field_name=tag_field)],
                prioritized_content_fields=[SemanticField(field_name=content_field)],
            ),
        )
    else:
        semantic_settings = None
    # Create the search index with the semantic settings and vector search
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_settings=semantic_settings,
    )
    index_client.create_index(index)


class AzureSearch(VectorStore):
    def __init__(
        self,
        client: SearchClient,
        embeddings: Embeddings,
        *,
        semantic_configuration_name: Optional[str] = None,
        query_language: str = "en-us",
        id_field: str = "id",
        content_field: str = "content",
        content_vector_field: str = "content_vector",
        metadata_field: Optional[str] = "metadata",
        metadata_fields_to_extract: Optional[Sequence[str]] = None,
    ):
        """Initialize with necessary components."""
        self.client = client
        self.embeddings = embeddings
        self.semantic_configuration_name = semantic_configuration_name
        self.query_language = query_language
        self.id_field = id_field
        self.content_field = content_field
        self.content_vector_field = content_vector_field
        self.metadata_field = metadata_field
        self.metadata_fields_to_extract = list(metadata_fields_to_extract or [])

    @classmethod
    def from_params(
        cls,
        embeddings: Embeddings,
        *,
        service_name: Optional[str] = None,
        index_name: Optional[str] = None,
        api_key: str = "",
        semantic_configuration_name: Optional[str] = None,
        id_field: str = "id",
        content_field: str = "content",
        content_vector_field: str = "content_vector",
        metadata_field: str = "metadata",
        title_field: str = "title",
        tag_field: str = "tag",
        embeddings_dim: int = 1536,
        **kwargs: Any,
    ) -> AzureSearch:
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.identity import DefaultAzureCredential
            from azure.search.documents import SearchClient
        except ImportError:
            raise ImportError
        api_key = api_key or get_from_env("api_key", "AZURE_COGNITIVE_SEARCH_API_KEY")
        credential: Union[AzureKeyCredential, DefaultAzureCredential] = (
            AzureKeyCredential(api_key) if api_key else DefaultAzureCredential()
        )
        service_name = service_name or get_from_env(
            "service_name", "AZURE_COGNITIVE_SEARCH_SERVICE_NAME"
        )
        index_name = index_name or get_from_env(
            "index_name", "AZURE_COGNITIVE_SEARCH_INDEX_NAME", uuid.uuid4().hex
        )
        _check_index(
            service_name,
            credential,
            index_name,
            id_field,
            content_field,
            content_vector_field,
            metadata_field,
            title_field,
            tag_field,
            embeddings_dim,
            semantic_configuration_name,
        )
        base_url = f"https://{service_name}.search.windows.net"
        client = SearchClient(
            endpoint=base_url, index_name=index_name, credential=credential
        )
        return cls(
            client,
            embeddings,
            semantic_configuration_name=semantic_configuration_name,
            id_field=id_field,
            content_field=content_field,
            content_vector_field=content_vector_field,
            metadata_field=metadata_field,
            **kwargs,
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts data to an existing index."""
        _ids = ids or [str(uuid.uuid4()) for _ in texts]
        _metadatas: Iterable[Dict] = metadatas or ({} for _ in texts)
        documents = []
        for id, text, metadata in zip(_ids, texts, _metadatas):
            # Use provided id otherwise generate uuid.
            # Add data to index
            doc = {
                "@search.action": "upload",
                self.id_field: id,
                self.content_field: text,
                self.content_vector_field: self.embeddings.embed_documents([text])[0],
            }
            if metadata and (self.metadata_field or self.metadata_fields_to_extract):
                for key in self.metadata_fields_to_extract:
                    doc[key] = metadata.pop(key, "")
                if self.metadata_field:
                    doc[self.metadata_field] = json.dumps(metadata)
            documents.append(doc)
            # Upload data in batches
            if len(documents) == MAX_UPLOAD_BATCH_SIZE:
                response = self.client.upload_documents(documents=documents)
                # Check if all documents were successfully uploaded
                if not all([r.succeeded for r in response]):
                    raise Exception(response)
                # Reset data
                documents = []
        if documents:
            response = self.client.upload_documents(documents=documents)
            if not all([r.succeeded for r in response]):
                raise Exception(response)
        return _ids

    def _search_api(
        self,
        query: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        k: int = 4,
        **kwargs: Any,
    ) -> SearchItemPaged[Dict]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        from azure.search.documents._generated.models import Vector

        if query is not None:
            kwargs["search_text"] = query
        if embedding is not None:
            vector = Vector(value=embedding, fields=self.content_vector_field)
            kwargs["vector"] = vector
        return self.client.search(top=k, **kwargs)

    def _search_with_score(
        self,
        query: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        results = self._search_api(query=query, embedding=embedding, k=k, **kwargs)
        docs_scores = []
        for res in results:
            if self.metadata_field:
                metadata = json.loads(res[self.metadata_field])
            else:
                metadata = {}
            extra = {k: res[k] for k in self.metadata_fields_to_extract}
            metadata = {**metadata, **extra}
            doc = Document(page_content=res[self.content_field], metadata=metadata)
            score = 1 - float(res["@search.score"])
            docs_scores.append((doc, score))
        return docs_scores

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
        docs_and_scores = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        return self._search_with_score(query=query, k=k, **kwargs)

    def hybrid_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = self.hybrid_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def hybrid_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query with a hybrid query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embeddings.embed_query(query)
        return self._search_with_score(query=query, embedding=embedding, k=k, **kwargs)

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
        docs_and_scores = self.semantic_hybrid_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def semantic_hybrid_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query with a hybrid query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embeddings.embed_query(query)
        search_params = {
            **kwargs,
            "query_type": "semantic",
            "query_language": self.query_language,
            "semantic_configuration_name": self.semantic_configuration_name,
            "query_caption": "extractive",
            "query_answer": "extractive",
        }
        results = self._search_api(
            query=query, embedding=embedding, k=k, **search_params
        )
        # Get Semantic Answers
        semantic_answers_dict = {}
        if semantic_answers := results.get_answers():
            semantic_answers_dict = {
                r.key: {"text": r.text, "highlights": r.highlights}
                for r in semantic_answers
            }
        docs_scores = []
        for res in results:
            if self.metadata_field:
                metadata = json.loads(res[self.metadata_field])
            else:
                metadata = {}
            extra = {k: res[k] for k in self.metadata_fields_to_extract}
            if "@search.captions" in res and res["@search.captions"]:
                captions = res["@search.captions"][0]
                extra["captions"] = {
                    "text": captions.text,
                    "highlights": captions.highlights,
                }
            if "key" in metadata and metadata["key"] in semantic_answers_dict:
                extra["answers"] = semantic_answers_dict[metadata["key"]]
            metadata = {**metadata, **extra}
            doc = Document(page_content=res[self.content_field], metadata=metadata)
            score = 1 - float(res["@search.score"])
            docs_scores.append((doc, score))
        return docs_scores

    @classmethod
    def from_texts(
        cls: Type[AzureSearch],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        add_texts_params: Optional[dict] = None,
        **kwargs: Any,
    ) -> AzureSearch:
        azure_search = cls.from_params(**kwargs)
        add_texts_params = add_texts_params or {}
        azure_search.add_texts(texts, metadatas, **add_texts_params)
        return azure_search

    def as_retriever(self, **kwargs: Any) -> AzureSearchVectorStoreRetriever:
        return AzureSearchVectorStoreRetriever(vectorstore=self, **kwargs)


class AzureSearchVectorStoreRetriever(VectorStoreRetriever, BaseModel):
    vectorstore: AzureSearch

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
            docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        elif self.search_type == "hybrid":
            docs = self.vectorstore.hybrid_search(query, **self.search_kwargs)
        elif self.search_type == "semantic_hybrid":
            docs = self.vectorstore.semantic_hybrid_search(query, **self.search_kwargs)
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError(
            "AzureSearchVectorStoreRetriever does not support async"
        )
