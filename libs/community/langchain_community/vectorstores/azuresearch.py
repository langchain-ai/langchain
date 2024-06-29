from __future__ import annotations

import base64
import itertools
import json
import logging
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import numpy as np
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

logger = logging.getLogger()

if TYPE_CHECKING:
    from azure.search.documents import SearchClient, SearchItemPaged
    from azure.search.documents.aio import SearchClient as AsyncSearchClient
    from azure.search.documents.indexes.models import (
        CorsOptions,
        ScoringProfile,
        SearchField,
        SemanticConfiguration,
        VectorSearch,
    )

# Allow overriding field names for Azure Search
FIELDS_ID = get_from_env(
    key="AZURESEARCH_FIELDS_ID", env_key="AZURESEARCH_FIELDS_ID", default="id"
)
FIELDS_CONTENT = get_from_env(
    key="AZURESEARCH_FIELDS_CONTENT",
    env_key="AZURESEARCH_FIELDS_CONTENT",
    default="content",
)
FIELDS_CONTENT_VECTOR = get_from_env(
    key="AZURESEARCH_FIELDS_CONTENT_VECTOR",
    env_key="AZURESEARCH_FIELDS_CONTENT_VECTOR",
    default="content_vector",
)
FIELDS_METADATA = get_from_env(
    key="AZURESEARCH_FIELDS_TAG", env_key="AZURESEARCH_FIELDS_TAG", default="metadata"
)

MAX_UPLOAD_BATCH_SIZE = 1000


def _get_search_client(
    endpoint: str,
    key: str,
    index_name: str,
    semantic_configuration_name: Optional[str] = None,
    fields: Optional[List[SearchField]] = None,
    vector_search: Optional[VectorSearch] = None,
    semantic_configurations: Optional[
        Union[SemanticConfiguration, List[SemanticConfiguration]]
    ] = None,
    scoring_profiles: Optional[List[ScoringProfile]] = None,
    default_scoring_profile: Optional[str] = None,
    default_fields: Optional[List[SearchField]] = None,
    user_agent: Optional[str] = "langchain",
    cors_options: Optional[CorsOptions] = None,
    async_: bool = False,
) -> Union[SearchClient, AsyncSearchClient]:
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import ResourceNotFoundError
    from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.aio import SearchClient as AsyncSearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        ExhaustiveKnnAlgorithmConfiguration,
        ExhaustiveKnnParameters,
        HnswAlgorithmConfiguration,
        HnswParameters,
        SearchIndex,
        SemanticConfiguration,
        SemanticField,
        SemanticPrioritizedFields,
        SemanticSearch,
        VectorSearch,
        VectorSearchAlgorithmKind,
        VectorSearchAlgorithmMetric,
        VectorSearchProfile,
    )

    default_fields = default_fields or []
    if key is None:
        credential = DefaultAzureCredential()
    elif key.upper() == "INTERACTIVE":
        credential = InteractiveBrowserCredential()
        credential.get_token("https://search.azure.com/.default")
    else:
        credential = AzureKeyCredential(key)
    index_client: SearchIndexClient = SearchIndexClient(
        endpoint=endpoint, credential=credential, user_agent=user_agent
    )
    try:
        index_client.get_index(name=index_name)
    except ResourceNotFoundError:
        # Fields configuration
        if fields is not None:
            # Check mandatory fields
            fields_types = {f.name: f.type for f in fields}
            mandatory_fields = {df.name: df.type for df in default_fields}
            # Check for missing keys
            missing_fields = {
                key: mandatory_fields[key]
                for key, value in set(mandatory_fields.items())
                - set(fields_types.items())
            }
            if len(missing_fields) > 0:
                # Helper for formatting field information for each missing field.
                def fmt_err(x: str) -> str:
                    return (
                        f"{x} current type: '{fields_types.get(x, 'MISSING')}'. "
                        f"It has to be '{mandatory_fields.get(x)}' or you can point "
                        f"to a different '{mandatory_fields.get(x)}' field name by "
                        f"using the env variable 'AZURESEARCH_FIELDS_{x.upper()}'"
                    )

                error = "\n".join([fmt_err(x) for x in missing_fields])
                raise ValueError(
                    f"You need to specify at least the following fields "
                    f"{missing_fields} or provide alternative field names in the env "
                    f"variables.\n\n{error}"
                )
        else:
            fields = default_fields
        # Vector search configuration
        if vector_search is None:
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="default",
                        kind=VectorSearchAlgorithmKind.HNSW,
                        parameters=HnswParameters(
                            m=4,
                            ef_construction=400,
                            ef_search=500,
                            metric=VectorSearchAlgorithmMetric.COSINE,
                        ),
                    ),
                    ExhaustiveKnnAlgorithmConfiguration(
                        name="default_exhaustive_knn",
                        kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                        parameters=ExhaustiveKnnParameters(
                            metric=VectorSearchAlgorithmMetric.COSINE
                        ),
                    ),
                ],
                profiles=[
                    VectorSearchProfile(
                        name="myHnswProfile",
                        algorithm_configuration_name="default",
                    ),
                    VectorSearchProfile(
                        name="myExhaustiveKnnProfile",
                        algorithm_configuration_name="default_exhaustive_knn",
                    ),
                ],
            )

        # Create the semantic settings with the configuration
        if semantic_configurations:
            if not isinstance(semantic_configurations, list):
                semantic_configurations = [semantic_configurations]
            semantic_search = SemanticSearch(
                configurations=semantic_configurations,
                default_configuration_name=semantic_configuration_name,
            )
        elif semantic_configuration_name:
            # use default semantic configuration
            semantic_configuration = SemanticConfiguration(
                name=semantic_configuration_name,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name=FIELDS_CONTENT)],
                ),
            )
            semantic_search = SemanticSearch(configurations=[semantic_configuration])
        else:
            # don't use semantic search
            semantic_search = None

        # Create the search index with the semantic settings and vector search
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
            scoring_profiles=scoring_profiles,
            default_scoring_profile=default_scoring_profile,
            cors_options=cors_options,
        )
        index_client.create_index(index)
    # Create the search client
    if not async_:
        return SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=credential,
            user_agent=user_agent,
        )
    else:
        return AsyncSearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=credential,
            user_agent=user_agent,
        )


class AzureSearch(VectorStore):
    """`Azure Cognitive Search` vector store."""

    def __init__(
        self,
        azure_search_endpoint: str,
        azure_search_key: str,
        index_name: str,
        embedding_function: Union[Callable, Embeddings],
        search_type: str = "hybrid",
        semantic_configuration_name: Optional[str] = None,
        fields: Optional[List[SearchField]] = None,
        vector_search: Optional[VectorSearch] = None,
        semantic_configurations: Optional[
            Union[SemanticConfiguration, List[SemanticConfiguration]]
        ] = None,
        scoring_profiles: Optional[List[ScoringProfile]] = None,
        default_scoring_profile: Optional[str] = None,
        cors_options: Optional[CorsOptions] = None,
        *,
        vector_search_dimensions: Optional[int] = None,
        **kwargs: Any,
    ):
        try:
            from azure.search.documents.indexes.models import (
                SearchableField,
                SearchField,
                SearchFieldDataType,
                SimpleField,
            )
        except ImportError as e:
            raise ImportError(
                "Unable to import azure.search.documents. Please install with "
                "`pip install -U azure-search-documents`."
            ) from e

        """Initialize with necessary components."""
        # Initialize base class
        self.embedding_function = embedding_function

        if isinstance(self.embedding_function, Embeddings):
            self.embed_query = self.embedding_function.embed_query
        else:
            self.embed_query = self.embedding_function

        default_fields = [
            SimpleField(
                name=FIELDS_ID,
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchableField(
                name=FIELDS_CONTENT,
                type=SearchFieldDataType.String,
            ),
            SearchField(
                name=FIELDS_CONTENT_VECTOR,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=vector_search_dimensions
                or len(self.embed_query("Text")),
                vector_search_profile_name="myHnswProfile",
            ),
            SearchableField(
                name=FIELDS_METADATA,
                type=SearchFieldDataType.String,
            ),
        ]
        user_agent = "langchain"
        if "user_agent" in kwargs and kwargs["user_agent"]:
            user_agent += " " + kwargs["user_agent"]
        self.client = _get_search_client(
            azure_search_endpoint,
            azure_search_key,
            index_name,
            semantic_configuration_name=semantic_configuration_name,
            fields=fields,
            vector_search=vector_search,
            semantic_configurations=semantic_configurations,
            scoring_profiles=scoring_profiles,
            default_scoring_profile=default_scoring_profile,
            default_fields=default_fields,
            user_agent=user_agent,
            cors_options=cors_options,
        )
        self.search_type = search_type
        self.semantic_configuration_name = semantic_configuration_name
        self.fields = fields if fields else default_fields

        self._azure_search_endpoint = azure_search_endpoint
        self._azure_search_key = azure_search_key
        self._index_name = index_name
        self._semantic_configuration_name = semantic_configuration_name
        self._fields = fields
        self._vector_search = vector_search
        self._semantic_configurations = semantic_configurations
        self._scoring_profiles = scoring_profiles
        self._default_scoring_profile = default_scoring_profile
        self._default_fields = default_fields
        self._user_agent = user_agent
        self._cors_options = cors_options

    def _async_client(self) -> AsyncSearchClient:
        return _get_search_client(
            self._azure_search_endpoint,
            self._azure_search_key,
            self._index_name,
            semantic_configuration_name=self._semantic_configuration_name,
            fields=self._fields,
            vector_search=self._vector_search,
            semantic_configurations=self._semantic_configurations,
            scoring_profiles=self._scoring_profiles,
            default_scoring_profile=self._default_scoring_profile,
            default_fields=self._default_fields,
            user_agent=self._user_agent,
            cors_options=self._cors_options,
            async_=True,
        )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        # TODO: Support embedding object directly
        return (
            self.embedding_function
            if isinstance(self.embedding_function, Embeddings)
            else None
        )

    async def _aembed_query(self, text: str) -> List[float]:
        if self.embeddings:
            return await self.embeddings.aembed_query(text)
        else:
            return cast(Callable, self.embedding_function)(text)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        keys: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts data to an existing index."""
        # batching support if embedding function is an Embeddings object
        if isinstance(self.embedding_function, Embeddings):
            try:
                embeddings = self.embedding_function.embed_documents(list(texts))
            except NotImplementedError:
                embeddings = [self.embedding_function.embed_query(x) for x in texts]
        else:
            embeddings = [self.embedding_function(x) for x in texts]

        if len(embeddings) == 0:
            logger.debug("Nothing to insert, skipping.")
            return []

        return self.add_embeddings(zip(texts, embeddings), metadatas, keys=keys)

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        keys: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if isinstance(self.embedding_function, Embeddings):
            try:
                embeddings = await self.embedding_function.aembed_documents(list(texts))
            except NotImplementedError:
                embeddings = [
                    await self.embedding_function.aembed_query(x) for x in texts
                ]
        else:
            embeddings = [self.embedding_function(x) for x in texts]

        if len(embeddings) == 0:
            logger.debug("Nothing to insert, skipping.")
            return []

        return await self.aadd_embeddings(zip(texts, embeddings), metadatas, keys=keys)

    def add_embeddings(
        self,
        text_embeddings: Iterable[Tuple[str, List[float]]],
        metadatas: Optional[List[dict]] = None,
        *,
        keys: Optional[List[str]] = None,
    ) -> List[str]:
        """Add embeddings to an existing index."""
        ids = []

        # Write data to index
        data = []
        for i, (text, embedding) in enumerate(text_embeddings):
            # Use provided key otherwise use default key
            key = keys[i] if keys else str(uuid.uuid4())
            # Encoding key for Azure Search valid characters
            key = base64.urlsafe_b64encode(bytes(key, "utf-8")).decode("ascii")
            metadata = metadatas[i] if metadatas else {}
            # Add data to index
            # Additional metadata to fields mapping
            doc = {
                "@search.action": "upload",
                FIELDS_ID: key,
                FIELDS_CONTENT: text,
                FIELDS_CONTENT_VECTOR: np.array(embedding, dtype=np.float32).tolist(),
                FIELDS_METADATA: json.dumps(metadata),
            }
            if metadata:
                additional_fields = {
                    k: v
                    for k, v in metadata.items()
                    if k in [x.name for x in self.fields]
                }
                doc.update(additional_fields)
            data.append(doc)
            ids.append(key)
            # Upload data in batches
            if len(data) == MAX_UPLOAD_BATCH_SIZE:
                response = self.client.upload_documents(documents=data)
                # Check if all documents were successfully uploaded
                if not all(r.succeeded for r in response):
                    raise Exception(response)
                # Reset data
                data = []

        # Considering case where data is an exact multiple of batch-size entries
        if len(data) == 0:
            return ids

        # Upload data to index
        response = self.client.upload_documents(documents=data)
        # Check if all documents were successfully uploaded
        if all(r.succeeded for r in response):
            return ids
        else:
            raise Exception(response)

    async def aadd_embeddings(
        self,
        text_embeddings: Iterable[Tuple[str, List[float]]],
        metadatas: Optional[List[dict]] = None,
        *,
        keys: Optional[List[str]] = None,
    ) -> List[str]:
        """Add embeddings to an existing index."""
        ids = []

        # Write data to index
        data = []
        for i, (text, embedding) in enumerate(text_embeddings):
            # Use provided key otherwise use default key
            key = keys[i] if keys else str(uuid.uuid4())
            # Encoding key for Azure Search valid characters
            key = base64.urlsafe_b64encode(bytes(key, "utf-8")).decode("ascii")
            metadata = metadatas[i] if metadatas else {}
            # Add data to index
            # Additional metadata to fields mapping
            doc = {
                "@search.action": "upload",
                FIELDS_ID: key,
                FIELDS_CONTENT: text,
                FIELDS_CONTENT_VECTOR: np.array(embedding, dtype=np.float32).tolist(),
                FIELDS_METADATA: json.dumps(metadata),
            }
            if metadata:
                additional_fields = {
                    k: v
                    for k, v in metadata.items()
                    if k in [x.name for x in self.fields]
                }
                doc.update(additional_fields)
            data.append(doc)
            ids.append(key)
            # Upload data in batches
            if len(data) == MAX_UPLOAD_BATCH_SIZE:
                async with self._async_client() as async_client:
                    response = await async_client.upload_documents(documents=data)
                    # Check if all documents were successfully uploaded
                    if not all(r.succeeded for r in response):
                        raise Exception(response)
                    # Reset data
                    data = []

        # Considering case where data is an exact multiple of batch-size entries
        if len(data) == 0:
            return ids

        # Upload data to index
        async with self._async_client() as async_client:
            response = await async_client.upload_documents(documents=data)
        # Check if all documents were successfully uploaded
        if all(r.succeeded for r in response):
            return ids
        else:
            raise Exception(response)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        """Delete by vector ID.

        Args:
            ids: List of ids to delete.

        Returns:
            bool: True if deletion is successful,
            False otherwise.
        """
        if ids:
            res = self.client.delete_documents([{FIELDS_ID: i} for i in ids])
            return len(res) > 0
        else:
            return False

    async def adelete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        """Delete by vector ID.

        Args:
            ids: List of ids to delete.

        Returns:
            bool: True if deletion is successful,
            False otherwise.
        """
        if ids:
            async with self._async_client() as async_client:
                res = await async_client.delete_documents([{"id": i} for i in ids])
                return len(res) > 0
        else:
            return False

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        search_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        search_type = search_type or self.search_type
        if search_type == "similarity":
            docs = self.vector_search(query, k=k, **kwargs)
        elif search_type == "hybrid":
            docs = self.hybrid_search(query, k=k, **kwargs)
        elif search_type == "semantic_hybrid":
            docs = self.semantic_hybrid_search(query, k=k, **kwargs)
        else:
            raise ValueError(f"search_type of {search_type} not allowed.")
        return docs

    def similarity_search_with_score(
        self, query: str, *, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance."""
        search_type = kwargs.get("search_type", self.search_type)
        if search_type == "similarity":
            return self.vector_search_with_score(query, k=k, **kwargs)
        elif search_type == "hybrid":
            return self.hybrid_search_with_score(query, k=k, **kwargs)
        elif search_type == "semantic_hybrid":
            return self.semantic_hybrid_search_with_score(query, k=k, **kwargs)
        else:
            raise ValueError(f"search_type of {search_type} not allowed.")

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        search_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        search_type = search_type or self.search_type
        if search_type == "similarity":
            docs = await self.avector_search(query, k=k, **kwargs)
        elif search_type == "hybrid":
            docs = await self.ahybrid_search(query, k=k, **kwargs)
        elif search_type == "semantic_hybrid":
            docs = await self.asemantic_hybrid_search(query, k=k, **kwargs)
        else:
            raise ValueError(f"search_type of {search_type} not allowed.")
        return docs

    async def asimilarity_search_with_score(
        self, query: str, *, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance."""
        search_type = kwargs.get("search_type", self.search_type)
        if search_type == "similarity":
            return await self.avector_search_with_score(query, k=k, **kwargs)
        elif search_type == "hybrid":
            return await self.ahybrid_search_with_score(query, k=k, **kwargs)
        elif search_type == "semantic_hybrid":
            return await self.asemantic_hybrid_search_with_score(query, k=k, **kwargs)
        else:
            raise ValueError(f"search_type of {search_type} not allowed.")

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        *,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        result = self.vector_search_with_score(query, k=k, **kwargs)
        return (
            result
            if score_threshold is None
            else [r for r in result if r[1] >= score_threshold]
        )

    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        *,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        result = await self.avector_search_with_score(query, k=k, **kwargs)
        return (
            result
            if score_threshold is None
            else [r for r in result if r[1] >= score_threshold]
        )

    def vector_search(
        self, query: str, k: int = 4, *, filters: Optional[str] = None, **kwargs: Any
    ) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = self.vector_search_with_score(query, k=k, filters=filters)
        return [doc for doc, _ in docs_and_scores]

    async def avector_search(
        self, query: str, k: int = 4, *, filters: Optional[str] = None, **kwargs: Any
    ) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = await self.avector_search_with_score(
            query, k=k, filters=filters
        )
        return [doc for doc, _ in docs_and_scores]

    def vector_search_with_score(
        self,
        query: str,
        k: int = 4,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query (str): Text to look up documents similar to.
            k (int, optional): Number of Documents to return. Defaults to 4.
            filters (str, optional): Filtering expression. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of Documents most similar
                to the query and score for each
        """
        embedding = self.embed_query(query)
        results = self._simple_search(embedding, "", k, filters=filters, **kwargs)

        return _results_to_documents(results)

    async def avector_search_with_score(
        self,
        query: str,
        k: int = 4,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query (str): Text to look up documents similar to.
            k (int, optional): Number of Documents to return. Defaults to 4.
            filters (str, optional): Filtering expression. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of Documents most similar
                to the query and score for each
        """
        embedding = await self._aembed_query(query)
        docs, scores, _ = await self._asimple_search(
            embedding, "", k, filters=filters, **kwargs
        )

        return list(zip(docs, scores))

    def max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search and return results that are reordered by MMR.

        Args:
            query (str): Text to look up documents similar to.
            k (int, optional): How many results to give. Defaults to 4.
            fetch_k (int, optional): Total results to select k from.
                Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5
            filters (str, optional): Filtering expression. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of Documents most similar
                to the query and score for each
        """
        embedding = self.embed_query(query)
        results = self._simple_search(embedding, "", fetch_k, filters=filters, **kwargs)

        return _reorder_results_with_maximal_marginal_relevance(
            results, query_embedding=np.array(embedding), lambda_mult=lambda_mult, k=k
        )

    async def amax_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search and return results that are reordered by MMR.

        Args:
            query (str): Text to look up documents similar to.
            k (int, optional): How many results to give. Defaults to 4.
            fetch_k (int, optional): Total results to select k from.
                Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5
            filters (str, optional): Filtering expression. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of Documents most similar
                to the query and score for each
        """
        embedding = await self._aembed_query(query)
        docs, scores, vectors = await self._asimple_search(
            embedding, "", fetch_k, filters=filters, **kwargs
        )

        return await self._areorder_results_with_maximal_marginal_relevance(
            docs,
            scores,
            vectors,
            query_embedding=np.array(embedding),
            lambda_mult=lambda_mult,
            k=k,
        )

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

    async def ahybrid_search(
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
        docs_and_scores = await self.ahybrid_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def hybrid_search_with_score(
        self,
        query: str,
        k: int = 4,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query with a hybrid query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """

        embedding = self.embed_query(query)
        results = self._simple_search(embedding, query, k, filters=filters, **kwargs)

        return _results_to_documents(results)

    async def ahybrid_search_with_score(
        self,
        query: str,
        k: int = 4,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query with a hybrid query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """

        embedding = await self._aembed_query(query)
        docs, scores, _ = await self._asimple_search(
            embedding, query, k, filters=filters, **kwargs
        )

        return list(zip(docs, scores))

    def hybrid_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        *,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        result = self.hybrid_search_with_score(query, k=k, **kwargs)
        return (
            result
            if score_threshold is None
            else [r for r in result if r[1] >= score_threshold]
        )

    async def ahybrid_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        *,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        result = await self.ahybrid_search_with_score(query, k=k, **kwargs)
        return (
            result
            if score_threshold is None
            else [r for r in result if r[1] >= score_threshold]
        )

    def hybrid_max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query with a hybrid query
            and reorder results by MMR.

        Args:
            query (str): Text to look up documents similar to.
            k (int, optional): Number of Documents to return. Defaults to 4.
            fetch_k (int, optional): Total results to select k from.
                Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5
            filters (str, optional): Filtering expression. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each
        """

        embedding = self.embed_query(query)
        results = self._simple_search(
            embedding, query, fetch_k, filters=filters, **kwargs
        )

        return _reorder_results_with_maximal_marginal_relevance(
            results, query_embedding=np.array(embedding), lambda_mult=lambda_mult, k=k
        )

    async def ahybrid_max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query with a hybrid query
            and reorder results by MMR.

        Args:
            query (str): Text to look up documents similar to.
            k (int, optional): Number of Documents to return. Defaults to 4.
            fetch_k (int, optional): Total results to select k from.
                Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5
            filters (str, optional): Filtering expression. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each
        """

        embedding = await self._aembed_query(query)
        docs, scores, vectors = await self._asimple_search(
            embedding, query, fetch_k, filters=filters, **kwargs
        )

        return await self._areorder_results_with_maximal_marginal_relevance(
            docs,
            scores,
            vectors,
            query_embedding=np.array(embedding),
            lambda_mult=lambda_mult,
            k=k,
        )

    def _simple_search(
        self,
        embedding: List[float],
        text_query: str,
        k: int,
        *,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> SearchItemPaged[dict]:
        """Perform vector or hybrid search in the Azure search index.

        Args:
            embedding: A vector embedding to search in the vector space.
            text_query: A full-text search query expression;
                Use "*" or omit this parameter to perform only vector search.
            k: Number of documents to return.
            filters: Filtering expression.
        Returns:
            Search items
        """
        from azure.search.documents.models import VectorizedQuery

        return self.client.search(
            search_text=text_query,
            vector_queries=[
                VectorizedQuery(
                    vector=np.array(embedding, dtype=np.float32).tolist(),
                    k_nearest_neighbors=k,
                    fields=FIELDS_CONTENT_VECTOR,
                )
            ],
            filter=filters,
            top=k,
            **kwargs,
        )

    async def _asimple_search(
        self,
        embedding: List[float],
        text_query: str,
        k: int,
        *,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[List[Document], List[float], List[List[float]]]:
        """Perform vector or hybrid search in the Azure search index.

        Args:
            embedding: A vector embedding to search in the vector space.
            text_query: A full-text search query expression;
                Use "*" or omit this parameter to perform only vector search.
            k: Number of documents to return.
            filters: Filtering expression.
        Returns:
            Search items
        """
        from azure.search.documents.models import VectorizedQuery

        async with self._async_client() as async_client:
            results = await async_client.search(
                search_text=text_query,
                vector_queries=[
                    VectorizedQuery(
                        vector=np.array(embedding, dtype=np.float32).tolist(),
                        k_nearest_neighbors=k,
                        fields=FIELDS_CONTENT_VECTOR,
                    )
                ],
                filter=filters,
                top=k,
                **kwargs,
            )
            docs = [
                (
                    _result_to_document(result),
                    float(result["@search.score"]),
                    result[FIELDS_CONTENT_VECTOR],
                )
                async for result in results
            ]
            if not docs:
                raise ValueError(f"No {docs=}")
        documents, scores, vectors = map(list, zip(*docs))
        return documents, scores, vectors

    def semantic_hybrid_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            filters: Filtering expression.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = self.semantic_hybrid_search_with_score_and_rerank(
            query, k=k, **kwargs
        )
        return [doc for doc, _, _ in docs_and_scores]

    async def asemantic_hybrid_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            filters: Filtering expression.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = await self.asemantic_hybrid_search_with_score_and_rerank(
            query, k=k, **kwargs
        )
        return [doc for doc, _, _ in docs_and_scores]

    def semantic_hybrid_search_with_score(
        self,
        query: str,
        k: int = 4,
        score_type: Literal["score", "reranker_score"] = "score",
        *,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            score_type: Must either be "score" or "reranker_score".
                Defaulted to "score".
            filters: Filtering expression.

        Returns:
            List[Tuple[Document, float]]: A list of documents and their
                corresponding scores.
        """
        docs_and_scores = self.semantic_hybrid_search_with_score_and_rerank(
            query, k=k, **kwargs
        )
        if score_type == "score":
            return [
                (doc, score)
                for doc, score, _ in docs_and_scores
                if score_threshold is None or score >= score_threshold
            ]
        elif score_type == "reranker_score":
            return [
                (doc, reranker_score)
                for doc, _, reranker_score in docs_and_scores
                if score_threshold is None or reranker_score >= score_threshold
            ]

    async def asemantic_hybrid_search_with_score(
        self,
        query: str,
        k: int = 4,
        score_type: Literal["score", "reranker_score"] = "score",
        *,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            score_type: Must either be "score" or "reranker_score".
                Defaulted to "score".
            filters: Filtering expression.

        Returns:
            List[Tuple[Document, float]]: A list of documents and their
                corresponding scores.
        """
        docs_and_scores = await self.asemantic_hybrid_search_with_score_and_rerank(
            query, k=k, **kwargs
        )
        if score_type == "score":
            return [
                (doc, score)
                for doc, score, _ in docs_and_scores
                if score_threshold is None or score >= score_threshold
            ]
        elif score_type == "reranker_score":
            return [
                (doc, reranker_score)
                for doc, _, reranker_score in docs_and_scores
                if score_threshold is None or reranker_score >= score_threshold
            ]

    def semantic_hybrid_search_with_score_and_rerank(
        self, query: str, k: int = 4, *, filters: Optional[str] = None, **kwargs: Any
    ) -> List[Tuple[Document, float, float]]:
        """Return docs most similar to query with a hybrid query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filters: Filtering expression.

        Returns:
            List of Documents most similar to the query and score for each
        """
        from azure.search.documents.models import VectorizedQuery

        results = self.client.search(
            search_text=query,
            vector_queries=[
                VectorizedQuery(
                    vector=np.array(self.embed_query(query), dtype=np.float32).tolist(),
                    k_nearest_neighbors=k,
                    fields=FIELDS_CONTENT_VECTOR,
                )
            ],
            filter=filters,
            query_type="semantic",
            semantic_configuration_name=self.semantic_configuration_name,
            query_caption="extractive",
            query_answer="extractive",
            top=k,
            **kwargs,
        )
        # Get Semantic Answers
        semantic_answers = results.get_answers() or []
        semantic_answers_dict: Dict = {}
        for semantic_answer in semantic_answers:
            semantic_answers_dict[semantic_answer.key] = {
                "text": semantic_answer.text,
                "highlights": semantic_answer.highlights,
            }
        # Convert results to Document objects
        docs = [
            (
                Document(
                    page_content=result.pop(FIELDS_CONTENT),
                    metadata={
                        **(
                            json.loads(result[FIELDS_METADATA])
                            if FIELDS_METADATA in result
                            else {
                                k: v
                                for k, v in result.items()
                                if k != FIELDS_CONTENT_VECTOR
                            }
                        ),
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
                                result.get(FIELDS_ID, ""),
                                "",
                            ),
                        },
                    },
                ),
                float(result["@search.score"]),
                float(result["@search.reranker_score"]),
            )
            for result in results
        ]
        return docs

    async def asemantic_hybrid_search_with_score_and_rerank(
        self, query: str, k: int = 4, *, filters: Optional[str] = None, **kwargs: Any
    ) -> List[Tuple[Document, float, float]]:
        """Return docs most similar to query with a hybrid query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filters: Filtering expression.

        Returns:
            List of Documents most similar to the query and score for each
        """
        from azure.search.documents.models import VectorizedQuery

        vector = await self._aembed_query(query)
        async with self._async_client() as async_client:
            results = await async_client.search(
                search_text=query,
                vector_queries=[
                    VectorizedQuery(
                        vector=np.array(vector, dtype=np.float32).tolist(),
                        k_nearest_neighbors=k,
                        fields=FIELDS_CONTENT_VECTOR,
                    )
                ],
                filter=filters,
                query_type="semantic",
                semantic_configuration_name=self.semantic_configuration_name,
                query_caption="extractive",
                query_answer="extractive",
                top=k,
                **kwargs,
            )
            # Get Semantic Answers
            semantic_answers = (await results.get_answers()) or []
            semantic_answers_dict: Dict = {}
            for semantic_answer in semantic_answers:
                semantic_answers_dict[semantic_answer.key] = {
                    "text": semantic_answer.text,
                    "highlights": semantic_answer.highlights,
                }
            # Convert results to Document objects
            docs = [
                (
                    Document(
                        page_content=result.pop(FIELDS_CONTENT),
                        metadata={
                            **(
                                json.loads(result[FIELDS_METADATA])
                                if FIELDS_METADATA in result
                                else {
                                    k: v
                                    for k, v in result.items()
                                    if k != FIELDS_CONTENT_VECTOR
                                }
                            ),
                            **{
                                "captions": {
                                    "text": result.get("@search.captions", [{}])[
                                        0
                                    ].text,
                                    "highlights": result.get("@search.captions", [{}])[
                                        0
                                    ].highlights,
                                }
                                if result.get("@search.captions")
                                else {},
                                "answers": semantic_answers_dict.get(
                                    result.get(FIELDS_ID, ""),
                                    "",
                                ),
                            },
                        },
                    ),
                    float(result["@search.score"]),
                    float(result["@search.reranker_score"]),
                )
                async for result in results
            ]
            return docs

    @classmethod
    def from_texts(
        cls: Type[AzureSearch],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        azure_search_endpoint: str = "",
        azure_search_key: str = "",
        index_name: str = "langchain-index",
        fields: Optional[List[SearchField]] = None,
        **kwargs: Any,
    ) -> AzureSearch:
        # Creating a new Azure Search instance
        azure_search = cls(
            azure_search_endpoint,
            azure_search_key,
            index_name,
            embedding,
            fields=fields,
            **kwargs,
        )
        azure_search.add_texts(texts, metadatas, **kwargs)
        return azure_search

    @classmethod
    async def afrom_texts(
        cls: Type[AzureSearch],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        azure_search_endpoint: str = "",
        azure_search_key: str = "",
        index_name: str = "langchain-index",
        fields: Optional[List[SearchField]] = None,
        **kwargs: Any,
    ) -> AzureSearch:
        # Creating a new Azure Search instance
        azure_search = cls(
            azure_search_endpoint,
            azure_search_key,
            index_name,
            embedding,
            fields=fields,
            **kwargs,
        )
        await azure_search.aadd_texts(texts, metadatas, **kwargs)
        return azure_search

    @classmethod
    async def afrom_embeddings(
        cls: Type[AzureSearch],
        text_embeddings: Iterable[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        azure_search_endpoint: str = "",
        azure_search_key: str = "",
        index_name: str = "langchain-index",
        fields: Optional[List[SearchField]] = None,
        **kwargs: Any,
    ) -> AzureSearch:
        text_embeddings, first_text_embedding = _peek(text_embeddings)
        if first_text_embedding is None:
            raise ValueError("Cannot create AzureSearch from empty embeddings.")
        vector_search_dimensions = len(first_text_embedding[1])

        azure_search = cls(
            azure_search_endpoint=azure_search_endpoint,
            azure_search_key=azure_search_key,
            index_name=index_name,
            embedding_function=embedding,
            fields=fields,
            vector_search_dimensions=vector_search_dimensions,
            **kwargs,
        )
        await azure_search.aadd_embeddings(text_embeddings, metadatas, **kwargs)
        return azure_search

    @classmethod
    def from_embeddings(
        cls: Type[AzureSearch],
        text_embeddings: Iterable[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        azure_search_endpoint: str = "",
        azure_search_key: str = "",
        index_name: str = "langchain-index",
        fields: Optional[List[SearchField]] = None,
        **kwargs: Any,
    ) -> AzureSearch:
        # Creating a new Azure Search instance
        text_embeddings, first_text_embedding = _peek(text_embeddings)
        if first_text_embedding is None:
            raise ValueError("Cannot create AzureSearch from empty embeddings.")
        vector_search_dimensions = len(first_text_embedding[1])

        azure_search = cls(
            azure_search_endpoint=azure_search_endpoint,
            azure_search_key=azure_search_key,
            index_name=index_name,
            embedding_function=embedding,
            fields=fields,
            vector_search_dimensions=vector_search_dimensions,
            **kwargs,
        )
        azure_search.add_embeddings(text_embeddings, metadatas, **kwargs)
        return azure_search

    async def _areorder_results_with_maximal_marginal_relevance(
        self,
        documents: List[Document],
        scores: List[float],
        vectors: List[List[float]],
        query_embedding: np.ndarray,
        lambda_mult: float = 0.5,
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        # Get the new order of results.
        new_ordering = maximal_marginal_relevance(
            query_embedding, vectors, k=k, lambda_mult=lambda_mult
        )

        # Reorder the values and return.
        ret: List[Tuple[Document, float]] = []
        for x in new_ordering:
            # Function can return -1 index
            if x == -1:
                break
            ret.append((documents[x], scores[x]))  # type: ignore

        return ret

    def as_retriever(self, **kwargs: Any) -> AzureSearchVectorStoreRetriever:  # type: ignore
        """Return AzureSearchVectorStoreRetriever initialized from this VectorStore.

        Args:
            search_type (Optional[str]): Defines the type of search that
                the Retriever should perform.
                Can be "similarity" (default), "hybrid", or
                    "semantic_hybrid".
            search_kwargs (Optional[Dict]): Keyword arguments to pass to the
                search function. Can include things like:
                    score_threshold: Minimum relevance threshold
                        for similarity_score_threshold
                    fetch_k: Amount of documents to pass to MMR algorithm (Default: 20)
                    lambda_mult: Diversity of results returned by MMR;
                        1 for minimum diversity and 0 for maximum. (Default: 0.5)
                    filter: Filter by document metadata

        Returns:
            AzureSearchVectorStoreRetriever: Retriever class for VectorStore.
        """
        tags = kwargs.pop("tags", None) or []
        tags.extend(self._get_retriever_tags())
        return AzureSearchVectorStoreRetriever(vectorstore=self, **kwargs, tags=tags)


class AzureSearchVectorStoreRetriever(BaseRetriever):
    """Retriever that uses `Azure Cognitive Search`."""

    vectorstore: AzureSearch
    """Azure Search instance used to find similar documents."""
    search_type: str = "hybrid"
    """Type of search to perform. Options are "similarity", "hybrid",
    "semantic_hybrid", "similarity_score_threshold", "hybrid_score_threshold", 
    or "semantic_hybrid_score_threshold"."""
    k: int = 4
    """Number of documents to return."""
    search_kwargs: dict = {}
    """Search params.
        score_threshold: Minimum relevance threshold
            for similarity_score_threshold
        fetch_k: Amount of documents to pass to MMR algorithm (Default: 20)
        lambda_mult: Diversity of results returned by MMR;
            1 for minimum diversity and 0 for maximum. (Default: 0.5)
        filter: Filter by document metadata
    """

    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold",
        "hybrid",
        "hybrid_score_threshold",
        "semantic_hybrid",
        "semantic_hybrid_score_threshold",
    )

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator()
    def validate_search_type(cls, values: Dict) -> Dict:
        """Validate search type."""
        if "search_type" in values:
            search_type = values["search_type"]
            if search_type not in cls.allowed_search_types:
                raise ValueError(
                    f"search_type of {search_type} not allowed. Valid values are: "
                    f"{cls.allowed_search_types}"
                )
        return values

    def _get_relevant_documents(
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        params = {**self.search_kwargs, **kwargs}

        if self.search_type == "similarity":
            docs = self.vectorstore.vector_search(query, k=self.k, **params)
        elif self.search_type == "similarity_score_threshold":
            docs = [
                doc
                for doc, _ in self.vectorstore.similarity_search_with_relevance_scores(
                    query, k=self.k, **params
                )
            ]
        elif self.search_type == "hybrid":
            docs = self.vectorstore.hybrid_search(query, k=self.k, **params)
        elif self.search_type == "hybrid_score_threshold":
            docs = [
                doc
                for doc, _ in self.vectorstore.hybrid_search_with_relevance_scores(
                    query, k=self.k, **params
                )
            ]
        elif self.search_type == "semantic_hybrid":
            docs = self.vectorstore.semantic_hybrid_search(query, k=self.k, **params)
        elif self.search_type == "semantic_hybrid_score_threshold":
            docs = [
                doc
                for doc, _ in self.vectorstore.semantic_hybrid_search_with_score(
                    query, k=self.k, **params
                )
            ]
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        params = {**self.search_kwargs, **kwargs}

        if self.search_type == "similarity":
            docs = await self.vectorstore.avector_search(query, k=self.k, **params)
        elif self.search_type == "similarity_score_threshold":
            docs_and_scores = (
                await self.vectorstore.asimilarity_search_with_relevance_scores(
                    query, k=self.k, **params
                )
            )
            docs = [doc for doc, _ in docs_and_scores]
        elif self.search_type == "hybrid":
            docs = await self.vectorstore.ahybrid_search(query, k=self.k, **params)
        elif self.search_type == "hybrid_score_threshold":
            docs_and_scores = (
                await self.vectorstore.ahybrid_search_with_relevance_scores(
                    query, k=self.k, **params
                )
            )
            docs = [doc for doc, _ in docs_and_scores]
        elif self.search_type == "semantic_hybrid":
            docs = await self.vectorstore.asemantic_hybrid_search(
                query, k=self.k, **params
            )
        elif self.search_type == "semantic_hybrid_score_threshold":
            docs = [
                doc
                for doc, _ in await self.vectorstore.asemantic_hybrid_search_with_score(
                    query, k=self.k, **params
                )
            ]
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs


def _results_to_documents(
    results: SearchItemPaged[Dict],
) -> List[Tuple[Document, float]]:
    docs = [
        (
            _result_to_document(result),
            float(result["@search.score"]),
        )
        for result in results
    ]
    return docs


def _reorder_results_with_maximal_marginal_relevance(
    results: SearchItemPaged[Dict],
    query_embedding: np.ndarray,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[Tuple[Document, float]]:
    # Convert results to Document objects
    docs = [
        (
            _result_to_document(result),
            float(result["@search.score"]),
            result[FIELDS_CONTENT_VECTOR],
        )
        for result in results
    ]
    documents, scores, vectors = map(list, zip(*docs))

    # Get the new order of results.
    new_ordering = maximal_marginal_relevance(
        query_embedding, vectors, k=k, lambda_mult=lambda_mult
    )

    # Reorder the values and return.
    ret: List[Tuple[Document, float]] = []
    for x in new_ordering:
        # Function can return -1 index
        if x == -1:
            break
        ret.append((documents[x], scores[x]))  # type: ignore

    return ret


def _result_to_document(result: Dict) -> Document:
    return Document(
        page_content=result.pop(FIELDS_CONTENT),
        metadata=json.loads(result[FIELDS_METADATA])
        if FIELDS_METADATA in result
        else {
            key: value for key, value in result.items() if key != FIELDS_CONTENT_VECTOR
        },
    )


def _peek(iterable: Iterable, default: Optional[Any] = None) -> Tuple[Iterable, Any]:
    try:
        iterator = iter(iterable)
        value = next(iterator)
        iterable = itertools.chain([value], iterator)
        return iterable, value
    except StopIteration:
        return iterable, default
