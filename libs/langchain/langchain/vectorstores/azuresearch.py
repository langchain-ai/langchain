from __future__ import annotations

import base64
import json
import logging
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

import numpy as np

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.docstore.document import Document
from langchain.pydantic_v1 import root_validator
from langchain.schema import BaseRetriever
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.utils import get_from_env

logger = logging.getLogger()

if TYPE_CHECKING:
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes.models import (
        ScoringProfile,
        SearchField,
        SemanticSettings,
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
    semantic_settings: Optional[SemanticSettings] = None,
    scoring_profiles: Optional[List[ScoringProfile]] = None,
    default_scoring_profile: Optional[str] = None,
    default_fields: Optional[List[SearchField]] = None,
    user_agent: Optional[str] = "langchain",
) -> SearchClient:
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import ResourceNotFoundError
    from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        HnswVectorSearchAlgorithmConfiguration,
        PrioritizedFields,
        SearchIndex,
        SemanticConfiguration,
        SemanticField,
        SemanticSettings,
        VectorSearch,
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
                fmt_err = lambda x: (  # noqa: E731
                    f"{x} current type: '{fields_types.get(x, 'MISSING')}'. It has to "
                    f"be '{mandatory_fields.get(x)}' or you can point to a different "
                    f"'{mandatory_fields.get(x)}' field name by using the env variable "
                    f"'AZURESEARCH_FIELDS_{x.upper()}'"
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
                algorithm_configurations=[
                    HnswVectorSearchAlgorithmConfiguration(
                        name="default",
                        kind="hnsw",
                        parameters={  # type: ignore
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine",
                        },
                    )
                ]
            )
        # Create the semantic settings with the configuration
        if semantic_settings is None and semantic_configuration_name is not None:
            semantic_settings = SemanticSettings(
                configurations=[
                    SemanticConfiguration(
                        name=semantic_configuration_name,
                        prioritized_fields=PrioritizedFields(
                            prioritized_content_fields=[
                                SemanticField(field_name=FIELDS_CONTENT)
                            ],
                        ),
                    )
                ]
            )
        # Create the search index with the semantic settings and vector search
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_settings=semantic_settings,
            scoring_profiles=scoring_profiles,
            default_scoring_profile=default_scoring_profile,
        )
        index_client.create_index(index)
    # Create the search client
    return SearchClient(
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
        embedding_function: Callable,
        search_type: str = "hybrid",
        semantic_configuration_name: Optional[str] = None,
        semantic_query_language: str = "en-us",
        fields: Optional[List[SearchField]] = None,
        vector_search: Optional[VectorSearch] = None,
        semantic_settings: Optional[SemanticSettings] = None,
        scoring_profiles: Optional[List[ScoringProfile]] = None,
        default_scoring_profile: Optional[str] = None,
        **kwargs: Any,
    ):
        from azure.search.documents.indexes.models import (
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SimpleField,
        )

        """Initialize with necessary components."""
        # Initialize base class
        self.embedding_function = embedding_function
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
                vector_search_dimensions=len(embedding_function("Text")),
                vector_search_configuration="default",
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
            semantic_settings=semantic_settings,
            scoring_profiles=scoring_profiles,
            default_scoring_profile=default_scoring_profile,
            default_fields=default_fields,
            user_agent=user_agent,
        )
        self.search_type = search_type
        self.semantic_configuration_name = semantic_configuration_name
        self.semantic_query_language = semantic_query_language
        self.fields = fields if fields else default_fields

    @property
    def embeddings(self) -> Optional[Embeddings]:
        # TODO: Support embedding object directly
        return None

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
            # Encoding key for Azure Search valid characters
            key = base64.urlsafe_b64encode(bytes(key, "utf-8")).decode("ascii")
            metadata = metadatas[i] if metadatas else {}
            # Add data to index
            # Additional metadata to fields mapping
            doc = {
                "@search.action": "upload",
                FIELDS_ID: key,
                FIELDS_CONTENT: text,
                FIELDS_CONTENT_VECTOR: np.array(
                    self.embedding_function(text), dtype=np.float32
                ).tolist(),
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
                if not all([r.succeeded for r in response]):
                    raise Exception(response)
                # Reset data
                data = []

        # Considering case where data is an exact multiple of batch-size entries
        if len(data) == 0:
            return ids

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
        search_type = kwargs.get("search_type", self.search_type)
        if search_type == "similarity":
            docs = self.vector_search(query, k=k, **kwargs)
        elif search_type == "hybrid":
            docs = self.hybrid_search(query, k=k, **kwargs)
        elif search_type == "semantic_hybrid":
            docs = self.semantic_hybrid_search(query, k=k, **kwargs)
        else:
            raise ValueError(f"search_type of {search_type} not allowed.")
        return docs

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        score_threshold = kwargs.pop("score_threshold", None)
        result = self.vector_search_with_score(query, k=k, **kwargs)
        return (
            result
            if score_threshold is None
            else [r for r in result if r[1] >= score_threshold]
        )

    def vector_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """
        Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = self.vector_search_with_score(
            query, k=k, filters=kwargs.get("filters", None)
        )
        return [doc for doc, _ in docs_and_scores]

    def vector_search_with_score(
        self, query: str, k: int = 4, filters: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
        from azure.search.documents.models import Vector

        results = self.client.search(
            search_text="",
            vectors=[
                Vector(
                    value=np.array(
                        self.embedding_function(query), dtype=np.float32
                    ).tolist(),
                    k=k,
                    fields=FIELDS_CONTENT_VECTOR,
                )
            ],
            filter=filters,
        )
        # Convert results to Document objects
        docs = [
            (
                Document(
                    page_content=result.pop(FIELDS_CONTENT),
                    metadata=json.loads(result[FIELDS_METADATA])
                    if FIELDS_METADATA in result
                    else {
                        k: v for k, v in result.items() if k != FIELDS_CONTENT_VECTOR
                    },
                ),
                float(result["@search.score"]),
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
        from azure.search.documents.models import Vector

        results = self.client.search(
            search_text=query,
            vectors=[
                Vector(
                    value=np.array(
                        self.embedding_function(query), dtype=np.float32
                    ).tolist(),
                    k=k,
                    fields=FIELDS_CONTENT_VECTOR,
                )
            ],
            filter=filters,
            top=k,
        )
        # Convert results to Document objects
        docs = [
            (
                Document(
                    page_content=result.pop(FIELDS_CONTENT),
                    metadata=json.loads(result[FIELDS_METADATA])
                    if FIELDS_METADATA in result
                    else {
                        k: v for k, v in result.items() if k != FIELDS_CONTENT_VECTOR
                    },
                ),
                float(result["@search.score"]),
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
        from azure.search.documents.models import Vector

        results = self.client.search(
            search_text=query,
            vectors=[
                Vector(
                    value=np.array(
                        self.embedding_function(query), dtype=np.float32
                    ).tolist(),
                    k=50,
                    fields=FIELDS_CONTENT_VECTOR,
                )
            ],
            filter=filters,
            query_type="semantic",
            query_language=self.semantic_query_language,
            semantic_configuration_name=self.semantic_configuration_name,
            query_caption="extractive",
            query_answer="extractive",
            top=k,
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
                                json.loads(result["metadata"]).get("key"), ""
                            ),
                        },
                    },
                ),
                float(result["@search.score"]),
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
        azure_search_endpoint: str = "",
        azure_search_key: str = "",
        index_name: str = "langchain-index",
        **kwargs: Any,
    ) -> AzureSearch:
        # Creating a new Azure Search instance
        azure_search = cls(
            azure_search_endpoint,
            azure_search_key,
            index_name,
            embedding.embed_query,
        )
        azure_search.add_texts(texts, metadatas, **kwargs)
        return azure_search


class AzureSearchVectorStoreRetriever(BaseRetriever):
    """Retriever that uses `Azure Cognitive Search`."""

    vectorstore: AzureSearch
    """Azure Search instance used to find similar documents."""
    search_type: str = "hybrid"
    """Type of search to perform. Options are "similarity", "hybrid",
    "semantic_hybrid"."""
    k: int = 4
    """Number of documents to return."""

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

    def _get_relevant_documents(
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = self.vectorstore.vector_search(query, k=self.k, **kwargs)
        elif self.search_type == "hybrid":
            docs = self.vectorstore.hybrid_search(query, k=self.k, **kwargs)
        elif self.search_type == "semantic_hybrid":
            docs = self.vectorstore.semantic_hybrid_search(query, k=self.k, **kwargs)
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        raise NotImplementedError(
            "AzureSearchVectorStoreRetriever does not support async"
        )
