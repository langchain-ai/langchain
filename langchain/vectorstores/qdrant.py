"""Wrapper around Qdrant vector database."""
from __future__ import annotations

import uuid
import warnings
from itertools import islice
from operator import itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    from qdrant_client.conversions import common_types
    from qdrant_client.http import models as rest

    DictFilter = Dict[str, Union[str, int, bool, dict, list]]
    MetadataFilter = Union[DictFilter, common_types.Filter]


class Qdrant(VectorStore):
    """Wrapper around Qdrant vector database.

    To use you should have the ``qdrant-client`` package installed.

    Example:
        .. code-block:: python

            from qdrant_client import QdrantClient
            from langchain import Qdrant

            client = QdrantClient()
            collection_name = "MyCollection"
            qdrant = Qdrant(client, collection_name, embedding_function)
    """

    CONTENT_KEY = "page_content"
    METADATA_KEY = "metadata"

    def __init__(
        self,
        client: Any,
        collection_name: str,
        embeddings: Optional[Embeddings] = None,
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        embedding_function: Optional[Callable] = None,  # deprecated
    ):
        """Initialize with necessary components."""
        try:
            import qdrant_client
        except ImportError:
            raise ValueError(
                "Could not import qdrant-client python package. "
                "Please install it with `pip install qdrant-client`."
            )

        if not isinstance(client, qdrant_client.QdrantClient):
            raise ValueError(
                f"client should be an instance of qdrant_client.QdrantClient, "
                f"got {type(client)}"
            )

        if embeddings is None and embedding_function is None:
            raise ValueError(
                "`embeddings` value can't be None. Pass `Embeddings` instance."
            )

        if embeddings is not None and embedding_function is not None:
            raise ValueError(
                "Both `embeddings` and `embedding_function` are passed. "
                "Use `embeddings` only."
            )

        self.embeddings = embeddings
        self._embeddings_function = embedding_function
        self.client: qdrant_client.QdrantClient = client
        self.collection_name = collection_name
        self.content_payload_key = content_payload_key or self.CONTENT_KEY
        self.metadata_payload_key = metadata_payload_key or self.METADATA_KEY

        if embedding_function is not None:
            warnings.warn(
                "Using `embedding_function` is deprecated. "
                "Pass `Embeddings` instance to `embeddings` instead."
            )

        if not isinstance(embeddings, Embeddings):
            warnings.warn(
                "`embeddings` should be an instance of `Embeddings`."
                "Using `embeddings` as `embedding_function` which is deprecated"
            )
            self._embeddings_function = embeddings
            self.embeddings = None

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Sequence[str]] = None,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids:
                Optional list of ids to associate with the texts. Ids have to be
                uuid-like strings.
            batch_size:
                How many vectors upload per-request.
                Default: 64

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        from qdrant_client.http import models as rest

        added_ids = []
        texts_iterator = iter(texts)
        metadatas_iterator = iter(metadatas or [])
        ids_iterator = iter(ids or [uuid.uuid4().hex for _ in iter(texts)])
        while batch_texts := list(islice(texts_iterator, batch_size)):
            # Take the corresponding metadata and id for each text in a batch
            batch_metadatas = list(islice(metadatas_iterator, batch_size)) or None
            batch_ids = list(islice(ids_iterator, batch_size))

            self.client.upsert(
                collection_name=self.collection_name,
                points=rest.Batch.construct(
                    ids=batch_ids,
                    vectors=self._embed_texts(batch_texts),
                    payloads=self._build_payloads(
                        batch_texts,
                        batch_metadatas,
                        self.content_payload_key,
                        self.metadata_payload_key,
                    ),
                ),
            )

            added_ids.extend(batch_ids)

        return added_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[MetadataFilter] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        results = self.similarity_search_with_score(query, k, filter=filter)
        return list(map(itemgetter(0), results))

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[MetadataFilter] = None
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.

        Returns:
            List of documents most similar to the query text and cosine
            distance in float for each.
            Lower score represents more similarity.
        """

        if filter is not None and isinstance(filter, dict):
            warnings.warn(
                "Using dict as a `filter` is deprecated. Please use qdrant-client "
                "filters directly: "
                "https://qdrant.tech/documentation/concepts/filtering/",
                DeprecationWarning,
            )
            qdrant_filter = self._qdrant_filter_from_dict(filter)
        else:
            qdrant_filter = filter
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=self._embed_query(query),
            query_filter=qdrant_filter,
            with_payload=True,
            limit=k,
        )
        return [
            (
                self._document_from_scored_point(
                    result, self.content_payload_key, self.metadata_payload_key
                ),
                result.score,
            )
            for result in results
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                     Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """

        embedding = self._embed_query(query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            with_payload=True,
            with_vectors=True,
            limit=fetch_k,
        )
        embeddings = [result.vector for result in results]
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding), embeddings, k=k, lambda_mult=lambda_mult
        )
        return [
            self._document_from_scored_point(
                results[i], self.content_payload_key, self.metadata_payload_key
            )
            for i in mmr_selected
        ]

    @classmethod
    def from_texts(
        cls: Type[Qdrant],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Sequence[str]] = None,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[float] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        collection_name: Optional[str] = None,
        distance_func: str = "Cosine",
        content_payload_key: str = CONTENT_KEY,
        metadata_payload_key: str = METADATA_KEY,
        batch_size: int = 64,
        **kwargs: Any,
    ) -> Qdrant:
        """Construct Qdrant wrapper from a list of texts.

        Args:
            texts: A list of texts to be indexed in Qdrant.
            embedding: A subclass of `Embeddings`, responsible for text vectorization.
            metadatas:
                An optional list of metadata. If provided it has to be of the same
                length as a list of texts.
            ids:
                Optional list of ids to associate with the texts. Ids have to be
                uuid-like strings.
            location:
                If `:memory:` - use in-memory Qdrant instance.
                If `str` - use it as a `url` parameter.
                If `None` - fallback to relying on `host` and `port` parameters.
            url: either host or str of "Optional[scheme], host, Optional[port],
                Optional[prefix]". Default: `None`
            port: Port of the REST API interface. Default: 6333
            grpc_port: Port of the gRPC interface. Default: 6334
            prefer_grpc:
                If true - use gPRC interface whenever possible in custom methods.
                Default: False
            https: If true - use HTTPS(SSL) protocol. Default: None
            api_key: API key for authentication in Qdrant Cloud. Default: None
            prefix:
                If not None - add prefix to the REST URL path.
                Example: service/v1 will result in
                    http://localhost:6333/service/v1/{qdrant-endpoint} for REST API.
                Default: None
            timeout:
                Timeout for REST and gRPC API requests.
                Default: 5.0 seconds for REST and unlimited for gRPC
            host:
                Host name of Qdrant service. If url and host are None, set to
                'localhost'. Default: None
            path:
                Path in which the vectors will be stored while using local mode.
                Default: None
            collection_name:
                Name of the Qdrant collection to be used. If not provided,
                it will be created randomly. Default: None
            distance_func:
                Distance function. One of: "Cosine" / "Euclid" / "Dot".
                Default: "Cosine"
            content_payload_key:
                A payload key used to store the content of the document.
                Default: "page_content"
            metadata_payload_key:
                A payload key used to store the metadata of the document.
                Default: "metadata"
            batch_size:
                How many vectors upload per-request.
                Default: 64
            **kwargs:
                Additional arguments passed directly into REST client initialization

        This is a user-friendly interface that:
        1. Creates embeddings, one for each text
        2. Initializes the Qdrant database as an in-memory docstore by default
           (and overridable to a remote docstore)
        3. Adds the text embeddings to the Qdrant database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import Qdrant
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                qdrant = Qdrant.from_texts(texts, embeddings, "localhost")
        """
        try:
            import qdrant_client
        except ImportError:
            raise ValueError(
                "Could not import qdrant-client python package. "
                "Please install it with `pip install qdrant-client`."
            )

        from qdrant_client.http import models as rest

        # Just do a single quick embedding to get vector size
        partial_embeddings = embedding.embed_documents(texts[:1])
        vector_size = len(partial_embeddings[0])

        collection_name = collection_name or uuid.uuid4().hex
        distance_func = distance_func.upper()

        client = qdrant_client.QdrantClient(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            **kwargs,
        )

        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=rest.Distance[distance_func],
            ),
        )

        texts_iterator = iter(texts)
        metadatas_iterator = iter(metadatas or [])
        ids_iterator = iter(ids or [uuid.uuid4().hex for _ in iter(texts)])
        while batch_texts := list(islice(texts_iterator, batch_size)):
            # Take the corresponding metadata and id for each text in a batch
            batch_metadatas = list(islice(metadatas_iterator, batch_size)) or None
            batch_ids = list(islice(ids_iterator, batch_size))

            # Generate the embeddings for all the texts in a batch
            batch_embeddings = embedding.embed_documents(batch_texts)

            client.upsert(
                collection_name=collection_name,
                points=rest.Batch.construct(
                    ids=batch_ids,
                    vectors=batch_embeddings,
                    payloads=cls._build_payloads(
                        batch_texts,
                        batch_metadatas,
                        content_payload_key,
                        metadata_payload_key,
                    ),
                ),
            )

        return cls(
            client=client,
            collection_name=collection_name,
            embeddings=embedding,
            content_payload_key=content_payload_key,
            metadata_payload_key=metadata_payload_key,
        )

    @classmethod
    def _build_payloads(
        cls,
        texts: Iterable[str],
        metadatas: Optional[List[dict]],
        content_payload_key: str,
        metadata_payload_key: str,
    ) -> List[dict]:
        payloads = []
        for i, text in enumerate(texts):
            if text is None:
                raise ValueError(
                    "At least one of the texts is None. Please remove it before "
                    "calling .from_texts or .add_texts on Qdrant instance."
                )
            metadata = metadatas[i] if metadatas is not None else None
            payloads.append(
                {
                    content_payload_key: text,
                    metadata_payload_key: metadata,
                }
            )

        return payloads

    @classmethod
    def _document_from_scored_point(
        cls,
        scored_point: Any,
        content_payload_key: str,
        metadata_payload_key: str,
    ) -> Document:
        return Document(
            page_content=scored_point.payload.get(content_payload_key),
            metadata=scored_point.payload.get(metadata_payload_key) or {},
        )

    def _build_condition(self, key: str, value: Any) -> List[rest.FieldCondition]:
        from qdrant_client.http import models as rest

        out = []

        if isinstance(value, dict):
            for _key, value in value.items():
                out.extend(self._build_condition(f"{key}.{_key}", value))
        elif isinstance(value, list):
            for _value in value:
                if isinstance(_value, dict):
                    out.extend(self._build_condition(f"{key}[]", _value))
                else:
                    out.extend(self._build_condition(f"{key}", _value))
        else:
            out.append(
                rest.FieldCondition(
                    key=f"{self.metadata_payload_key}.{key}",
                    match=rest.MatchValue(value=value),
                )
            )

        return out

    def _qdrant_filter_from_dict(
        self, filter: Optional[DictFilter]
    ) -> Optional[rest.Filter]:
        from qdrant_client.http import models as rest

        if not filter:
            return None

        return rest.Filter(
            must=[
                condition
                for key, value in filter.items()
                for condition in self._build_condition(key, value)
            ]
        )

    def _embed_query(self, query: str) -> List[float]:
        """Embed query text.

        Used to provide backward compatibility with `embedding_function` argument.

        Args:
            query: Query text.

        Returns:
            List of floats representing the query embedding.
        """
        if self.embeddings is not None:
            embedding = self.embeddings.embed_query(query)
        else:
            if self._embeddings_function is not None:
                embedding = self._embeddings_function(query)
            else:
                raise ValueError("Neither of embeddings or embedding_function is set")
        return embedding.tolist() if hasattr(embedding, "tolist") else embedding

    def _embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed search texts.

        Used to provide backward compatibility with `embedding_function` argument.

        Args:
            texts: Iterable of texts to embed.

        Returns:
            List of floats representing the texts embedding.
        """
        if self.embeddings is not None:
            embeddings = self.embeddings.embed_documents(list(texts))
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()
        elif self._embeddings_function is not None:
            embeddings = []
            for text in texts:
                embedding = self._embeddings_function(text)
                if hasattr(embeddings, "tolist"):
                    embedding = embedding.tolist()
                embeddings.append(embedding)
        else:
            raise ValueError("Neither of embeddings or embedding_function is set")

        return embeddings
