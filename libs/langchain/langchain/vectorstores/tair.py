from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Iterable, List, Optional, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


def _uuid_key() -> str:
    return uuid.uuid4().hex


class Tair(VectorStore):
    """`Tair` vector store."""

    def __init__(
        self,
        embedding_function: Embeddings,
        url: str,
        index_name: str,
        content_key: str = "content",
        metadata_key: str = "metadata",
        search_params: Optional[dict] = None,
        **kwargs: Any,
    ):
        self.embedding_function = embedding_function
        self.index_name = index_name
        try:
            from tair import Tair as TairClient
        except ImportError:
            raise ImportError(
                "Could not import tair python package. "
                "Please install it with `pip install tair`."
            )
        try:
            # connect to tair from url
            client = TairClient.from_url(url, **kwargs)
        except ValueError as e:
            raise ValueError(f"Tair failed to connect: {e}")

        self.client = client
        self.content_key = content_key
        self.metadata_key = metadata_key
        self.search_params = search_params

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def create_index_if_not_exist(
        self,
        dim: int,
        distance_type: str,
        index_type: str,
        data_type: str,
        **kwargs: Any,
    ) -> bool:
        index = self.client.tvs_get_index(self.index_name)
        if index is not None:
            logger.info("Index already exists")
            return False
        self.client.tvs_create_index(
            self.index_name,
            dim,
            distance_type,
            index_type,
            data_type,
            **kwargs,
        )
        return True

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts data to an existing index."""
        ids = []
        keys = kwargs.get("keys", None)
        use_hybrid_search = False
        index = self.client.tvs_get_index(self.index_name)
        if index is not None and index.get("lexical_algorithm") == "bm25":
            use_hybrid_search = True
        # Write data to tair
        pipeline = self.client.pipeline(transaction=False)
        embeddings = self.embedding_function.embed_documents(list(texts))
        for i, text in enumerate(texts):
            # Use provided key otherwise use default key
            key = keys[i] if keys else _uuid_key()
            metadata = metadatas[i] if metadatas else {}
            if use_hybrid_search:
                # tair use TEXT attr hybrid search
                pipeline.tvs_hset(
                    self.index_name,
                    key,
                    embeddings[i],
                    False,
                    **{
                        "TEXT": text,
                        self.content_key: text,
                        self.metadata_key: json.dumps(metadata),
                    },
                )
            else:
                pipeline.tvs_hset(
                    self.index_name,
                    key,
                    embeddings[i],
                    False,
                    **{
                        self.content_key: text,
                        self.metadata_key: json.dumps(metadata),
                    },
                )
            ids.append(key)
        pipeline.execute()
        return ids

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
        # Creates embedding vector from user query
        embedding = self.embedding_function.embed_query(query)

        keys_and_scores = self.client.tvs_knnsearch(
            self.index_name, k, embedding, False, None, **kwargs
        )

        pipeline = self.client.pipeline(transaction=False)
        for key, _ in keys_and_scores:
            pipeline.tvs_hmget(
                self.index_name, key, self.metadata_key, self.content_key
            )
        docs = pipeline.execute()

        return [
            Document(
                page_content=d[1],
                metadata=json.loads(d[0]),
            )
            for d in docs
        ]

    @classmethod
    def from_texts(
        cls: Type[Tair],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        index_name: str = "langchain",
        content_key: str = "content",
        metadata_key: str = "metadata",
        **kwargs: Any,
    ) -> Tair:
        try:
            from tair import tairvector
        except ImportError:
            raise ValueError(
                "Could not import tair python package. "
                "Please install it with `pip install tair`."
            )
        url = get_from_dict_or_env(kwargs, "tair_url", "TAIR_URL")
        if "tair_url" in kwargs:
            kwargs.pop("tair_url")

        distance_type = tairvector.DistanceMetric.InnerProduct
        if "distance_type" in kwargs:
            distance_type = kwargs.pop("distance_type")
        index_type = tairvector.IndexType.HNSW
        if "index_type" in kwargs:
            index_type = kwargs.pop("index_type")
        data_type = tairvector.DataType.Float32
        if "data_type" in kwargs:
            data_type = kwargs.pop("data_type")
        index_params = {}
        if "index_params" in kwargs:
            index_params = kwargs.pop("index_params")
        search_params = {}
        if "search_params" in kwargs:
            search_params = kwargs.pop("search_params")

        keys = None
        if "keys" in kwargs:
            keys = kwargs.pop("keys")
        try:
            tair_vector_store = cls(
                embedding,
                url,
                index_name,
                content_key=content_key,
                metadata_key=metadata_key,
                search_params=search_params,
                **kwargs,
            )
        except ValueError as e:
            raise ValueError(f"tair failed to connect: {e}")

        # Create embeddings for documents
        embeddings = embedding.embed_documents(texts)

        tair_vector_store.create_index_if_not_exist(
            len(embeddings[0]),
            distance_type,
            index_type,
            data_type,
            **index_params,
        )

        tair_vector_store.add_texts(texts, metadatas, keys=keys)
        return tair_vector_store

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        index_name: str = "langchain",
        content_key: str = "content",
        metadata_key: str = "metadata",
        **kwargs: Any,
    ) -> Tair:
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        return cls.from_texts(
            texts, embedding, metadatas, index_name, content_key, metadata_key, **kwargs
        )

    @staticmethod
    def drop_index(
        index_name: str = "langchain",
        **kwargs: Any,
    ) -> bool:
        """
        Drop an existing index.

        Args:
            index_name (str): Name of the index to drop.

        Returns:
            bool: True if the index is dropped successfully.
        """
        try:
            from tair import Tair as TairClient
        except ImportError:
            raise ValueError(
                "Could not import tair python package. "
                "Please install it with `pip install tair`."
            )
        url = get_from_dict_or_env(kwargs, "tair_url", "TAIR_URL")

        try:
            if "tair_url" in kwargs:
                kwargs.pop("tair_url")
            client = TairClient.from_url(url=url, **kwargs)
        except ValueError as e:
            raise ValueError(f"Tair connection error: {e}")
        # delete index
        ret = client.tvs_del_index(index_name)
        if ret == 0:
            # index not exist
            logger.info("Index does not exist")
            return False
        return True

    @classmethod
    def from_existing_index(
        cls,
        embedding: Embeddings,
        index_name: str = "langchain",
        content_key: str = "content",
        metadata_key: str = "metadata",
        **kwargs: Any,
    ) -> Tair:
        """Connect to an existing Tair index."""
        url = get_from_dict_or_env(kwargs, "tair_url", "TAIR_URL")

        search_params = {}
        if "search_params" in kwargs:
            search_params = kwargs.pop("search_params")

        return cls(
            embedding,
            url,
            index_name,
            content_key=content_key,
            metadata_key=metadata_key,
            search_params=search_params,
            **kwargs,
        )
