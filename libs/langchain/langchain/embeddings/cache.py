"""Module contains code for a cache backed embedder.

The cache backed embedder is a wrapper around an embedder that caches
embeddings in a key-value store. The cache is used to avoid recomputing
embeddings for the same text.

The text is hashed and the hash is used as the key in the cache.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from functools import partial
from typing import Callable, List, Optional, Sequence, Union, cast

from langchain_core.embeddings import Embeddings
from langchain_core.stores import BaseStore, ByteStore
from langchain_core.utils.iter import batch_iterate

from langchain.storage.encoder_backed import EncoderBackedStore

NAMESPACE_UUID = uuid.UUID(int=1985)


def _hash_string_to_uuid(input_string: str) -> uuid.UUID:
    """Hash a string and returns the corresponding UUID."""
    hash_value = hashlib.sha1(input_string.encode("utf-8")).hexdigest()
    return uuid.uuid5(NAMESPACE_UUID, hash_value)


def _key_encoder(key: str, namespace: str) -> str:
    """Encode a key."""
    return namespace + str(_hash_string_to_uuid(key))


def _create_key_encoder(namespace: str) -> Callable[[str], str]:
    """Create an encoder for a key."""
    return partial(_key_encoder, namespace=namespace)


def _value_serializer(value: Sequence[float]) -> bytes:
    """Serialize a value."""
    return json.dumps(value).encode()


def _value_deserializer(serialized_value: bytes) -> List[float]:
    """Deserialize a value."""
    return cast(List[float], json.loads(serialized_value.decode()))


class CacheBackedEmbeddings(Embeddings):
    """Interface for caching results from embedding models.

    The interface allows works with any store that implements
    the abstract store interface accepting keys of type str and values of list of
    floats.

    If need be, the interface can be extended to accept other implementations
    of the value serializer and deserializer, as well as the key encoder.

    Examples:

        .. code-block: python

            from langchain.embeddings import CacheBackedEmbeddings
            from langchain.storage import LocalFileStore
            from langchain_community.embeddings import OpenAIEmbeddings

            store = LocalFileStore('./my_cache')

            underlying_embedder = OpenAIEmbeddings()
            embedder = CacheBackedEmbeddings.from_bytes_store(
                underlying_embedder, store, namespace=underlying_embedder.model
            )

            # Embedding is computed and cached
            embeddings = embedder.embed_documents(["hello", "goodbye"])

            # Embeddings are retrieved from the cache, no computation is done
            embeddings = embedder.embed_documents(["hello", "goodbye"])
    """

    def __init__(
        self,
        underlying_embeddings: Embeddings,
        document_embedding_store: BaseStore[str, List[float]],
        *,
        batch_size: Optional[int] = None,
    ) -> None:
        """Initialize the embedder.

        Args:
            underlying_embeddings: the embedder to use for computing embeddings.
            document_embedding_store: The store to use for caching document embeddings.
            batch_size: The number of documents to embed between store updates.
        """
        super().__init__()
        self.document_embedding_store = document_embedding_store
        self.underlying_embeddings = underlying_embeddings
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts.

        The method first checks the cache for the embeddings.
        If the embeddings are not found, the method uses the underlying embedder
        to embed the documents and stores the results in the cache.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of embeddings for the given texts.
        """
        vectors: List[Union[List[float], None]] = self.document_embedding_store.mget(
            texts
        )
        all_missing_indices: List[int] = [
            i for i, vector in enumerate(vectors) if vector is None
        ]

        for missing_indices in batch_iterate(self.batch_size, all_missing_indices):
            missing_texts = [texts[i] for i in missing_indices]
            missing_vectors = self.underlying_embeddings.embed_documents(missing_texts)
            self.document_embedding_store.mset(
                list(zip(missing_texts, missing_vectors))
            )
            for index, updated_vector in zip(missing_indices, missing_vectors):
                vectors[index] = updated_vector

        return cast(
            List[List[float]], vectors
        )  # Nones should have been resolved by now

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts.

        The method first checks the cache for the embeddings.
        If the embeddings are not found, the method uses the underlying embedder
        to embed the documents and stores the results in the cache.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of embeddings for the given texts.
        """
        vectors: List[
            Union[List[float], None]
        ] = await self.document_embedding_store.amget(texts)
        all_missing_indices: List[int] = [
            i for i, vector in enumerate(vectors) if vector is None
        ]

        # batch_iterate supports None batch_size which returns all elements at once
        # as a single batch.
        for missing_indices in batch_iterate(self.batch_size, all_missing_indices):
            missing_texts = [texts[i] for i in missing_indices]
            missing_vectors = await self.underlying_embeddings.aembed_documents(
                missing_texts
            )
            await self.document_embedding_store.amset(
                list(zip(missing_texts, missing_vectors))
            )
            for index, updated_vector in zip(missing_indices, missing_vectors):
                vectors[index] = updated_vector

        return cast(
            List[List[float]], vectors
        )  # Nones should have been resolved by now

    def embed_query(self, text: str) -> List[float]:
        """Embed query text.

        This method does not support caching at the moment.

        Support for caching queries is easily to implement, but might make
        sense to hold off to see the most common patterns.

        If the cache has an eviction policy, we may need to be a bit more careful
        about sharing the cache between documents and queries. Generally,
        one is OK evicting query caches, but document caches should be kept.

        Args:
            text: The text to embed.

        Returns:
            The embedding for the given text.
        """
        return self.underlying_embeddings.embed_query(text)

    async def aembed_query(self, text: str) -> List[float]:
        """Embed query text.

        This method does not support caching at the moment.

        Support for caching queries is easily to implement, but might make
        sense to hold off to see the most common patterns.

        If the cache has an eviction policy, we may need to be a bit more careful
        about sharing the cache between documents and queries. Generally,
        one is OK evicting query caches, but document caches should be kept.

        Args:
            text: The text to embed.

        Returns:
            The embedding for the given text.
        """
        return await self.underlying_embeddings.aembed_query(text)

    @classmethod
    def from_bytes_store(
        cls,
        underlying_embeddings: Embeddings,
        document_embedding_cache: ByteStore,
        *,
        namespace: str = "",
        batch_size: Optional[int] = None,
    ) -> CacheBackedEmbeddings:
        """On-ramp that adds the necessary serialization and encoding to the store.

        Args:
            underlying_embeddings: The embedder to use for embedding.
            document_embedding_cache: The cache to use for storing document embeddings.
            *,
            namespace: The namespace to use for document cache.
                       This namespace is used to avoid collisions with other caches.
                       For example, set it to the name of the embedding model used.
            batch_size: The number of documents to embed between store updates.
        """
        namespace = namespace
        key_encoder = _create_key_encoder(namespace)
        encoder_backed_store = EncoderBackedStore[str, List[float]](
            document_embedding_cache,
            key_encoder,
            _value_serializer,
            _value_deserializer,
        )
        return cls(underlying_embeddings, encoder_backed_store, batch_size=batch_size)
