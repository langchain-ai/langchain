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
import warnings
from collections.abc import Sequence
from typing import Callable, Literal, Optional, Union, cast

from langchain_core.embeddings import Embeddings
from langchain_core.stores import BaseStore, ByteStore
from langchain_core.utils.iter import batch_iterate

from langchain.storage.encoder_backed import EncoderBackedStore

NAMESPACE_UUID = uuid.UUID(int=1985)


def _sha1_hash_to_uuid(text: str) -> uuid.UUID:
    """Return a UUID derived from *text* using SHA-1 (deterministic).

    Deterministic and fast, **but not collision-resistant**.

    A malicious attacker could try to create two different texts that hash to the same
    UUID. This may not necessarily be an issue in the context of caching embeddings,
    but new applications should swap this out for a stronger hash function like
    xxHash, BLAKE2 or SHA-256, which are collision-resistant.
    """
    sha1_hex = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()
    # Embed the hex string in `uuid5` to obtain a valid UUID.
    return uuid.uuid5(NAMESPACE_UUID, sha1_hex)


def _make_default_key_encoder(namespace: str, algorithm: str) -> Callable[[str], str]:
    """Create a default key encoder function.

    Args:
        namespace: Prefix that segregates keys from different embedding models.
        algorithm:
           * ``'sha1'`` - fast but not collision-resistant
           * ``'blake2b'`` - cryptographically strong, faster than SHA-1
           * ``'sha256'`` - cryptographically strong, slower than SHA-1
           * ``'sha512'`` - cryptographically strong, slower than SHA-1

    Returns:
        A function that encodes a key using the specified algorithm.
    """
    if algorithm == "sha1":
        _warn_about_sha1_encoder()

    def _key_encoder(key: str) -> str:
        """Encode a key using the specified algorithm."""
        if algorithm == "sha1":
            return f"{namespace}{_sha1_hash_to_uuid(key)}"
        if algorithm == "blake2b":
            return f"{namespace}{hashlib.blake2b(key.encode('utf-8')).hexdigest()}"
        if algorithm == "sha256":
            return f"{namespace}{hashlib.sha256(key.encode('utf-8')).hexdigest()}"
        if algorithm == "sha512":
            return f"{namespace}{hashlib.sha512(key.encode('utf-8')).hexdigest()}"
        msg = f"Unsupported algorithm: {algorithm}"
        raise ValueError(msg)

    return _key_encoder


def _value_serializer(value: Sequence[float]) -> bytes:
    """Serialize a value."""
    return json.dumps(value).encode()


def _value_deserializer(serialized_value: bytes) -> list[float]:
    """Deserialize a value."""
    return cast(list[float], json.loads(serialized_value.decode()))


# The warning is global; track emission, so it appears only once.
_warned_about_sha1: bool = False


def _warn_about_sha1_encoder() -> None:
    """Emit a one-time warning about SHA-1 collision weaknesses."""
    global _warned_about_sha1  # noqa: PLW0603
    if not _warned_about_sha1:
        warnings.warn(
            "Using default key encoder: SHA-1 is *not* collision-resistant. "
            "While acceptable for most cache scenarios, a motivated attacker "
            "can craft two different payloads that map to the same cache key. "
            "If that risk matters in your environment, supply a stronger "
            "encoder (e.g. SHA-256 or BLAKE2) via the `key_encoder` argument. "
            "If you change the key encoder, consider also creating a new cache, "
            "to avoid (the potential for) collisions with existing keys.",
            category=UserWarning,
            stacklevel=2,
        )
        _warned_about_sha1 = True


class CacheBackedEmbeddings(Embeddings):
    """Interface for caching results from embedding models.

    The interface allows works with any store that implements
    the abstract store interface accepting keys of type str and values of list of
    floats.

    If need be, the interface can be extended to accept other implementations
    of the value serializer and deserializer, as well as the key encoder.

    Note that by default only document embeddings are cached. To cache query
    embeddings too, pass in a query_embedding_store to constructor.

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
        document_embedding_store: BaseStore[str, list[float]],
        *,
        batch_size: Optional[int] = None,
        query_embedding_store: Optional[BaseStore[str, list[float]]] = None,
    ) -> None:
        """Initialize the embedder.

        Args:
            underlying_embeddings: the embedder to use for computing embeddings.
            document_embedding_store: The store to use for caching document embeddings.
            batch_size: The number of documents to embed between store updates.
            query_embedding_store: The store to use for caching query embeddings.
                If ``None``, query embeddings are not cached.
        """
        super().__init__()
        self.document_embedding_store = document_embedding_store
        self.query_embedding_store = query_embedding_store
        self.underlying_embeddings = underlying_embeddings
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts.

        The method first checks the cache for the embeddings.
        If the embeddings are not found, the method uses the underlying embedder
        to embed the documents and stores the results in the cache.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of embeddings for the given texts.
        """
        vectors: list[Union[list[float], None]] = self.document_embedding_store.mget(
            texts,
        )
        all_missing_indices: list[int] = [
            i for i, vector in enumerate(vectors) if vector is None
        ]

        for missing_indices in batch_iterate(self.batch_size, all_missing_indices):
            missing_texts = [texts[i] for i in missing_indices]
            missing_vectors = self.underlying_embeddings.embed_documents(missing_texts)
            self.document_embedding_store.mset(
                list(zip(missing_texts, missing_vectors)),
            )
            for index, updated_vector in zip(missing_indices, missing_vectors):
                vectors[index] = updated_vector

        return cast(
            list[list[float]],
            vectors,
        )  # Nones should have been resolved by now

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts.

        The method first checks the cache for the embeddings.
        If the embeddings are not found, the method uses the underlying embedder
        to embed the documents and stores the results in the cache.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of embeddings for the given texts.
        """
        vectors: list[
            Union[list[float], None]
        ] = await self.document_embedding_store.amget(texts)
        all_missing_indices: list[int] = [
            i for i, vector in enumerate(vectors) if vector is None
        ]

        # batch_iterate supports None batch_size which returns all elements at once
        # as a single batch.
        for missing_indices in batch_iterate(self.batch_size, all_missing_indices):
            missing_texts = [texts[i] for i in missing_indices]
            missing_vectors = await self.underlying_embeddings.aembed_documents(
                missing_texts,
            )
            await self.document_embedding_store.amset(
                list(zip(missing_texts, missing_vectors)),
            )
            for index, updated_vector in zip(missing_indices, missing_vectors):
                vectors[index] = updated_vector

        return cast(
            list[list[float]],
            vectors,
        )  # Nones should have been resolved by now

    def embed_query(self, text: str) -> list[float]:
        """Embed query text.

        By default, this method does not cache queries. To enable caching, set the
        ``cache_query`` parameter to ``True`` when initializing the embedder.

        Args:
            text: The text to embed.

        Returns:
            The embedding for the given text.
        """
        if not self.query_embedding_store:
            return self.underlying_embeddings.embed_query(text)

        (cached,) = self.query_embedding_store.mget([text])
        if cached is not None:
            return cached

        vector = self.underlying_embeddings.embed_query(text)
        self.query_embedding_store.mset([(text, vector)])
        return vector

    async def aembed_query(self, text: str) -> list[float]:
        """Embed query text.

        By default, this method does not cache queries. To enable caching, set the
        ``cache_query`` parameter to ``True`` when initializing the embedder.

        Args:
            text: The text to embed.

        Returns:
            The embedding for the given text.
        """
        if not self.query_embedding_store:
            return await self.underlying_embeddings.aembed_query(text)

        (cached,) = await self.query_embedding_store.amget([text])
        if cached is not None:
            return cached

        vector = await self.underlying_embeddings.aembed_query(text)
        await self.query_embedding_store.amset([(text, vector)])
        return vector

    @classmethod
    def from_bytes_store(
        cls,
        underlying_embeddings: Embeddings,
        document_embedding_cache: ByteStore,
        *,
        namespace: str = "",
        batch_size: Optional[int] = None,
        query_embedding_cache: Union[bool, ByteStore] = False,
        key_encoder: Union[
            Callable[[str], str],
            Literal["sha1", "blake2b", "sha256", "sha512"],
        ] = "sha1",
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
            query_embedding_cache: The cache to use for storing query embeddings.
                True to use the same cache as document embeddings.
                False to not cache query embeddings.
            key_encoder: Optional callable to encode keys. If not provided,
                a default encoder using SHA-1 will be used. SHA-1 is not
                collision-resistant, and a motivated attacker could craft two
                different texts that hash to the same cache key.

                New applications should use one of the alternative encoders
                or provide a custom and strong key encoder function to avoid this risk.

                If you change a key encoder in an existing cache, consider
                just creating a new cache, to avoid (the potential for)
                collisions with existing keys or having duplicate keys
                for the same text in the cache.

        Returns:
            An instance of CacheBackedEmbeddings that uses the provided cache.
        """
        if isinstance(key_encoder, str):
            key_encoder = _make_default_key_encoder(namespace, key_encoder)
        elif callable(key_encoder):
            # If a custom key encoder is provided, it should not be used with a
            # namespace.
            # A user can handle namespacing in directly their custom key encoder.
            if namespace:
                msg = (
                    "Do not supply `namespace` when using a custom key_encoder; "
                    "add any prefixing inside the encoder itself."
                )
                raise ValueError(msg)
        else:
            msg = (
                "key_encoder must be either 'blake2b', 'sha1', 'sha256', 'sha512' "
                "or a callable that encodes keys."
            )
            raise ValueError(msg)  # noqa: TRY004

        document_embedding_store = EncoderBackedStore[str, list[float]](
            document_embedding_cache,
            key_encoder,
            _value_serializer,
            _value_deserializer,
        )
        if query_embedding_cache is True:
            query_embedding_store = document_embedding_store
        elif query_embedding_cache is False:
            query_embedding_store = None
        else:
            query_embedding_store = EncoderBackedStore[str, list[float]](
                query_embedding_cache,
                key_encoder,
                _value_serializer,
                _value_deserializer,
            )

        return cls(
            underlying_embeddings,
            document_embedding_store,
            batch_size=batch_size,
            query_embedding_store=query_embedding_store,
        )
