"""Coherence vector store."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Optional,
    cast,
)

from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

import jsonpickle  # type: ignore[import-untyped]
from coherence import (  # type: ignore[import-untyped]
    Extractors,
    Filters,
    NamedCache,
)
from coherence.ai import (  # type: ignore[import-untyped]
    CosineDistance,
    DistanceAlgorithm,
    FloatVector,
    HnswIndex,
    QueryResult,
    SimilaritySearch,
    Vector,
    Vectors,
)
from coherence.extractor import (  # type: ignore[import-untyped]
    ValueExtractor,
)
from coherence.filter import (  # type: ignore[import-untyped]
    Filter,
)
from coherence.serialization import (  # type: ignore[import-untyped]
    JSONSerializer,
    SerializerRegistry,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class CoherenceVectorStore(VectorStore):
    """Coherence VectorStore implementation.

    Uses Coherence NamedCache, for similarity search.

    Setup:
        Install ``langchain-core``.

        .. code-block:: bash

            pip install -U langchain-core

    Add Documents and retrieve them:
        .. code-block:: python

            from langchain_core.documents import Document
            from langchain_core.embeddings import Embeddings
            from langchain_huggingface.embeddings import HuggingFaceEmbeddings

            from coherence import NamedMap, Session
            from langchain_core.vectorstores.coherence_store import CoherenceVectorStore

            session: Session = await Session.create()
            try:
                named_map: NamedMap[str, Document] = await session.get_map("my-map")
                embedding :Embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-l6-v2")
                # this embedding generates vectors of dimension 384
                cvs :CoherenceVectorStore = await CoherenceVectorStore.create(
                                                    named_map,embedding,384)
                d1 :Document = Document(id="1", page_content="apple")
                d2 :Document = Document(id="2", page_content="orange")
                documents = [d1, d2]
                await cvs.aadd_documents(documents)

                ids = [doc.id for doc in documents]
                l = await cvs.aget_by_ids(ids)
                assert len(l) == len(ids)
                print("====")
                for e in l:
                    print(e)
            finally:
                await session.close()

    Delete Documents:
        .. code-block:: python

            from langchain_core.documents import Document
            from langchain_core.embeddings import Embeddings
            from langchain_huggingface.embeddings import HuggingFaceEmbeddings

            from coherence import NamedMap, Session
            from langchain_core.vectorstores.coherence_store import CoherenceVectorStore

            session: Session = await Session.create()
            try:
                named_map: NamedMap[str, Document] = await session.get_map("my-map")
                embedding :Embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-l6-v2")
                # this embedding generates vectors of dimension 384
                cvs :CoherenceVectorStore = await CoherenceVectorStore.create(
                                                    named_map,embedding,384)
                d1 :Document = Document(id="1", page_content="apple")
                d2 :Document = Document(id="2", page_content="orange")
                documents = [d1, d2]
                await cvs.aadd_documents(documents)

                ids = [doc.id for doc in documents]
                await cvs.adelete(ids)
            finally:
                await session.close()

    Similarity Search:
        .. code-block:: python

            from langchain_core.documents import Document
            from langchain_core.embeddings import Embeddings
            from langchain_huggingface.embeddings import HuggingFaceEmbeddings

            from coherence import NamedMap, Session
            from langchain_core.vectorstores.coherence_store import CoherenceVectorStore

            def test_data():
                d1 :Document = Document(id="1", page_content="apple")
                d2 :Document = Document(id="2", page_content="orange")
                d3 :Document = Document(id="3", page_content="tiger")
                d4 :Document = Document(id="4", page_content="cat")
                d5 :Document = Document(id="5", page_content="dog")
                d6 :Document = Document(id="6", page_content="fox")
                d7 :Document = Document(id="7", page_content="pear")
                d8 :Document = Document(id="8", page_content="banana")
                d9 :Document = Document(id="9", page_content="plum")
                d10 :Document = Document(id="10", page_content="lion")

                documents = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10]
                return documents

            async def test_asimilarity_search():
                documents = test_data()
                session: Session = await Session.create()
                try:
                    named_map: NamedMap[str, Document] = await session.get_map("my-map")
                    embedding :Embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-l6-v2")
                    # this embedding generates vectors of dimension 384
                    cvs :CoherenceVectorStore = await CoherenceVectorStore.create(
                                                        named_map,embedding,384)
                    await cvs.aadd_documents(documents)
                    ids = [doc.id for doc in documents]
                    l = await cvs.aget_by_ids(ids)
                    assert len(l) == 10

                    result = await cvs.asimilarity_search("fruit")
                    assert len(result) == 4
                    print("====")
                    for e in result:
                        print(e)
                finally:
                    await session.close()

    Similarity Search by vector :
        .. code-block:: python

            from langchain_core.documents import Document
            from langchain_core.embeddings import Embeddings
            from langchain_huggingface.embeddings import HuggingFaceEmbeddings

            from coherence import NamedMap, Session
            from langchain_core.vectorstores.coherence_store import CoherenceVectorStore

            def test_data():
                d1 :Document = Document(id="1", page_content="apple")
                d2 :Document = Document(id="2", page_content="orange")
                d3 :Document = Document(id="3", page_content="tiger")
                d4 :Document = Document(id="4", page_content="cat")
                d5 :Document = Document(id="5", page_content="dog")
                d6 :Document = Document(id="6", page_content="fox")
                d7 :Document = Document(id="7", page_content="pear")
                d8 :Document = Document(id="8", page_content="banana")
                d9 :Document = Document(id="9", page_content="plum")
                d10 :Document = Document(id="10", page_content="lion")

                documents = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10]
                return documents

            async def test_asimilarity_search_by_vector():
                documents = test_data()
                session: Session = await Session.create()
                try:
                    named_map: NamedMap[str, Document] = await session.get_map("my-map")
                    embedding :Embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-l6-v2")
                    # this embedding generates vectors of dimension 384
                    cvs :CoherenceVectorStore = await CoherenceVectorStore.create(
                                                        named_map,embedding,384)
                    await cvs.aadd_documents(documents)
                    ids = [doc.id for doc in documents]
                    l = await cvs.aget_by_ids(ids)
                    assert len(l) == 10

                    vector = cvs.embeddings.embed_query("fruit")
                    result = await cvs.asimilarity_search_by_vector(vector)
                    assert len(result) == 4
                    print("====")
                    for e in result:
                        print(e)
                finally:
                    await session.close()

    """

    VECTOR_FIELD: Final[str] = "__dict__.metadata.vector"
    """The name of the field containing the vector embeddings."""

    VECTOR_EXTRACTOR: Final[ValueExtractor] = Extractors.extract(VECTOR_FIELD)
    """The ValueExtractor to extract the embeddings vector."""

    def __init__(self, coherence_cache: NamedCache, embedding: Embeddings):
        """Initialize with Coherence cache and embedding function.

        Args:
            coherence_cache: Coherence NamedCache to use
            embedding: embedding function to use.
        """
        self.cache = coherence_cache
        self.embedding = embedding

    @staticmethod
    async def create(coherence_cache: NamedCache, embedding: Embeddings,
        dimensions: int
    ) -> CoherenceVectorStore:
        """Create an instance of CoherenceVectorStore.

        Args:
            coherence_cache: Coherence NamedCache to use
            embedding: embedding function to use.
            dimensions: size of the vector created by the embedding function
        """
        coh_store: CoherenceVectorStore = CoherenceVectorStore(coherence_cache,
                                                               embedding)
        await coherence_cache.add_index(HnswIndex(
            CoherenceVectorStore.VECTOR_EXTRACTOR, dimensions))
        return coh_store

    @property
    @override
    def embeddings(self) -> Embeddings:
        return self.embedding

    @override
    def add_documents(
        self, documents: list[Document], ids: Optional[list[str]] = None, **kwargs: Any
    ) -> list[str]:
        raise NotImplementedError

    @override
    async def aadd_documents(
        self, documents: list[Document], ids: Optional[list[str]] = None, **kwargs: Any
    ) -> list[str]:
        """Add documents to the store."""
        texts = [doc.page_content for doc in documents]
        vectors = await self.embedding.aembed_documents(texts)

        # Apply normalization and wrap in FloatVector
        float_vectors = [FloatVector(Vectors.normalize(vector)) for vector in vectors]

        if ids and len(ids) != len(texts):
            msg = (
                f"ids must be the same length as texts. "
                f"Got {len(ids)} ids and {len(texts)} texts."
            )
            raise ValueError(msg)

        id_iterator: Iterator[Optional[str]] = (
            iter(ids) if ids else iter(doc.id for doc in documents)
        )
        ids_: list[str] = []

        doc_map: dict[str, Document] = {}
        for doc, vector in zip(documents, float_vectors):
            doc_id = next(id_iterator)
            doc_id_ = doc_id or str(uuid.uuid4())
            ids_.append(doc_id_)
            doc.metadata["vector"] = vector
            doc_map[doc_id_] = doc

        await self.cache.put_all(doc_map)

        return ids_

    @override
    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        raise NotImplementedError

    @override
    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their ids.

        Args:
            ids: The ids of the documents to get.

        Returns:
            A list of Document objects.
        """
        return [e.value async for e in await self.cache.get_all(set(ids))]

    @override
    async def adelete(self, ids: Optional[Sequence[str]] = None, **kwargs: Any) -> None:
        """Async delete by Documeny ID or other criteria.

        Args:
            ids: List of ids to delete. If None, delete all. Default is None.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        if ids is None:
            await self.cache.clear()
        else:
            # Efficient parallel delete
            await asyncio.gather(*(self.cache.remove(i) for i in ids))

    def _parse_coherence_kwargs(self, **kwargs: Any
    ) -> tuple[DistanceAlgorithm, Filter, bool]:
        allowed_keys = {"algorithm", "filter", "brute_force"}
        extra_keys = set(kwargs) - allowed_keys
        if extra_keys:
            # Silently ignore or log if needed
            for key in extra_keys:
                kwargs.pop(key)

        algorithm: DistanceAlgorithm = kwargs.get("algorithm", CosineDistance())
        filter_: Filter = kwargs.get("filter", Filters.always())
        brute_force: bool = kwargs.get("brute_force", False)

        return (algorithm, filter_, brute_force)

    @override
    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Async method return docs most similar to query.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Optional arguments:
                - algorithm: DistanceAlgorithm to use.(default CosineDistance)
                  https://oracle.github.io/coherence-py-client/api_reference/ai.html#cosinedistance
                - filter: filter to use to limit the set of entries to search.
                  (default Filters.always())
                  https://oracle.github.io/coherence-py-client/api_reference/filter.html
                - brute_force: Force brute force search, ignoring any available indexes.
                  (default False)
                  https://oracle.github.io/coherence-py-client/api_reference/ai.html#similaritysearch

        Returns:
            List of Documents most similar to the query.
        """
        algorithm, filter_, brute_force = self._parse_coherence_kwargs(**kwargs)

        query_vector = self.embedding.embed_query(query)
        float_query_vector = FloatVector(Vectors.normalize(query_vector))

        search: SimilaritySearch = SimilaritySearch(
            CoherenceVectorStore.VECTOR_EXTRACTOR,
            float_query_vector,
            k,
            algorithm=algorithm,
            filter=filter_,
            brute_force=brute_force,
        )
        query_results = await self.cache.aggregate(search)

        return [e.value for e in query_results]

    @override
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        raise NotImplementedError

    @override
    async def asimilarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Async method return docs most similar to passed embedding vector.

        Args:
            embedding: Input vector.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Optional arguments:
                - algorithm: DistanceAlgorithm to use.(default CosineDistance)
                  https://oracle.github.io/coherence-py-client/api_reference/ai.html#cosinedistance
                - filter: filter to use to limit the set of entries to search.
                  (default Filters.always())
                  https://oracle.github.io/coherence-py-client/api_reference/filter.html
                - brute_force: Force brute force search, ignoring any available indexes.
                  (default False)
                  https://oracle.github.io/coherence-py-client/api_reference/ai.html#similaritysearch

        Returns:
            List of Documents most similar to the query.
        """
        algorithm, filter_, brute_force = self._parse_coherence_kwargs(**kwargs)
        float_query_vector = FloatVector(Vectors.normalize(embedding))

        search: SimilaritySearch = SimilaritySearch(
            CoherenceVectorStore.VECTOR_EXTRACTOR,
            float_query_vector,
            k,
            algorithm=algorithm,
            filter=filter_,
            brute_force=brute_force,
        )
        query_results = await self.cache.aggregate(search, filter=Filters.always())

        return [e.value for e in query_results]

    @override
    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        raise NotImplementedError

    @override
    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Async method return list tuple(Document, score) most similar to query.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: Optional arguments:
                - algorithm: DistanceAlgorithm to use.(default CosineDistance)
                  https://oracle.github.io/coherence-py-client/api_reference/ai.html#cosinedistance
                - filter: filter to use to limit the set of entries to search.
                  (default Filters.always())
                  https://oracle.github.io/coherence-py-client/api_reference/filter.html
                - brute_force: Force brute force search, ignoring any available indexes.
                  (default False)
                  https://oracle.github.io/coherence-py-client/api_reference/ai.html#similaritysearch

        Returns:
            List of tuple(Document, score) most similar to the query.
        """
        algorithm, filter_, brute_force = self._parse_coherence_kwargs(**kwargs)
        query_vector = self.embedding.embed_query(query)
        float_query_vector = FloatVector(Vectors.normalize(query_vector))

        search: SimilaritySearch = SimilaritySearch(
            CoherenceVectorStore.VECTOR_EXTRACTOR,
            float_query_vector,
            k,
            algorithm=algorithm,
            filter=filter_,
            brute_force=brute_force,
        )
        query_results: list[QueryResult] = await self.cache.aggregate(
            search, filter=Filters.always()
        )

        return [(e.value, e.distance) for e in query_results]

    @override
    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        raise NotImplementedError

    @classmethod
    @override
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> CoherenceVectorStore:
        raise NotImplementedError

    @classmethod
    @override
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> CoherenceVectorStore:
        raise NotImplementedError


@jsonpickle.handlers.register(Document)
class _LangChainDocumentHandler(jsonpickle.handlers.BaseHandler):  # type: ignore[misc]
    def flatten(self, obj: object, data: dict[str, Any]) -> dict[str, Any]:
        """Flatten object to a dictionary for handler to use."""
        ser = SerializerRegistry.serializer(JSONSerializer.SER_FORMAT)
        json_ser = cast("JSONSerializer", ser)
        o = cast("Document", obj)
        vector = o.metadata["vector"]
        if vector is not None and isinstance(vector, Vector):
            s = json_ser.serialize(vector)
            d = json.loads(s[1:])
            o.metadata["vector"] = json_ser.flatten_to_dict(d)

        data["__dict__"] = obj.__dict__
        return data

    def restore(self, obj: dict[str, Any]) -> Document:
        """Convert dictionary to an object for handler to use."""
        ser = SerializerRegistry.serializer(JSONSerializer.SER_FORMAT)
        json_ser = cast("JSONSerializer", ser)
        d = Document(page_content="")
        d.__dict__ = obj["__dict__"]
        vector = d.metadata["vector"]
        if vector is not None and isinstance(vector, dict):
            d.metadata["vector"] = json_ser.restore_to_object(vector)
        return d
