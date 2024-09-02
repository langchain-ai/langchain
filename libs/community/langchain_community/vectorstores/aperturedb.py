# System imports
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

# Third-party imports
import numpy as np

# Local imports
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing.base import UpsertResponse
from langchain_core.vectorstores import VectorStore
from typing_extensions import override

# Configure some defaults
ENGINE = "HNSW"
METRIC = "CS"
DESCRIPTOR_SET = "langchain"
BATCHSIZE = 1000
PROPERTY_PREFIX = "lc_"  # Prefix for properties that are in the client metadata
TEXT_PROPERTY = "text"  # Property name for the text
UNIQUEID_PROPERTY = "uniqueid"  # Property name for the unique id


class ApertureDB(VectorStore):
    @override
    def __init__(
        self,
        embeddings: Embeddings,
        descriptor_set: str = DESCRIPTOR_SET,
        dimensions: Optional[int] = None,
        engine: Optional[str] = None,
        metric: Optional[str] = None,
        log_level: int = logging.WARN,
        properties: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """Create a vectorstore backed by ApertureDB

        A single ApertureDB instance can support many vectorstores,
        distinguished by 'descriptor_set' name.  The descriptor set is created
        if it does not exist.  Different descriptor sets can use different
        engines and metrics, be supplied by different embedding models, and have
        different dimensions.

        See ApertureDB documentation on `AddDescriptorSet`
        https://docs.aperturedata.io/query_language/Reference/descriptor_commands/desc_set_commands/AddDescriptorSet
        for more information on the engine and metric options.

        Args:
            embeddings (Embeddings): Embeddings object
            descriptor_set (str, optional): Descriptor set name. Defaults to
                "langchain".
            dimensions (Optional[int], optional): Number of dimensions of the
                embeddings. Defaults to None.
            engine (str, optional): Engine to use. Defaults to "HNSW" for new
                descriptorsets.
            metric (str, optional): Metric to use. Defaults to "CS" for new
                descriptorsets.
            log_level (int, optional): Logging level. Defaults to logging.WARN.
        """
        # ApertureDB imports
        try:
            from aperturedb.Utils import Utils, create_connector
        except ImportError:
            raise ImportError(
                "ApertureDB is not installed. Please install it using "
                "'pip install aperturedb'"
            )

        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.descriptor_set = descriptor_set

        self.embedding_function = embeddings
        self.dimensions = dimensions
        self.engine = engine
        self.metric = metric
        self.properties = properties
        if embeddings is None:
            self.logger.fatal("No embedding function provided.")
            raise ValueError("No embedding function provided.")

        try:
            from aperturedb.Utils import Utils, create_connector
        except ImportError:
            self.logger.exception(
                "ApertureDB is not installed. Please install it using "
                "'pip install aperturedb'"
            )
            raise

        self.connection = create_connector()
        self.utils = Utils(self.connection)
        try:
            self.utils.status()
        except Exception:
            self.logger.exception("Failed to connect to ApertureDB")
            raise

        self._find_or_add_descriptor_set()

    def _find_or_add_descriptor_set(self) -> None:
        descriptor_set = self.descriptor_set
        """Checks if the descriptor set exists, if not, creates it"""
        find_ds_query = [
            {
                "FindDescriptorSet": {
                    "with_name": descriptor_set,
                    "engines": True,
                    "metrics": True,
                    "dimensions": True,
                    "results": {"all_properties": True},
                }
            }
        ]
        r, b = self.connection.query(find_ds_query)
        assert self.connection.last_query_ok(), r
        n_entities = (
            len(r[0]["FindDescriptorSet"]["entities"])
            if "entities" in r[0]["FindDescriptorSet"]
            else 0
        )
        assert n_entities <= 1, "Multiple descriptor sets with the same name"

        if n_entities == 1:  # Descriptor set exists already
            e = r[0]["FindDescriptorSet"]["entities"][0]
            self.logger.info(f"Descriptor set {descriptor_set} already exists")

            engines = e["_engines"]
            assert len(engines) == 1, "Only one engine is supported"

            if self.engine is None:
                self.engine = engines[0]
            elif self.engine != engines[0]:
                self.logger.error(f"Engine mismatch: {self.engine} != {engines[0]}")

            metrics = e["_metrics"]
            assert len(metrics) == 1, "Only one metric is supported"
            if self.metric is None:
                self.metric = metrics[0]
            elif self.metric != metrics[0]:
                self.logger.error(f"Metric mismatch: {self.metric} != {metrics[0]}")

            dimensions = e["_dimensions"]
            if self.dimensions is None:
                self.dimensions = dimensions
            elif self.dimensions != dimensions:
                self.logger.error(
                    f"Dimensions mismatch: {self.dimensions} != {dimensions}"
                )

            self.properties = {
                k[len(PROPERTY_PREFIX) :]: v
                for k, v in e.items()
                if k.startswith(PROPERTY_PREFIX)
            }

        else:
            self.logger.info(
                f"Descriptor set {descriptor_set} does not exist. Creating it"
            )
            if self.engine is None:
                self.engine = ENGINE
            if self.metric is None:
                self.metric = METRIC
            if self.dimensions is None:
                self.dimensions = len(self.embedding_function.embed_query("test"))

            properties = (
                {PROPERTY_PREFIX + k: v for k, v in self.properties.items()}
                if self.properties is not None
                else None
            )

            self.utils.add_descriptorset(
                name=descriptor_set,
                dim=self.dimensions,
                engine=self.engine,
                metric=self.metric,
                properties=properties,
            )

            # Create indexes
            self.utils.create_entity_index("_Descriptor", "_create_txn")
            self.utils.create_entity_index("_DescriptorSet", "_name")
            self.utils.create_entity_index("_Descriptor", UNIQUEID_PROPERTY)

    @override
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents from the vectorstore by id.

        Args:
            ids: List of ids to delete from the vectorstore.

        Returns:
            True if the deletion was successful, False otherwise
        """
        assert ids is not None, "ids must be provided"
        query = [
            {
                "DeleteDescriptor": {
                    "set": self.descriptor_set,
                    "constraints": {UNIQUEID_PROPERTY: ["in", ids]},
                }
            }
        ]

        result, _ = self.utils.execute(query)
        return result

    @override
    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        """Find documents in the vectorstore by id.

        Args:
            ids: List of ids to find in the vectorstore.

        Returns:
            documents: List of Document objects found in the vectorstore.
        """
        query = [
            {
                "FindDescriptor": {
                    "set": self.descriptor_set,
                    "constraints": {UNIQUEID_PROPERTY: ["in", ids]},
                    "results": {"all_properties": True},
                }
            }
        ]

        results, _ = self.utils.execute(query)
        docs = [
            self._descriptor_to_document(d)
            for d in results[0]["FindDescriptor"].get("entities", [])
        ]
        return docs

    @override
    def similarity_search(
        self, query: str, k: int = 4, *args: Any, **kwargs: Any
    ) -> List[Document]:
        """Search for documents similar to the query using the vectorstore

        Args:
            query: Query string to search for.
            k: Number of results to return.

        Returns:
            List of Document objects ordered by decreasing similarity to the query.
        """
        assert self.embedding_function is not None, "Embedding function is not set"
        embedding = self.embedding_function.embed_query(query)
        return self.similarity_search_by_vector(embedding, k, *args, **kwargs)

    @override
    def similarity_search_with_score(
        self, query: str, *args: Any, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding_function.embed_query(query)
        return self._similarity_search_with_score_by_vector(embedding, *args, **kwargs)

    def _descriptor_to_document(self, d: dict) -> Document:
        metadata = {}
        for k, v in d.items():
            if k.startswith(PROPERTY_PREFIX):
                metadata[k[len(PROPERTY_PREFIX) :]] = v
        text = d[TEXT_PROPERTY]
        uniqueid = d[UNIQUEID_PROPERTY]
        doc = Document(page_content=text, metadata=metadata, id=uniqueid)
        return doc

    def _similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4, vectors: bool = False
    ) -> List[Tuple[Document, float]]:
        from aperturedb.Descriptors import Descriptors

        descriptors = Descriptors(self.connection)
        start_time = time.time()
        descriptors.find_similar(
            set=self.descriptor_set, vector=embedding, k_neighbors=k, distances=True
        )
        self.logger.info(
            f"ApertureDB similarity search took {time.time() - start_time} seconds"
        )
        return [(self._descriptor_to_document(d), d["_distance"]) for d in descriptors]

    @override
    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Returns the k most similar documents to the given embedding vector

        Args:
            embedding: The embedding vector to search for
            k: The number of similar documents to return

        Returns:
            List of Document objects ordered by decreasing similarity to the query.
        """
        from aperturedb.Descriptors import Descriptors

        descriptors = Descriptors(self.connection)
        start_time = time.time()
        descriptors.find_similar(
            set=self.descriptor_set, vector=embedding, k_neighbors=k
        )
        self.logger.info(
            f"ApertureDB similarity search took {time.time() - start_time} seconds"
        )
        return [self._descriptor_to_document(d) for d in descriptors]

    @override
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Returns similar documents to the query that also have diversity

        This algorithm balances relevance and diversity in the search results.

        Args:
            query: Query string to search for.
            k: Number of results to return.
            fetch_k: Number of results to fetch.
            lambda_mult: Lambda multiplier for MMR.

        Returns:
            List of Document objects ordered by decreasing similarity/diversty.
        """
        self.logger.info(f"Max Marginal Relevance search for query: {query}")
        embedding = self.embedding_function.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, **kwargs
        )

    @override
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Returns similar documents to the vector that also have diversity

        This algorithm balances relevance and diversity in the search results.

        Args:
            embedding: Embedding vector to search for.
            k: Number of results to return.
            fetch_k: Number of results to fetch.
            lambda_mult: Lambda multiplier for MMR.

        Returns:
            List of Document objects ordered by decreasing similarity/diversty.
        """
        from aperturedb.Descriptors import Descriptors

        descriptors = Descriptors(self.connection)
        start_time = time.time()
        descriptors.find_similar_mmr(
            set=self.descriptor_set,
            vector=embedding,
            k_neighbors=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
        self.logger.info(
            f"ApertureDB similarity search mmr took {time.time() - start_time} seconds"
        )
        return [self._descriptor_to_document(d) for d in descriptors]

    @classmethod
    @override
    def from_texts(
        cls: Type[ApertureDB],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> ApertureDB:
        """Creates a new vectorstore from a list of texts

        Args:
            texts: List of text strings
            embedding: Embeddings object as for constructing the vectorstore
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: Additional arguments to pass to the constructor
        """
        store = cls(embeddings=embedding, **kwargs)
        store.add_texts(texts, metadatas)
        return store

    @classmethod
    @override
    def from_documents(
        cls: Type[ApertureDB],
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> ApertureDB:
        """Creates a new vectorstore from a list of documents

        Args:
            documents: List of Document objects
            embedding: Embeddings object as for constructing the vectorstore
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: Additional arguments to pass to the constructor
        """
        store = cls(embeddings=embedding, **kwargs)
        store.add_documents(documents)
        return store

    @classmethod
    def delete_vectorstore(class_, descriptor_set: str) -> None:
        """Deletes a vectorstore and all its data from the database

        Args:
            descriptor_set: The name of the descriptor set to delete
        """
        from aperturedb.Utils import Utils, create_connector

        db = create_connector()
        utils = Utils(db)
        utils.remove_descriptorset(descriptor_set)

    @classmethod
    def list_vectorstores(class_) -> None:
        """Returns a list of all vectorstores in the database

        Returns:
            List of descriptor sets with properties
        """
        from aperturedb.Utils import create_connector

        db = create_connector()
        query = [
            {
                "FindDescriptorSet": {
                    # Return all properties
                    "results": {"all_properties": True},
                    "engines": True,
                    "metrics": True,
                    "dimensions": True,
                }
            }
        ]
        response, _ = db.query(query)
        assert db.last_query_ok(), response
        return response[0]["FindDescriptorSet"]["entities"]

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add or update documents in the vectorstore.

        Args:
            documents: Documents to add to the vectorstore.
            kwargs: Additional keyword arguments.
                if kwargs contains ids and documents contain ids,
                the ids in the kwargs will receive precedence.

        Returns:
            List of IDs of the added texts.

        Raises:
            ValueError: If the number of ids does not match the number of documents.
        """

        if "ids" in kwargs:
            ids = kwargs.pop("ids")
            if ids and len(ids) != len(documents):
                raise ValueError(
                    "The number of ids must match the number of documents. "
                    "Got {len(ids)} ids and {len(documents)} documents."
                )

            documents_ = []

            for id_, document in zip(ids, documents):
                doc_with_id = Document(
                    page_content=document.page_content,
                    metadata=document.metadata,
                    id=id_,
                )
                documents_.append(doc_with_id)
        else:
            documents_ = documents

        # If upsert has been implemented, we can use it to add documents
        return self.upsert(documents_, **kwargs)["succeeded"]

    def upsert(self, items: Sequence[Document], /, **kwargs: Any) -> UpsertResponse:
        """Insert or update items

        Updating documents is dependent on the documents' `id` attribute.

        Args:
            items: List of Document objects to upsert

        Returns:
            UpsertResponse object with succeeded and failed
        """
        # For now, simply delete and add
        # We could do something more efficient to update metadata,
        # but we don't support changing the embedding of a descriptor.

        from aperturedb.ParallelLoader import ParallelLoader

        ids_to_delete: List[str] = [
            item.id for item in items if hasattr(item, "id") and item.id is not None
        ]
        if ids_to_delete:
            self.delete(ids_to_delete)

        texts = [doc.page_content for doc in items]
        metadatas = [
            doc.metadata if getattr(doc, "metadata", None) is not None else {}
            for doc in items
        ]
        embeddings = self.embedding_function.embed_documents(texts)
        ids: List[str] = [
            doc.id if hasattr(doc, "id") and doc.id is not None else str(uuid.uuid4())
            for doc in items
        ]

        data = []
        for text, embedding, metadata, unique_id in zip(
            texts, embeddings, metadatas, ids
        ):
            properties = {PROPERTY_PREFIX + k: v for k, v in metadata.items()}
            properties[TEXT_PROPERTY] = text
            properties[UNIQUEID_PROPERTY] = unique_id
            command = {
                "AddDescriptor": {
                    "set": self.descriptor_set,
                    "properties": properties,
                }
            }
            query = [command]
            blobs = [np.array(embedding, dtype=np.float32).tobytes()]
            data.append((query, blobs))
        loader = ParallelLoader(self.connection)
        loader.ingest(data, batchsize=BATCHSIZE)
        return UpsertResponse(succeeded=ids, failed=[])
