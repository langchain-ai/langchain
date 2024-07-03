# System imports
from __future__ import annotations

import logging
import time
import uuid
from itertools import repeat
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

# Third-party imports
import numpy as np

# Local imports
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
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
        **kwargs,
    ):
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
    def add_documents(self, documents: List[Document], **kwargs: Any):
        """Adds documents to the vectorstore with embeddings

        Args:
            documents: List of Document objects to add to the vectorstore.

        Returns:
            List of ids from adding the documents into the vectorstore.
        """
        from aperturedb.ParallelLoader import ParallelLoader

        texts = [doc.page_content for doc in documents]
        metadatas = [
            doc.metadata if getattr(doc, "metadata", None) is not None else {}
            for doc in documents
        ]
        embeddings = self.embedding_function.embed_documents(texts)
        ids = [
            doc.id if getattr(doc, "id", None) is not None else str(uuid.uuid4())
            for doc in documents
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
        return ids

    @override
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> List[str]:
        """Creates embeddings of texts, then adds each text object, its embedding and
        associated metadata to aperturedb

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if metadatas is not None:
            assert len(texts) == len(
                metadatas
            ), "Length of texts and metadatas should be the same"

            docs = [
                Document(page_content=text, metadata=metadata)
                for text, metadata in zip(texts, metadatas)
            ]
        else:
            docs = [Document(page_content=text) for text in texts]

        return self.add_documents(docs)

    @override
    def delete(self, ids: List[str]) -> Optional[bool]:
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
    def similarity_search(
        self, query: str, k: int = 4, *args: Any, **kwargs: Any
    ) -> List[Document]:
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
        store = cls(embeddings=embedding, **kwargs)
        store.add_texts(texts, metadatas)
        return store

    @classmethod
    def delete_vectorstore(class_, descriptor_set: str) -> None:
        """Deletes a vectorstore and all its data from the database"""
        from aperturedb.Utils import Utils, create_connector

        db = create_connector()
        utils = Utils(db)
        utils.remove_descriptorset(descriptor_set)

    @classmethod
    def list_vectorstores(class_) -> None:
        """Returns a list of all vectorstores in the database"""
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
