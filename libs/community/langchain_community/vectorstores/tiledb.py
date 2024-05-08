"""Wrapper around TileDB vector database."""
from __future__ import annotations

import pickle
import random
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

INDEX_METRICS = frozenset(["euclidean"])
DEFAULT_METRIC = "euclidean"
DOCUMENTS_ARRAY_NAME = "documents"
VECTOR_INDEX_NAME = "vectors"
MAX_UINT64 = np.iinfo(np.dtype("uint64")).max
MAX_FLOAT_32 = np.finfo(np.dtype("float32")).max
MAX_FLOAT = sys.float_info.max


def dependable_tiledb_import() -> Any:
    """Import tiledb-vector-search if available, otherwise raise error."""
    try:
        import tiledb as tiledb
        import tiledb.vector_search as tiledb_vs
    except ImportError:
        raise ImportError(
            "Could not import tiledb-vector-search python package. "
            "Please install it with `conda install -c tiledb tiledb-vector-search` "
            "or `pip install tiledb-vector-search`"
        )
    return tiledb_vs, tiledb


def get_vector_index_uri_from_group(group: Any) -> str:
    """Get the URI of the vector index."""
    return group[VECTOR_INDEX_NAME].uri


def get_documents_array_uri_from_group(group: Any) -> str:
    """Get the URI of the documents array from group.

    Args:
        group: TileDB group object.

    Returns:
        URI of the documents array.
    """
    return group[DOCUMENTS_ARRAY_NAME].uri


def get_vector_index_uri(uri: str) -> str:
    """Get the URI of the vector index."""
    return f"{uri}/{VECTOR_INDEX_NAME}"


def get_documents_array_uri(uri: str) -> str:
    """Get the URI of the documents array."""
    return f"{uri}/{DOCUMENTS_ARRAY_NAME}"


class TileDB(VectorStore):
    """TileDB vector store.

    To use, you should have the ``tiledb-vector-search`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community import TileDB
            embeddings = OpenAIEmbeddings()
            db = TileDB(embeddings, index_uri, metric)

    """

    def __init__(
        self,
        embedding: Embeddings,
        index_uri: str,
        metric: str,
        *,
        vector_index_uri: str = "",
        docs_array_uri: str = "",
        config: Optional[Mapping[str, Any]] = None,
        timestamp: Any = None,
        allow_dangerous_deserialization: bool = False,
        **kwargs: Any,
    ):
        """Initialize with necessary components.

        Args:
            allow_dangerous_deserialization: whether to allow deserialization
                of the data which involves loading data using pickle.
                data can be modified by malicious actors to deliver a
                malicious payload that results in execution of
                arbitrary code on your machine.
        """
        if not allow_dangerous_deserialization:
            raise ValueError(
                "TileDB relies on pickle for serialization and deserialization. "
                "This can be dangerous if the data is intercepted and/or modified "
                "by malicious actors prior to being de-serialized. "
                "If you are sure that the data is safe from modification, you can "
                " set allow_dangerous_deserialization=True to proceed. "
                "Loading of compromised data using pickle can result in execution of "
                "arbitrary code on your machine."
            )
        self.embedding = embedding
        self.embedding_function = embedding.embed_query
        self.index_uri = index_uri
        self.metric = metric
        self.config = config

        tiledb_vs, tiledb = dependable_tiledb_import()
        with tiledb.scope_ctx(ctx_or_config=config):
            index_group = tiledb.Group(self.index_uri, "r")
            self.vector_index_uri = (
                vector_index_uri
                if vector_index_uri != ""
                else get_vector_index_uri_from_group(index_group)
            )
            self.docs_array_uri = (
                docs_array_uri
                if docs_array_uri != ""
                else get_documents_array_uri_from_group(index_group)
            )
            index_group.close()
            group = tiledb.Group(self.vector_index_uri, "r")
            self.index_type = group.meta.get("index_type")
            group.close()
            self.timestamp = timestamp
            if self.index_type == "FLAT":
                self.vector_index = tiledb_vs.flat_index.FlatIndex(
                    uri=self.vector_index_uri,
                    config=self.config,
                    timestamp=self.timestamp,
                    **kwargs,
                )
            elif self.index_type == "IVF_FLAT":
                self.vector_index = tiledb_vs.ivf_flat_index.IVFFlatIndex(
                    uri=self.vector_index_uri,
                    config=self.config,
                    timestamp=self.timestamp,
                    **kwargs,
                )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding

    def process_index_results(
        self,
        ids: List[int],
        scores: List[float],
        *,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: float = MAX_FLOAT,
    ) -> List[Tuple[Document, float]]:
        """Turns TileDB results into a list of documents and scores.

        Args:
            ids: List of indices of the documents in the index.
            scores: List of distances of the documents in the index.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, Any]]): Filter by metadata. Defaults to None.
            score_threshold: Optional, a floating point value to filter the
                resulting set of retrieved docs
        Returns:
            List of Documents and scores.
        """
        tiledb_vs, tiledb = dependable_tiledb_import()
        docs = []
        docs_array = tiledb.open(
            self.docs_array_uri, "r", timestamp=self.timestamp, config=self.config
        )
        for idx, score in zip(ids, scores):
            if idx == 0 and score == 0:
                continue
            if idx == MAX_UINT64 and score == MAX_FLOAT_32:
                continue
            doc = docs_array[idx]
            if doc is None or len(doc["text"]) == 0:
                raise ValueError(f"Could not find document for id {idx}, got {doc}")
            pickled_metadata = doc.get("metadata")
            result_doc = Document(page_content=str(doc["text"][0]))
            if pickled_metadata is not None:
                metadata = pickle.loads(
                    np.array(pickled_metadata.tolist()).astype(np.uint8).tobytes()
                )
                result_doc.metadata = metadata
            if filter is not None:
                filter = {
                    key: [value] if not isinstance(value, list) else value
                    for key, value in filter.items()
                }
                if all(
                    result_doc.metadata.get(key) in value
                    for key, value in filter.items()
                ):
                    docs.append((result_doc, score))
            else:
                docs.append((result_doc, score))
        docs_array.close()
        docs = [(doc, score) for doc, score in docs if score <= score_threshold]
        return docs[:k]

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        *,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, Any]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.
            **kwargs: kwargs to be passed to similarity search. Can include:
                nprobe: Optional, number of partitions to check if using IVF_FLAT index
                score_threshold: Optional, a floating point value to filter the
                    resulting set of retrieved docs

        Returns:
            List of documents most similar to the query text and distance
            in float for each. Lower score represents more similarity.
        """
        if "score_threshold" in kwargs:
            score_threshold = kwargs.pop("score_threshold")
        else:
            score_threshold = MAX_FLOAT
        d, i = self.vector_index.query(
            np.array([np.array(embedding).astype(np.float32)]).astype(np.float32),
            k=k if filter is None else fetch_k,
            **kwargs,
        )
        return self.process_index_results(
            ids=i[0], scores=d[0], filter=filter, k=k, score_threshold=score_threshold
        )

    def similarity_search_with_score(
        self,
        query: str,
        *,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of documents most similar to the query text with
            Distance as float. Lower score represents more similarity.
        """
        embedding = self.embedding_function(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return docs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of Documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(
            query, k=k, filter=filter, fetch_k=fetch_k, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        *,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and their similarity scores selected using the maximal marginal
            relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents and similarity scores selected by maximal marginal
                relevance and score for each.
        """
        if "score_threshold" in kwargs:
            score_threshold = kwargs.pop("score_threshold")
        else:
            score_threshold = MAX_FLOAT
        scores, indices = self.vector_index.query(
            np.array([np.array(embedding).astype(np.float32)]).astype(np.float32),
            k=fetch_k if filter is None else fetch_k * 2,
            **kwargs,
        )
        results = self.process_index_results(
            ids=indices[0],
            scores=scores[0],
            filter=filter,
            k=fetch_k if filter is None else fetch_k * 2,
            score_threshold=score_threshold,
        )
        embeddings = [
            self.embedding.embed_documents([doc.page_content])[0] for doc, _ in results
        ]
        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )
        docs_and_scores = []
        for i in mmr_selected:
            docs_and_scores.append(results[i])
        return docs_and_scores

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering (if needed) to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self.embedding_function(query)
        docs = self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return docs

    @classmethod
    def create(
        cls,
        index_uri: str,
        index_type: str,
        dimensions: int,
        vector_type: np.dtype,
        *,
        metadatas: bool = True,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        tiledb_vs, tiledb = dependable_tiledb_import()
        with tiledb.scope_ctx(ctx_or_config=config):
            try:
                tiledb.group_create(index_uri)
            except tiledb.TileDBError as err:
                raise err
            group = tiledb.Group(index_uri, "w")
            vector_index_uri = get_vector_index_uri(group.uri)
            docs_uri = get_documents_array_uri(group.uri)
            if index_type == "FLAT":
                tiledb_vs.flat_index.create(
                    uri=vector_index_uri,
                    dimensions=dimensions,
                    vector_type=vector_type,
                    config=config,
                )
            elif index_type == "IVF_FLAT":
                tiledb_vs.ivf_flat_index.create(
                    uri=vector_index_uri,
                    dimensions=dimensions,
                    vector_type=vector_type,
                    config=config,
                )
            group.add(vector_index_uri, name=VECTOR_INDEX_NAME)

            # Create TileDB array to store Documents
            # TODO add a Document store API to tiledb-vector-search to allow storing
            #  different types of objects and metadata in a more generic way.
            dim = tiledb.Dim(
                name="id",
                domain=(0, MAX_UINT64 - 1),
                dtype=np.dtype(np.uint64),
            )
            dom = tiledb.Domain(dim)

            text_attr = tiledb.Attr(name="text", dtype=np.dtype("U1"), var=True)
            attrs = [text_attr]
            if metadatas:
                metadata_attr = tiledb.Attr(name="metadata", dtype=np.uint8, var=True)
                attrs.append(metadata_attr)
            schema = tiledb.ArraySchema(
                domain=dom,
                sparse=True,
                allows_duplicates=False,
                attrs=attrs,
            )
            tiledb.Array.create(docs_uri, schema)
            group.add(docs_uri, name=DOCUMENTS_ARRAY_NAME)
            group.close()

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        index_uri: str,
        *,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        metric: str = DEFAULT_METRIC,
        index_type: str = "FLAT",
        config: Optional[Mapping[str, Any]] = None,
        index_timestamp: int = 0,
        **kwargs: Any,
    ) -> TileDB:
        if metric not in INDEX_METRICS:
            raise ValueError(
                (
                    f"Unsupported distance metric: {metric}. "
                    f"Expected one of {list(INDEX_METRICS)}"
                )
            )
        tiledb_vs, tiledb = dependable_tiledb_import()
        input_vectors = np.array(embeddings).astype(np.float32)
        cls.create(
            index_uri=index_uri,
            index_type=index_type,
            dimensions=input_vectors.shape[1],
            vector_type=input_vectors.dtype,
            metadatas=metadatas is not None,
            config=config,
        )
        with tiledb.scope_ctx(ctx_or_config=config):
            if not embeddings:
                raise ValueError("embeddings must be provided to build a TileDB index")

            vector_index_uri = get_vector_index_uri(index_uri)
            docs_uri = get_documents_array_uri(index_uri)
            if ids is None:
                ids = [str(random.randint(0, MAX_UINT64 - 1)) for _ in texts]
            external_ids = np.array(ids).astype(np.uint64)

            tiledb_vs.ingestion.ingest(
                index_type=index_type,
                index_uri=vector_index_uri,
                input_vectors=input_vectors,
                external_ids=external_ids,
                index_timestamp=index_timestamp if index_timestamp != 0 else None,
                config=config,
                **kwargs,
            )
            with tiledb.open(docs_uri, "w") as A:
                if external_ids is None:
                    external_ids = np.zeros(len(texts), dtype=np.uint64)
                    for i in range(len(texts)):
                        external_ids[i] = i
                data = {}
                data["text"] = np.array(texts)
                if metadatas is not None:
                    metadata_attr = np.empty([len(metadatas)], dtype=object)
                    i = 0
                    for metadata in metadatas:
                        metadata_attr[i] = np.frombuffer(
                            pickle.dumps(metadata), dtype=np.uint8
                        )
                        i += 1
                    data["metadata"] = metadata_attr

                A[external_ids] = data
        return cls(
            embedding=embedding,
            index_uri=index_uri,
            metric=metric,
            config=config,
            **kwargs,
        )

    def delete(
        self, ids: Optional[List[str]] = None, timestamp: int = 0, **kwargs: Any
    ) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            timestamp: Optional timestamp to delete with.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """

        external_ids = np.array(ids).astype(np.uint64)
        self.vector_index.delete_batch(
            external_ids=external_ids, timestamp=timestamp if timestamp != 0 else None
        )
        return True

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        timestamp: int = 0,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional ids of each text object.
            timestamp: Optional timestamp to write new texts with.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        tiledb_vs, tiledb = dependable_tiledb_import()
        embeddings = self.embedding.embed_documents(list(texts))
        if ids is None:
            ids = [str(random.randint(0, MAX_UINT64 - 1)) for _ in texts]

        external_ids = np.array(ids).astype(np.uint64)
        vectors = np.empty((len(embeddings)), dtype="O")
        for i in range(len(embeddings)):
            vectors[i] = np.array(embeddings[i], dtype=np.float32)
        self.vector_index.update_batch(
            vectors=vectors,
            external_ids=external_ids,
            timestamp=timestamp if timestamp != 0 else None,
        )

        docs = {}
        docs["text"] = np.array(texts)
        if metadatas is not None:
            metadata_attr = np.empty([len(metadatas)], dtype=object)
            i = 0
            for metadata in metadatas:
                metadata_attr[i] = np.frombuffer(pickle.dumps(metadata), dtype=np.uint8)
                i += 1
            docs["metadata"] = metadata_attr

        docs_array = tiledb.open(
            self.docs_array_uri,
            "w",
            timestamp=timestamp if timestamp != 0 else None,
            config=self.config,
        )
        docs_array[external_ids] = docs
        docs_array.close()
        return ids

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        metric: str = DEFAULT_METRIC,
        index_uri: str = "/tmp/tiledb_array",
        index_type: str = "FLAT",
        config: Optional[Mapping[str, Any]] = None,
        index_timestamp: int = 0,
        **kwargs: Any,
    ) -> TileDB:
        """Construct a TileDB index from raw documents.

        Args:
            texts: List of documents to index.
            embedding: Embedding function to use.
            metadatas: List of metadata dictionaries to associate with documents.
            ids: Optional ids of each text object.
            metric: Metric to use for indexing. Defaults to "euclidean".
            index_uri: The URI to write the TileDB arrays
            index_type: Optional,  Vector index type ("FLAT", IVF_FLAT")
            config: Optional, TileDB config
            index_timestamp: Optional, timestamp to write new texts with.

        Example:
            .. code-block:: python

                from langchain_community import TileDB
                from langchain_community.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                index = TileDB.from_texts(texts, embeddings)
        """
        embeddings = []
        embeddings = embedding.embed_documents(texts)
        return cls.__from(
            texts=texts,
            embeddings=embeddings,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            metric=metric,
            index_uri=index_uri,
            index_type=index_type,
            config=config,
            index_timestamp=index_timestamp,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        index_uri: str,
        *,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        metric: str = DEFAULT_METRIC,
        index_type: str = "FLAT",
        config: Optional[Mapping[str, Any]] = None,
        index_timestamp: int = 0,
        **kwargs: Any,
    ) -> TileDB:
        """Construct TileDB index from embeddings.

        Args:
            text_embeddings: List of tuples of (text, embedding)
            embedding: Embedding function to use.
            index_uri: The URI to write the TileDB arrays
            metadatas: List of metadata dictionaries to associate with documents.
            metric: Optional, Metric to use for indexing. Defaults to "euclidean".
            index_type: Optional, Vector index type ("FLAT", IVF_FLAT")
            config: Optional, TileDB config
            index_timestamp: Optional, timestamp to write new texts with.

        Example:
            .. code-block:: python

                from langchain_community import TileDB
                from langchain_community.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                db = TileDB.from_embeddings(text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        return cls.__from(
            texts=texts,
            embeddings=embeddings,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            metric=metric,
            index_uri=index_uri,
            index_type=index_type,
            config=config,
            index_timestamp=index_timestamp,
            **kwargs,
        )

    @classmethod
    def load(
        cls,
        index_uri: str,
        embedding: Embeddings,
        *,
        metric: str = DEFAULT_METRIC,
        config: Optional[Mapping[str, Any]] = None,
        timestamp: Any = None,
        **kwargs: Any,
    ) -> TileDB:
        """Load a TileDB index from a URI.

        Args:
            index_uri: The URI of the TileDB vector index.
            embedding: Embeddings to use when generating queries.
            metric: Optional, Metric to use for indexing. Defaults to "euclidean".
            config: Optional, TileDB config
            timestamp: Optional, timestamp to use for opening the arrays.
        """
        return cls(
            embedding=embedding,
            index_uri=index_uri,
            metric=metric,
            config=config,
            timestamp=timestamp,
            **kwargs,
        )

    def consolidate_updates(self, **kwargs: Any) -> None:
        self.vector_index = self.vector_index.consolidate_updates(**kwargs)
