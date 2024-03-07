from __future__ import annotations

import os
import pickle
import uuid
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.docstore.base import Docstore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import maximal_marginal_relevance

INDEX_METRICS = frozenset(["angular", "euclidean", "manhattan", "hamming", "dot"])
DEFAULT_METRIC = "angular"


def dependable_annoy_import() -> Any:
    """Import annoy if available, otherwise raise error."""
    try:
        import annoy
    except ImportError:
        raise ImportError(
            "Could not import annoy python package. "
            "Please install it with `pip install --user annoy` "
        )
    return annoy


class Annoy(VectorStore):
    """`Annoy` vector store.

    To use, you should have the ``annoy`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import Annoy
            db = Annoy(embedding_function, index, docstore, index_to_docstore_id)

    """

    def __init__(
        self,
        embedding_function: Callable,
        index: Any,
        metric: str,
        docstore: Docstore,
        index_to_docstore_id: Dict[int, str],
    ):
        """Initialize with necessary components."""
        self.embedding_function = embedding_function
        self.index = index
        self.metric = metric
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id

    @property
    def embeddings(self) -> Optional[Embeddings]:
        # TODO: Accept embedding object directly
        return None

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        raise NotImplementedError(
            "Annoy does not allow to add new data once the index is build."
        )

    def process_index_results(
        self, idxs: List[int], dists: List[float]
    ) -> List[Tuple[Document, float]]:
        """Turns annoy results into a list of documents and scores.

        Args:
            idxs: List of indices of the documents in the index.
            dists: List of distances of the documents in the index.
        Returns:
            List of Documents and scores.
        """
        docs = []
        for idx, dist in zip(idxs, dists):
            _id = self.index_to_docstore_id[idx]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs.append((doc, dist))
        return docs

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4, search_k: int = -1
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            search_k: inspect up to search_k nodes which defaults
                to n_trees * n if not provided
        Returns:
            List of Documents most similar to the query and score for each
        """
        idxs, dists = self.index.get_nns_by_vector(
            embedding, k, search_k=search_k, include_distances=True
        )
        return self.process_index_results(idxs, dists)

    def similarity_search_with_score_by_index(
        self, docstore_index: int, k: int = 4, search_k: int = -1
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            search_k: inspect up to search_k nodes which defaults
                to n_trees * n if not provided
        Returns:
            List of Documents most similar to the query and score for each
        """
        idxs, dists = self.index.get_nns_by_item(
            docstore_index, k, search_k=search_k, include_distances=True
        )
        return self.process_index_results(idxs, dists)

    def similarity_search_with_score(
        self, query: str, k: int = 4, search_k: int = -1
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            search_k: inspect up to search_k nodes which defaults
                to n_trees * n if not provided

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embedding_function(query)
        docs = self.similarity_search_with_score_by_vector(embedding, k, search_k)
        return docs

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, search_k: int = -1, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            search_k: inspect up to search_k nodes which defaults
                to n_trees * n if not provided

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding, k, search_k
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_by_index(
        self, docstore_index: int, k: int = 4, search_k: int = -1, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to docstore_index.

        Args:
            docstore_index: Index of document in docstore
            k: Number of Documents to return. Defaults to 4.
            search_k: inspect up to search_k nodes which defaults
                to n_trees * n if not provided

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_and_scores = self.similarity_search_with_score_by_index(
            docstore_index, k, search_k
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search(
        self, query: str, k: int = 4, search_k: int = -1, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            search_k: inspect up to search_k nodes which defaults
                to n_trees * n if not provided

        Returns:
            List of Documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(query, k, search_k)
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            k: Number of Documents to return. Defaults to 4.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        idxs = self.index.get_nns_by_vector(
            embedding, fetch_k, search_k=-1, include_distances=False
        )
        embeddings = [self.index.get_item_vector(i) for i in idxs]
        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )
        # ignore the -1's if not enough docs are returned/indexed
        selected_indices = [idxs[i] for i in mmr_selected if i != -1]

        docs = []
        for i in selected_indices:
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs.append(doc)
        return docs

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
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self.embedding_function(query)
        docs = self.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult=lambda_mult
        )
        return docs

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        metric: str = DEFAULT_METRIC,
        trees: int = 100,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> Annoy:
        if metric not in INDEX_METRICS:
            raise ValueError(
                (
                    f"Unsupported distance metric: {metric}. "
                    f"Expected one of {list(INDEX_METRICS)}"
                )
            )
        annoy = dependable_annoy_import()
        if not embeddings:
            raise ValueError("embeddings must be provided to build AnnoyIndex")
        f = len(embeddings[0])
        index = annoy.AnnoyIndex(f, metric=metric)
        for i, emb in enumerate(embeddings):
            index.add_item(i, emb)
        index.build(trees, n_jobs=n_jobs)

        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))
        index_to_id = {i: str(uuid.uuid4()) for i in range(len(documents))}
        docstore = InMemoryDocstore(
            {index_to_id[i]: doc for i, doc in enumerate(documents)}
        )
        return cls(embedding.embed_query, index, metric, docstore, index_to_id)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        metric: str = DEFAULT_METRIC,
        trees: int = 100,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> Annoy:
        """Construct Annoy wrapper from raw documents.

        Args:
            texts: List of documents to index.
            embedding: Embedding function to use.
            metadatas: List of metadata dictionaries to associate with documents.
            metric: Metric to use for indexing. Defaults to "angular".
            trees: Number of trees to use for indexing. Defaults to 100.
            n_jobs: Number of jobs to use for indexing. Defaults to -1.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the Annoy database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import Annoy
                from langchain_community.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                index = Annoy.from_texts(texts, embeddings)
        """
        embeddings = embedding.embed_documents(texts)
        return cls.__from(
            texts, embeddings, embedding, metadatas, metric, trees, n_jobs, **kwargs
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        metric: str = DEFAULT_METRIC,
        trees: int = 100,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> Annoy:
        """Construct Annoy wrapper from embeddings.

        Args:
            text_embeddings: List of tuples of (text, embedding)
            embedding: Embedding function to use.
            metadatas: List of metadata dictionaries to associate with documents.
            metric: Metric to use for indexing. Defaults to "angular".
            trees: Number of trees to use for indexing. Defaults to 100.
            n_jobs: Number of jobs to use for indexing. Defaults to -1

        This is a user friendly interface that:
            1. Creates an in memory docstore with provided embeddings
            2. Initializes the Annoy database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import Annoy
                from langchain_community.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                db = Annoy.from_embeddings(text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        return cls.__from(
            texts, embeddings, embedding, metadatas, metric, trees, n_jobs, **kwargs
        )

    def save_local(self, folder_path: str, prefault: bool = False) -> None:
        """Save Annoy index, docstore, and index_to_docstore_id to disk.

        Args:
            folder_path: folder path to save index, docstore,
                and index_to_docstore_id to.
            prefault: Whether to pre-load the index into memory.
        """
        path = Path(folder_path)
        os.makedirs(path, exist_ok=True)
        # save index, index config, docstore and index_to_docstore_id
        config_object = ConfigParser()
        config_object["ANNOY"] = {
            "f": self.index.f,
            "metric": self.metric,
        }
        self.index.save(str(path / "index.annoy"), prefault=prefault)
        with open(path / "index.pkl", "wb") as file:
            pickle.dump((self.docstore, self.index_to_docstore_id, config_object), file)

    @classmethod
    def load_local(
        cls,
        folder_path: str,
        embeddings: Embeddings,
        *,
        allow_dangerous_deserialization: bool = False,
    ) -> Annoy:
        """Load Annoy index, docstore, and index_to_docstore_id to disk.

        Args:
            folder_path: folder path to load index, docstore,
                and index_to_docstore_id from.
            embeddings: Embeddings to use when generating queries.
            allow_dangerous_deserialization: whether to allow deserialization
                of the data which involves loading a pickle file.
                Pickle files can be modified by malicious actors to deliver a
                malicious payload that results in execution of
                arbitrary code on your machine.
        """
        if not allow_dangerous_deserialization:
            raise ValueError(
                "The de-serialization relies loading a pickle file. "
                "Pickle files can be modified to deliver a malicious payload that "
                "results in execution of arbitrary code on your machine."
                "You will need to set `allow_dangerous_deserialization` to `True` to "
                "enable deserialization. If you do this, make sure that you "
                "trust the source of the data. For example, if you are loading a "
                "file that you created, and no that no one else has modified the file, "
                "then this is safe to do. Do not set this to `True` if you are loading "
                "a file from an untrusted source (e.g., some random site on the "
                "internet.)."
            )
        path = Path(folder_path)
        # load index separately since it is not picklable
        annoy = dependable_annoy_import()
        # load docstore and index_to_docstore_id
        with open(path / "index.pkl", "rb") as file:
            docstore, index_to_docstore_id, config_object = pickle.load(file)

        f = int(config_object["ANNOY"]["f"])
        metric = config_object["ANNOY"]["metric"]

        index = annoy.AnnoyIndex(f, metric=metric)
        index.load(str(path / "index.annoy"))

        return cls(
            embeddings.embed_query, index, metric, docstore, index_to_docstore_id
        )
