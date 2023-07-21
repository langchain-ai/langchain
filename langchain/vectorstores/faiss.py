"""Wrapper around FAISS vector database."""
from __future__ import annotations

import operator
import os
import pickle
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from langchain.docstore.base import AddableMixin, Docstore
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import DistanceStrategy, maximal_marginal_relevance


def dependable_faiss_import(no_avx2: Optional[bool] = None) -> Any:
    """
    Import faiss if available, otherwise raise error.
    If FAISS_NO_AVX2 environment variable is set, it will be considered
    to load FAISS with no AVX2 optimization.

    Args:
        no_avx2: Load FAISS strictly with no AVX2 optimization
            so that the vectorstore is portable and compatible with other devices.
    """
    if no_avx2 is None and "FAISS_NO_AVX2" in os.environ:
        no_avx2 = bool(os.getenv("FAISS_NO_AVX2"))

    try:
        if no_avx2:
            from faiss import swigfaiss as faiss
        else:
            import faiss
    except ImportError:
        raise ImportError(
            "Could not import faiss python package. "
            "Please install it with `pip install faiss` "
            "or `pip install faiss-cpu` (depending on Python version)."
        )
    return faiss


class FAISS(VectorStore):
    """Wrapper around FAISS vector database.

    To use, you should have the ``faiss`` python package installed.

    Example:
        .. code-block:: python

            from langchain import FAISS
            faiss = FAISS(embedding_function, index, docstore, index_to_docstore_id)

    """

    def __init__(
        self,
        embedding_function: Callable,
        index: Any,
        docstore: Docstore,
        index_to_docstore_id: Dict[int, str],
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        normalize_L2: bool = False,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
    ):
        """Initialize with necessary components."""
        self.embedding_function = embedding_function
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id
        self.distance_strategy = distance_strategy
        self.override_relevance_score_fn = relevance_score_fn
        self._normalize_L2 = normalize_L2
        if (
            self.distance_strategy != DistanceStrategy.EUCLIDEAN_DISTANCE
            and self._normalize_L2
        ):
            warnings.warn(
                "Normalizing L2 is not applicable for metric type: {strategy}".format(
                    strategy=self.distance_strategy
                )
            )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        # TODO: Accept embeddings object directly
        return None

    def __add(
        self,
        texts: Iterable[str],
        embeddings: Iterable[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.docstore} does not"
            )
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        # Add to the index, the index_to_id mapping, and the docstore.
        starting_len = len(self.index_to_docstore_id)
        faiss = dependable_faiss_import()
        vector = np.array(embeddings, dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        self.index.add(vector)
        # Get list of index, id, and docs.
        full_info = [(starting_len + i, ids[i], doc) for i, doc in enumerate(documents)]
        # Add information to docstore and index.
        self.docstore.add({_id: doc for _, _id, doc in full_info})
        index_to_id = {index: _id for index, _id, _ in full_info}
        self.index_to_docstore_id.update(index_to_id)
        return [_id for _, _id, _ in full_info]

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.docstore} does not"
            )
        # Embed and create the documents.
        embeddings = [self.embedding_function(text) for text in texts]
        return self.__add(texts, embeddings, metadatas=metadatas, ids=ids, **kwargs)

    def add_embeddings(
        self,
        text_embeddings: Iterable[Tuple[str, List[float]]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            text_embeddings: Iterable pairs of string and embedding to
                add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.docstore} does not"
            )
        # Embed and create the documents.
        texts, embeddings = zip(*text_embeddings)

        return self.__add(texts, embeddings, metadatas=metadatas, ids=ids, **kwargs)

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
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
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of documents most similar to the query text and L2 distance
            in float for each. Lower score represents more similarity.
        """
        faiss = dependable_faiss_import()
        vector = np.array([embedding], dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, k if filter is None else fetch_k)
        docs = []
        for j, i in enumerate(indices[0]):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            if filter is not None:
                filter = {
                    key: [value] if not isinstance(value, list) else value
                    for key, value in filter.items()
                }
                if all(doc.metadata.get(key) in value for key, value in filter.items()):
                    docs.append((doc, scores[0][j]))
            else:
                docs.append((doc, scores[0][j]))

        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            cmp = (
                operator.ge
                if self.distance_strategy
                in (DistanceStrategy.MAX_INNER_PRODUCT, DistanceStrategy.JACCARD)
                else operator.le
            )
            docs = [
                (doc, similarity)
                for doc, similarity in docs
                if cmp(similarity, score_threshold)
            ]
        return docs[:k]

    def similarity_search_with_score(
        self,
        query: str,
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
            L2 distance in float. Lower score represents more similarity.
        """
        embedding = self.embedding_function(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding,
            k,
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
            k,
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
            query, k, filter=filter, fetch_k=fetch_k, **kwargs
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
        scores, indices = self.index.search(
            np.array([embedding], dtype=np.float32),
            fetch_k if filter is None else fetch_k * 2,
        )
        if filter is not None:
            filtered_indices = []
            for i in indices[0]:
                if i == -1:
                    # This happens when not enough docs are returned.
                    continue
                _id = self.index_to_docstore_id[i]
                doc = self.docstore.search(_id)
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                if all(doc.metadata.get(key) == value for key, value in filter.items()):
                    filtered_indices.append(i)
            indices = np.array([filtered_indices])
        # -1 happens when not enough docs are returned.
        embeddings = [self.index.reconstruct(int(i)) for i in indices[0] if i != -1]
        mmr_selected = maximal_marginal_relevance(
            np.array([embedding], dtype=np.float32),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )
        selected_indices = [indices[0][i] for i in mmr_selected]
        selected_scores = [scores[0][i] for i in mmr_selected]
        docs_and_scores = []
        for i, score in zip(selected_indices, selected_scores):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            docs_and_scores.append((doc, score))
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
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
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

    def merge_from(self, target: FAISS) -> None:
        """Merge another FAISS object with the current one.

        Add the target FAISS to the current one.

        Args:
            target: FAISS object you wish to merge into the current one

        Returns:
            None.
        """
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError("Cannot merge with this type of docstore")
        # Numerical index for target docs are incremental on existing ones
        starting_len = len(self.index_to_docstore_id)

        # Merge two IndexFlatL2
        self.index.merge_from(target.index)

        # Get id and docs from target FAISS object
        full_info = []
        for i, target_id in target.index_to_docstore_id.items():
            doc = target.docstore.search(target_id)
            if not isinstance(doc, Document):
                raise ValueError("Document should be returned")
            full_info.append((starting_len + i, target_id, doc))

        # Add information to docstore and index_to_docstore_id.
        self.docstore.add({_id: doc for _, _id, doc in full_info})
        index_to_id = {index: _id for index, _id, _ in full_info}
        self.index_to_docstore_id.update(index_to_id)

    @classmethod
    def __from(
        cls,
        texts: List[str],
        embeddings: List[List[float]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        normalize_L2: bool = False,
        **kwargs: Any,
    ) -> FAISS:
        faiss = dependable_faiss_import()
        distance_strategy = kwargs.get(
            "distance_strategy", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        if distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            index = faiss.IndexFlatIP(len(embeddings[0]))
        else:
            # Default to L2, currently other metric types not initialized.
            index = faiss.IndexFlatL2(len(embeddings[0]))
        vector = np.array(embeddings, dtype=np.float32)
        if normalize_L2 and distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            faiss.normalize_L2(vector)
        index.add(vector)
        documents = []
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            documents.append(Document(page_content=text, metadata=metadata))
        index_to_id = dict(enumerate(ids))

        if len(index_to_id) != len(documents):
            raise Exception(
                f"{len(index_to_id)} ids provided for {len(documents)} documents."
                " Each document should have an id."
            )

        docstore = InMemoryDocstore(dict(zip(index_to_id.values(), documents)))
        return cls(
            embedding.embed_query,
            index,
            docstore,
            index_to_id,
            normalize_L2=normalize_L2,
            **kwargs,
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> FAISS:
        """Construct FAISS wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the FAISS database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import FAISS
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                faiss = FAISS.from_texts(texts, embeddings)
        """
        embeddings = embedding.embed_documents(texts)
        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
        cls,
        text_embeddings: List[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> FAISS:
        """Construct FAISS wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the FAISS database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import FAISS
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                faiss = FAISS.from_embeddings(text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]
        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    def save_local(self, folder_path: str, index_name: str = "index") -> None:
        """Save FAISS index, docstore, and index_to_docstore_id to disk.

        Args:
            folder_path: folder path to save index, docstore,
                and index_to_docstore_id to.
            index_name: for saving with a specific index file name
        """
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)

        # save index separately since it is not picklable
        faiss = dependable_faiss_import()
        faiss.write_index(
            self.index, str(path / "{index_name}.faiss".format(index_name=index_name))
        )

        # save docstore and index_to_docstore_id
        with open(path / "{index_name}.pkl".format(index_name=index_name), "wb") as f:
            pickle.dump((self.docstore, self.index_to_docstore_id), f)

    @classmethod
    def load_local(
        cls,
        folder_path: str,
        embeddings: Embeddings,
        index_name: str = "index",
        **kwargs: Any,
    ) -> FAISS:
        """Load FAISS index, docstore, and index_to_docstore_id from disk.

        Args:
            folder_path: folder path to load index, docstore,
                and index_to_docstore_id from.
            embeddings: Embeddings to use when generating queries
            index_name: for saving with a specific index file name
        """
        path = Path(folder_path)
        # load index separately since it is not picklable
        faiss = dependable_faiss_import()
        index = faiss.read_index(
            str(path / "{index_name}.faiss".format(index_name=index_name))
        )

        # load docstore and index_to_docstore_id
        with open(path / "{index_name}.pkl".format(index_name=index_name), "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)
        return cls(
            embeddings.embed_query, index, docstore, index_to_docstore_id, **kwargs
        )

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # Default strategy is to rely on distance strategy provided in
        # vectorstore constructor
        if self.distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        elif self.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            # Default behavior is to use euclidean distance relevancy
            return self._euclidean_relevance_score_fn
        else:
            raise ValueError(
                "Unknown distance strategy, must be cosine, max_inner_product,"
                " or euclidean"
            )

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and their similarity scores on a scale from 0 to 1."""
        # Pop score threshold so that only relevancy scores, not raw scores, are
        # filtered.
        score_threshold = kwargs.pop("score_threshold", None)
        relevance_score_fn = self._select_relevance_score_fn()
        if relevance_score_fn is None:
            raise ValueError(
                "normalize_score_fn must be provided to"
                " FAISS constructor to normalize scores"
            )
        docs_and_scores = self.similarity_search_with_score(
            query,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        docs_and_rel_scores = [
            (doc, relevance_score_fn(score)) for doc, score in docs_and_scores
        ]
        if score_threshold is not None:
            docs_and_rel_scores = [
                (doc, similarity)
                for doc, similarity in docs_and_rel_scores
                if similarity >= score_threshold
            ]
        return docs_and_rel_scores
