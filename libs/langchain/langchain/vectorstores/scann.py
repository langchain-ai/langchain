from __future__ import annotations

import operator
import pickle
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain.docstore.base import AddableMixin, Docstore
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.vectorstores.utils import DistanceStrategy


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length."""
    x /= np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)
    return x


def dependable_scann_import() -> Any:
    """
    Import `scann` if available, otherwise raise error.
    """
    try:
        import scann
    except ImportError:
        raise ImportError(
            "Could not import scann python package. "
            "Please install it with `pip install scann` "
        )
    return scann


class ScaNN(VectorStore):
    """`ScaNN` vector store.

    To use, you should have the ``scann`` python package installed.

    Example:
        .. code-block:: python

            from langchain.embeddings import HuggingFaceEmbeddings
            from langchain.vectorstores import ScaNN

            db = ScaNN.from_texts(
                ['foo', 'bar', 'barz', 'qux'],
                HuggingFaceEmbeddings())
            db.similarity_search('foo?', k=1)
    """

    def __init__(
        self,
        embedding: Embeddings,
        index: Any,
        docstore: Docstore,
        index_to_docstore_id: Dict[int, str],
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        normalize_L2: bool = False,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        scann_config: Optional[str] = None,
    ):
        """Initialize with necessary components."""
        self.embedding = embedding
        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id
        self.distance_strategy = distance_strategy
        self.override_relevance_score_fn = relevance_score_fn
        self._normalize_L2 = normalize_L2
        self._scann_config = scann_config

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
        raise NotImplementedError("Updates are not available in ScaNN, yet.")

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
        # Embed and create the documents.
        embeddings = self.embedding.embed_documents(list(texts))
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

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """

        raise NotImplementedError("Deletions are not available in ScaNN, yet.")

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
        vector = np.array([embedding], dtype=np.float32)
        if self._normalize_L2:
            vector = normalize(vector)
        indices, scores = self.index.search_batched(
            vector, k if filter is None else fetch_k
        )
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
        embedding = self.embedding.embed_query(query)
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
    ) -> ScaNN:
        scann = dependable_scann_import()
        distance_strategy = kwargs.get(
            "distance_strategy", DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        scann_config = kwargs.get("scann_config", None)

        vector = np.array(embeddings, dtype=np.float32)
        if normalize_L2:
            vector = normalize(vector)
        if scann_config is not None:
            index = scann.scann_ops_pybind.create_searcher(vector, scann_config)
        else:
            if distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
                index = (
                    scann.scann_ops_pybind.builder(vector, 1, "dot_product")
                    .score_brute_force()
                    .build()
                )
            else:
                # Default to L2, currently other metric types not initialized.
                index = (
                    scann.scann_ops_pybind.builder(vector, 1, "squared_l2")
                    .score_brute_force()
                    .build()
                )
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
            embedding,
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
    ) -> ScaNN:
        """Construct ScaNN wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the ScaNN database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain.vectorstores import ScaNN
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                scann = ScaNN.from_texts(texts, embeddings)
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
    ) -> ScaNN:
        """Construct ScaNN wrapper from raw documents.

        This is a user friendly interface that:
            1. Embeds documents.
            2. Creates an in memory docstore
            3. Initializes the ScaNN database

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain.vectorstores import ScaNN
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                scann = ScaNN.from_embeddings(text_embedding_pairs, embeddings)
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
        """Save ScaNN index, docstore, and index_to_docstore_id to disk.

        Args:
            folder_path: folder path to save index, docstore,
                and index_to_docstore_id to.
        """
        path = Path(folder_path)
        scann_path = path / "{index_name}.scann".format(index_name=index_name)
        scann_path.mkdir(exist_ok=True, parents=True)

        # save index separately since it is not picklable
        self.index.serialize(str(scann_path))

        # save docstore and index_to_docstore_id
        with open(path / "{index_name}.pkl".format(index_name=index_name), "wb") as f:
            pickle.dump((self.docstore, self.index_to_docstore_id), f)

    @classmethod
    def load_local(
        cls,
        folder_path: str,
        embedding: Embeddings,
        index_name: str = "index",
        **kwargs: Any,
    ) -> ScaNN:
        """Load ScaNN index, docstore, and index_to_docstore_id from disk.

        Args:
            folder_path: folder path to load index, docstore,
                and index_to_docstore_id from.
            embeddings: Embeddings to use when generating queries
            index_name: for saving with a specific index file name
        """
        path = Path(folder_path)
        scann_path = path / "{index_name}.scann".format(index_name=index_name)
        scann_path.mkdir(exist_ok=True, parents=True)
        # load index separately since it is not picklable
        scann = dependable_scann_import()
        index = scann.scann_ops_pybind.load_searcher(str(scann_path))

        # load docstore and index_to_docstore_id
        with open(path / "{index_name}.pkl".format(index_name=index_name), "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)
        return cls(embedding, index, docstore, index_to_docstore_id, **kwargs)

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
                " ScaNN constructor to normalize scores"
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
