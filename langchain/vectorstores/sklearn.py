""" Wrapper around scikit-learn NearestNeighbors implementation.

The vector store can be persisted in json, bson or parquet format.
"""

import json
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Type
from uuid import uuid4

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.utils import guard_import
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance

DEFAULT_K = 4  # Number of Documents to return.
DEFAULT_FETCH_K = 20  # Number of Documents to initially fetch during MMR search.


class BaseSerializer(ABC):
    """Abstract base class for saving and loading data."""

    def __init__(self, persist_path: str) -> None:
        self.persist_path = persist_path

    @classmethod
    @abstractmethod
    def extension(cls) -> str:
        """The file extension suggested by this serializer (without dot)."""

    @abstractmethod
    def save(self, data: Any) -> None:
        """Saves the data to the persist_path"""

    @abstractmethod
    def load(self) -> Any:
        """Loads the data from the persist_path"""


class JsonSerializer(BaseSerializer):
    """Serializes data in json using the json package from python standard library."""

    @classmethod
    def extension(cls) -> str:
        return "json"

    def save(self, data: Any) -> None:
        with open(self.persist_path, "w") as fp:
            json.dump(data, fp)

    def load(self) -> Any:
        with open(self.persist_path, "r") as fp:
            return json.load(fp)


class BsonSerializer(BaseSerializer):
    """Serializes data in binary json using the bson python package."""

    def __init__(self, persist_path: str) -> None:
        super().__init__(persist_path)
        self.bson = guard_import("bson")

    @classmethod
    def extension(cls) -> str:
        return "bson"

    def save(self, data: Any) -> None:
        with open(self.persist_path, "wb") as fp:
            fp.write(self.bson.dumps(data))

    def load(self) -> Any:
        with open(self.persist_path, "rb") as fp:
            return self.bson.loads(fp.read())


class ParquetSerializer(BaseSerializer):
    """Serializes data in Apache Parquet format using the pyarrow package."""

    def __init__(self, persist_path: str) -> None:
        super().__init__(persist_path)
        self.pd = guard_import("pandas")
        self.pa = guard_import("pyarrow")
        self.pq = guard_import("pyarrow.parquet")

    @classmethod
    def extension(cls) -> str:
        return "parquet"

    def save(self, data: Any) -> None:
        df = self.pd.DataFrame(data)
        table = self.pa.Table.from_pandas(df)
        if os.path.exists(self.persist_path):
            backup_path = str(self.persist_path) + "-backup"
            os.rename(self.persist_path, backup_path)
            try:
                self.pq.write_table(table, self.persist_path)
            except Exception as exc:
                os.rename(backup_path, self.persist_path)
                raise exc
            else:
                os.remove(backup_path)
        else:
            self.pq.write_table(table, self.persist_path)

    def load(self) -> Any:
        table = self.pq.read_table(self.persist_path)
        df = table.to_pandas()
        return {col: series.tolist() for col, series in df.items()}


SERIALIZER_MAP: Dict[str, Type[BaseSerializer]] = {
    "json": JsonSerializer,
    "bson": BsonSerializer,
    "parquet": ParquetSerializer,
}


class SKLearnVectorStoreException(RuntimeError):
    pass

class SKLearnVectorStoreBase(VectorStore):
    """A simple in-memory vector store based on the scikit-learn library
    NearestNeighbors implementation."""

    def __init__(
        self,
        embedding: Embeddings,
        *,
        persist_path: Optional[str] = None,
        serializer: Literal["json", "bson", "parquet"] = "json",
    ) -> None:
        np = guard_import("numpy")
        self._np = np
        # non-persistent properties
        self._embedding_function = embedding
        self._persist_path = persist_path
        self._serializer: Optional[BaseSerializer] = None
        if self._persist_path is not None:
            serializer_cls = SERIALIZER_MAP[serializer]
            self._serializer = serializer_cls(persist_path=self._persist_path)

        # data properties
        self._embeddings: List[List[float]] = []
        self._texts: List[str] = []
        self._metadatas: List[dict] = []
        self._ids: List[str] = []

        # cache properties
        self._embeddings_np: Any = np.asarray([])

        if self._persist_path is not None and os.path.isfile(self._persist_path):
            self._load()

    def persist(self) -> None:
        if self._serializer is None:
            raise SKLearnVectorStoreException(
                "You must specify a persist_path on creation to persist the "
                "collection."
            )
        data = {
            "ids": self._ids,
            "texts": self._texts,
            "metadatas": self._metadatas,
            "embeddings": self._embeddings,
        }
        self._serializer.save(data)

    def _load(self) -> None:
        if self._serializer is None:
            raise SKLearnVectorStoreException(
                "You must specify a persist_path on creation to load the " "collection."
            )
        data = self._serializer.load()
        self._embeddings = data["embeddings"]
        self._texts = data["texts"]
        self._metadatas = data["metadatas"]
        self._ids = data["ids"]
        self._update()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        _texts = list(texts)
        _ids = ids or [str(uuid4()) for _ in _texts]
        self._texts.extend(_texts)
        self._embeddings.extend(self._embedding_function.embed_documents(_texts))
        self._metadatas.extend(metadatas or ([{}] * len(_texts)))
        self._ids.extend(_ids)
        self._update()
        return _ids

    def _update(self) -> None:
        if len(self._embeddings) == 0:
            raise SKLearnVectorStoreException(
                "No data was added to SKLearnVectorStore."
            )
        self._embeddings_np = self._np.asarray(self._embeddings)

    def _get_filtered_data(self, filter: Optional[Dict[str, str]]=None):
        mask = self._np.ones(len(self._ids), dtype=bool)
        if filter is None or len(filter) == 0:
            return mask
        for key, value in filter.items():
            for i, metadata_ in enumerate(self._metadatas):
                if isinstance(value, str):
                    passed = metadata_[key] == value
                elif callable(value):
                    try:
                        passed = value(metadata_[key])
                        if not isinstance(passed, bool):
                            raise TypeError("Callable filters have to return a bool")
                    except Exception as err:
                        raise RuntimeError(
                            f'Callable filter for {key} failed. It has to be a callable taking a single argument as input and returning a bool'
                        ) from err
                    else:
                        raise ValueError(f'Filters have to be either a callable or str. Got {type(value)} for {key}')
                        
                mask[i] = mask[i] and passed
        return mask

    @abstractmethod
    def _similarity_index_search_with_score(
        self, query_embedding: List[float], *, k: int = DEFAULT_K, filter: Optional[dict] = None,  **kwargs: Any
    ) -> List[Tuple[int, float]]:
        ...

    def similarity_search_with_score(
        self, query: str, *, k: int = DEFAULT_K, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        query_embedding = self._embedding_function.embed_query(query)
        indices_dists = self._similarity_index_search_with_score(
            query_embedding, k=k, **kwargs
        )
        return [
            (
                Document(
                    page_content=self._texts[idx],
                    metadata={"id": self._ids[idx], **self._metadatas[idx]},
                ),
                dist,
            )
            for idx, dist in indices_dists
        ]


    def similarity_search(
        self, query: str, k: int = DEFAULT_K, **kwargs: Any
    ) -> List[Document]:
        docs_scores = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in docs_scores]

    def _similarity_search_with_relevance_scores(
        self, query: str, k: int = DEFAULT_K, **kwargs: Any, 
    ) -> List[Tuple[Document, float]]:
        docs_dists = self.similarity_search_with_score(query, k=k, **kwargs)
        docs, dists = zip(*docs_dists)
        scores = [1 / math.exp(dist) for dist in dists]
        return list(zip(list(docs), scores))

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_K,
        fetch_k: int = DEFAULT_FETCH_K,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.
        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        indices_dists = self._similarity_index_search_with_score(
            embedding, k=fetch_k, **kwargs
        )
        indices, _ = zip(*indices_dists)
        result_embeddings = self._embeddings_np[indices,]
        mmr_selected = maximal_marginal_relevance(
            self._np.array(embedding, dtype=self._np.float32),
            result_embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )
        mmr_indices = [indices[i] for i in mmr_selected]
        return [
            Document(
                page_content=self._texts[idx],
                metadata={"id": self._ids[idx], **self._metadatas[idx]},
            )
            for idx in mmr_indices
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = DEFAULT_K,
        fetch_k: int = DEFAULT_FETCH_K,
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
        if self._embedding_function is None:
            raise ValueError(
                "For MMR search, you must specify an embedding function on creation."
            )

        embedding = self._embedding_function.embed_query(query)
        docs = self.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mul=lambda_mult
        )
        return docs

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        persist_path: Optional[str] = None,
        **kwargs: Any,
    ) -> "SKLearnVectorStore":
        vs = SKLearnVectorStore(embedding, persist_path=persist_path, **kwargs)
        vs.add_texts(texts, metadatas=metadatas, ids=ids)
        return vs


class SKLearnKNNVectorStore(SKLearnVectorStoreBase):
    """A simple in-memory vector store based on the scikit-learn library
    NearestNeighbors implementation."""

    def __init__(
        self,
        embedding: Embeddings,
        *,
        persist_path: Optional[str] = None,
        serializer: Literal["json", "bson", "parquet"] = "json",
        metric: str = "cosine",
        **kwargs: Any,
    ) -> None:
        sklearn_neighbors = guard_import("sklearn.neighbors", pip_name="scikit-learn")
        self._neighbors = sklearn_neighbors.NearestNeighbors(metric=metric, **kwargs)
        self._neighbors_fitted = False
        super().__init__(
            embedding=embedding,
            persist_path=persist_path,
            serializer=serializer
        )

        # algorithm specific properties

    def _update(self) -> None:
        super()._update()
        self._neighbors.fit(self._embeddings_np)
        self._neighbors_fitted = True


    def _similarity_index_search_with_score(
        self, query_embedding: List[float], *, k: int = DEFAULT_K, filter: Optional[dict] = None,
        **kwargs: Any
    ) -> List[Tuple[int, float]]:
        query_embedding = self._np.asarray(query_embedding)
        if filter is not None:
            mask = self._get_filtered_data(filter)
            clf = type(self._neighbors)(**self._neighbors.get_params()).fit(self._embeddings_np[mask])
            mask_indices = self._np.arange(len(self._embeddings_np))[mask]
        else:
            clf = self._neighbors
            if not self._neighbors_fitted:
                raise SKLearnVectorStoreException(
                    "No data was added to SKLearnVectorStore."
                )
            mask_indices = self._np.arange(len(self._embeddings_np))
        neigh_dists, neigh_idxs = clf.kneighbors(
            [query_embedding], n_neighbors=k
        )
        indices = [int(mask_indices[int(idx)]) for idx in neigh_idxs[0]]
       
        return list(zip(indices, neigh_dists[0]))

SKLearnVectorStore = SKLearnKNNVectorStore # for backwards-compatibility


class SKLearnSVMVectorStore(SKLearnVectorStore):
    """A simple in-memory vector store based on the scikit-learn library
    LinearSVC implementation."""

    def __init__(
        self,
        embedding: Embeddings,
        *,
        persist_path: Optional[str] = None,
        serializer: Literal["json", "bson", "parquet"] = "json",
        svm_c: float = 0.1,
        max_itr=10000,
        tol=1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            embedding=embedding,
            persist_path=persist_path,
            serializer=serializer
        )
        # algorithm specific properties
        sklearn_svm = guard_import("sklearn.svm", pip_name="scikit-learn")
        self._svm = sklearn_svm.LinearSVC(class_weight='balanced', max_iter=max_itr, tol=tol, C=svm_c, **kwargs)

    def _similarity_index_search_with_score(
        self, query_embedding: List[float], *, k: int = DEFAULT_K, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Tuple[int, float]]:
        query_embedding = self._np.asarray(query_embedding)
        mask = self._get_filtered_data(filter)
        masked_indices = self._np.arange(len(self._embeddings_np))[mask]
        x = self._np.concatenate([query_embedding[None,...], self._embeddings_np[mask]]) # x is (1001, 1536) array, with query now as the first row
        y = self._np.zeros(len(x))
        y[0] = 1
        self._svm.fit(x, y) # train
        similarities = self._svm.decision_function(x)[1:]
        sorted_indices = self._np.argsort(-similarities)
        dist = similarities[:k]
        indices = [masked_indices[idx] for idx in sorted_indices]
        return  list(zip(indices, dist))

