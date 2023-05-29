""" Wrapper around scikit-learn NearestNeighbors implementation.

The vector store can be persisted in json, bson or parquet format.
"""

import importlib
import json
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Type
from uuid import uuid4

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore


def guard_import(
    module_name: str, *, pip_name: Optional[str] = None, package: Optional[str] = None
) -> Any:
    """Dynamically imports a module and raises a helpful exception if the module is not
    installed."""
    try:
        module = importlib.import_module(module_name, package)
    except ImportError:
        raise ImportError(
            f"Could not import {module_name} python package. "
            f"Please install it with `pip install {pip_name or module_name}`."
        )
    return module


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


class SKLearnVectorStore(VectorStore):
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
        np = guard_import("numpy")
        sklearn_neighbors = guard_import("sklearn.neighbors", pip_name="scikit-learn")

        # non-persistent properties
        self._np = np
        self._neighbors = sklearn_neighbors.NearestNeighbors(metric=metric, **kwargs)
        self._neighbors_fitted = False
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
        self._update_neighbors()

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
        self._update_neighbors()
        return _ids

    def _update_neighbors(self) -> None:
        if len(self._embeddings) == 0:
            raise SKLearnVectorStoreException(
                "No data was added to SKLearnVectorStore."
            )
        self._embeddings_np = self._np.asarray(self._embeddings)
        self._neighbors.fit(self._embeddings_np)
        self._neighbors_fitted = True

    def similarity_search_with_score(
        self, query: str, *, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        if not self._neighbors_fitted:
            raise SKLearnVectorStoreException(
                "No data was added to SKLearnVectorStore."
            )
        query_embedding = self._embedding_function.embed_query(query)
        neigh_dists, neigh_idxs = self._neighbors.kneighbors(
            [query_embedding], n_neighbors=k
        )
        res = []
        for idx, dist in zip(neigh_idxs[0], neigh_dists[0]):
            _idx = int(idx)
            metadata = {"id": self._ids[_idx], **self._metadatas[_idx]}
            doc = Document(page_content=self._texts[_idx], metadata=metadata)
            res.append((doc, dist))
        return res

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        docs_scores = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in docs_scores]

    def _similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        docs_dists = self.similarity_search_with_score(query=query, k=k, **kwargs)
        docs, dists = zip(*docs_dists)
        scores = [1 / math.exp(dist) for dist in dists]
        return list(zip(list(docs), scores))

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
