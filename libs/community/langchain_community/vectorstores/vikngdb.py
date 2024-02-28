from __future__ import annotations

import logging
import uuid
from typing import Any, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

logger = logging.getLogger(__name__)


class VikingDBConfig(object):
    def __init__(self, host="host", region="region", ak="ak", sk="sk", scheme="http"):  # type: ignore[no-untyped-def]
        self.host = host
        self.region = region
        self.ak = ak
        self.sk = sk
        self.scheme = scheme


class VikingDB(VectorStore):
    def __init__(
        self,
        embedding_function: Embeddings,
        collection_name: str = "LangChainCollection",
        connection_args: Optional[VikingDBConfig] = None,
        index_params: Optional[dict] = None,
        drop_old: Optional[bool] = False,
        **kwargs: Any,
    ):
        try:
            from volcengine.viking_db import Collection, VikingDBService
        except ImportError:
            raise ValueError(
                "Could not import volcengine python package. "
                "Please install it with `pip install --upgrade volcengine`."
            )
        self.embedding_func = embedding_function
        self.collection_name = collection_name
        self.index_name = "LangChainIndex"
        self.connection_args = connection_args
        self.index_params = index_params
        self.drop_old = drop_old
        self.service = VikingDBService(
            connection_args.host,  # type: ignore[union-attr]
            connection_args.region,  # type: ignore[union-attr]
            connection_args.ak,  # type: ignore[union-attr]
            connection_args.sk,  # type: ignore[union-attr]
            connection_args.scheme,  # type: ignore[union-attr]
        )

        try:
            col = self.service.get_collection(collection_name)
        except Exception:
            col = None
        self.collection = col
        self.index = None
        if self.collection is not None:
            self.index = self.service.get_index(self.collection_name, self.index_name)

        if drop_old and isinstance(self.collection, Collection):
            indexes = self.service.list_indexes(collection_name)
            for index in indexes:
                self.service.drop_index(collection_name, index.index_name)
            self.service.drop_collection(collection_name)
            self.collection = None
            self.index = None

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_func

    def _create_collection(
        self, embeddings: List, metadatas: Optional[List[dict]] = None
    ) -> None:
        try:
            from volcengine.viking_db import Field, FieldType
        except ImportError:
            raise ValueError(
                "Could not import volcengine python package. "
                "Please install it with `pip install --upgrade volcengine`."
            )
        dim = len(embeddings[0])
        fields = []
        if metadatas:
            for key, value in metadatas[0].items():
                # print(key, value)
                if isinstance(value, str):
                    fields.append(Field(key, FieldType.String))
                if isinstance(value, int):
                    fields.append(Field(key, FieldType.Int64))
                if isinstance(value, bool):
                    fields.append(Field(key, FieldType.Bool))
                if isinstance(value, list) and all(
                    isinstance(item, str) for item in value
                ):
                    fields.append(Field(key, FieldType.List_String))
                if isinstance(value, list) and all(
                    isinstance(item, int) for item in value
                ):
                    fields.append(Field(key, FieldType.List_Int64))
        fields.append(Field("text", FieldType.String))

        fields.append(Field("primary_key", FieldType.String, is_primary_key=True))

        fields.append(Field("vector", FieldType.Vector, dim=dim))

        self.collection = self.service.create_collection(self.collection_name, fields)

    def _create_index(self) -> None:
        try:
            from volcengine.viking_db import VectorIndexParams
        except ImportError:
            raise ValueError(
                "Could not import volcengine python package. "
                "Please install it with `pip install --upgrade volcengine`."
            )
        cpu_quota = 2
        vector_index = VectorIndexParams()
        partition_by = ""
        scalar_index = None
        if self.index_params is not None:
            if self.index_params.get("cpu_quota") is not None:
                cpu_quota = self.index_params["cpu_quota"]
            if self.index_params.get("vector_index") is not None:
                vector_index = self.index_params["vector_index"]
            if self.index_params.get("partition_by") is not None:
                partition_by = self.index_params["partition_by"]
            if self.index_params.get("scalar_index") is not None:
                scalar_index = self.index_params["scalar_index"]

        self.index = self.service.create_index(
            self.collection_name,
            self.index_name,
            vector_index=vector_index,
            cpu_quota=cpu_quota,
            partition_by=partition_by,
            scalar_index=scalar_index,
        )

    def add_texts(  # type: ignore[override]
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> List[str]:
        try:
            from volcengine.viking_db import Data
        except ImportError:
            raise ValueError(
                "Could not import volcengine python package. "
                "Please install it with `pip install --upgrade volcengine`."
            )
        texts = list(texts)
        try:
            embeddings = self.embedding_func.embed_documents(texts)
        except NotImplementedError:
            embeddings = [self.embedding_func.embed_query(x) for x in texts]
        if len(embeddings) == 0:
            logger.debug("Nothing to insert, skipping.")
            return []
        if self.collection is None:
            self._create_collection(embeddings, metadatas)
            self._create_index()

        # insert data
        data = []
        pks: List[str] = []
        for index in range(len(embeddings)):
            primary_key = str(uuid.uuid4())
            pks.append(primary_key)
            field = {
                "text": texts[index],
                "primary_key": primary_key,
                "vector": embeddings[index],
            }
            if metadatas is not None and index < len(metadatas):
                names = list(metadatas[index].keys())
                for name in names:
                    field[name] = metadatas[index].get(name)  # type: ignore[assignment]
            data.append(Data(field))

        total_count = len(data)
        for i in range(0, total_count, batch_size):
            end = min(i + batch_size, total_count)
            insert_data = data[i:end]
            # print(insert_data)
            self.collection.upsert_data(insert_data)  # type: ignore[union-attr]
        return pks

    def similarity_search(  # type: ignore[override]
        self,
        query: str,
        params: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        res = self.similarity_search_with_score(query=query, params=params, **kwargs)
        return [doc for doc, _ in res]

    def similarity_search_with_score(
        self,
        query: str,
        params: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding_func.embed_query(query)

        res = self.similarity_search_with_score_by_vector(
            embedding=embedding, params=params, **kwargs
        )
        return res

    def similarity_search_by_vector(  # type: ignore[override]
        self,
        embedding: List[float],
        params: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        res = self.similarity_search_with_score_by_vector(
            embedding=embedding, params=params, **kwargs
        )
        return [doc for doc, _ in res]

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        params: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        if self.collection is None:
            logger.debug("No existing collection to search.")
            return []

        filter = None
        limit = 10
        output_fields = None
        partition = "default"
        if params is not None:
            if params.get("filter") is not None:
                filter = params["filter"]
            if params.get("limit") is not None:
                limit = params["limit"]
            if params.get("output_fields") is not None:
                output_fields = params["output_fields"]
            if params.get("partition") is not None:
                partition = params["partition"]

        res = self.index.search_by_vector(  # type: ignore[union-attr]
            embedding,
            filter=filter,
            limit=limit,
            output_fields=output_fields,
            partition=partition,
        )

        ret = []
        for item in res:
            item.fields.pop("primary_key")
            item.fields.pop("vector")
            page_content = item.fields.pop("text")
            doc = Document(page_content=page_content, metadata=item.fields)
            pair = (doc, item.score)
            ret.append(pair)
        return ret

    def max_marginal_relevance_search(  # type: ignore[override]
        self,
        query: str,
        k: int = 4,
        lambda_mult: float = 0.5,
        params: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        embedding = self.embedding_func.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            lambda_mult=lambda_mult,
            params=params,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(  # type: ignore[override]
        self,
        embedding: List[float],
        k: int = 4,
        lambda_mult: float = 0.5,
        params: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        if self.collection is None:
            logger.debug("No existing collection to search.")
            return []
        filter = None
        limit = 10
        output_fields = None
        partition = "default"
        if params is not None:
            if params.get("filter") is not None:
                filter = params["filter"]
            if params.get("limit") is not None:
                limit = params["limit"]
            if params.get("output_fields") is not None:
                output_fields = params["output_fields"]
            if params.get("partition") is not None:
                partition = params["partition"]

        res = self.index.search_by_vector(  # type: ignore[union-attr]
            embedding,
            filter=filter,
            limit=limit,
            output_fields=output_fields,
            partition=partition,
        )
        documents = []
        ordered_result_embeddings = []
        for item in res:
            ordered_result_embeddings.append(item.fields.pop("vector"))
            item.fields.pop("primary_key")
            page_content = item.fields.pop("text")
            doc = Document(page_content=page_content, metadata=item.fields)
            documents.append(doc)

        new_ordering = maximal_marginal_relevance(
            np.array(embedding), ordered_result_embeddings, k=k, lambda_mult=lambda_mult
        )
        # Reorder the values and return.
        ret = []
        for x in new_ordering:
            # Function can return -1 index
            if x == -1:
                break
            else:
                ret.append(documents[x])
        return ret

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        if self.collection is None:
            logger.debug("No existing collection to search.")
        self.collection.delete_data(ids)  # type: ignore[union-attr]

    @classmethod
    def from_texts(  # type: ignore[no-untyped-def, override]
        cls,
        texts: List[str],
        embedding: Embeddings,
        connection_args: Optional[VikingDBConfig] = None,
        metadatas: Optional[List[dict]] = None,
        collection_name: str = "LangChainCollection",
        index_params: Optional[dict] = None,
        drop_old: bool = False,
        **kwargs: Any,
    ):
        if connection_args is None:
            raise Exception("VikingDBConfig does not exists")
        vector_db = cls(
            embedding_function=embedding,
            collection_name=collection_name,
            connection_args=connection_args,
            index_params=index_params,
            drop_old=drop_old,
            **kwargs,
        )
        vector_db.add_texts(texts=texts, metadatas=metadatas)
        return vector_db
