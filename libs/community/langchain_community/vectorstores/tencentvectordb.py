"""Wrapper around the Tencent vector database."""

from __future__ import annotations

import json
import logging
import time
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils import guard_import
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

logger = logging.getLogger(__name__)


META_FIELD_TYPE_UINT64 = "uint64"
META_FIELD_TYPE_STRING = "string"
META_FIELD_TYPE_ARRAY = "array"
META_FIELD_TYPE_VECTOR = "vector"

META_FIELD_TYPES = [
    META_FIELD_TYPE_UINT64,
    META_FIELD_TYPE_STRING,
    META_FIELD_TYPE_ARRAY,
    META_FIELD_TYPE_VECTOR,
]


class ConnectionParams:
    """Tencent vector DB Connection params.

    See the following documentation for details:
    https://cloud.tencent.com/document/product/1709/95820

    Attribute:
        url (str) : The access address of the vector database server
            that the client needs to connect to.
        key (str): API key for client to access the vector database server,
            which is used for authentication.
        username (str) : Account for client to access the vector database server.
        timeout (int) : Request Timeout.
    """

    def __init__(self, url: str, key: str, username: str = "root", timeout: int = 10):
        self.url = url
        self.key = key
        self.username = username
        self.timeout = timeout


class IndexParams:
    """Tencent vector DB Index params.

    See the following documentation for details:
    https://cloud.tencent.com/document/product/1709/95826
    """

    def __init__(
        self,
        dimension: int,
        shard: int = 1,
        replicas: int = 2,
        index_type: str = "HNSW",
        metric_type: str = "L2",
        params: Optional[Dict] = None,
    ):
        self.dimension = dimension
        self.shard = shard
        self.replicas = replicas
        self.index_type = index_type
        self.metric_type = metric_type
        self.params = params


class MetaField(BaseModel):
    """MetaData Field for Tencent vector DB."""

    name: str
    description: Optional[str]
    data_type: Union[str, Enum]
    index: bool = False

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        enum = guard_import("tcvectordb.model.enum")
        if isinstance(self.data_type, str):
            if self.data_type not in META_FIELD_TYPES:
                raise ValueError(f"unsupported data_type {self.data_type}")
            target = [
                fe
                for fe in enum.FieldType
                if fe.value.lower() == self.data_type.lower()
            ]
            if target:
                self.data_type = target[0]
            else:
                raise ValueError(f"unsupported data_type {self.data_type}")
        else:
            if self.data_type not in enum.FieldType:
                raise ValueError(f"unsupported data_type {self.data_type}")


def translate_filter(
    lc_filter: str, allowed_fields: Optional[Sequence[str]] = None
) -> str:
    """Translate LangChain filter to Tencent VectorDB filter.

    Args:
        lc_filter (str): LangChain filter.
        allowed_fields (Optional[Sequence[str]]): Allowed fields for filter.

    Returns:
        str: Translated filter.
    """
    from langchain.chains.query_constructor.base import fix_filter_directive
    from langchain.chains.query_constructor.parser import get_parser
    from langchain.retrievers.self_query.tencentvectordb import (
        TencentVectorDBTranslator,
    )
    from langchain_core.structured_query import FilterDirective

    tvdb_visitor = TencentVectorDBTranslator(allowed_fields)
    flt = cast(
        Optional[FilterDirective],
        get_parser(
            allowed_comparators=tvdb_visitor.allowed_comparators,
            allowed_operators=tvdb_visitor.allowed_operators,
            allowed_attributes=allowed_fields,
        ).parse(lc_filter),
    )
    flt = fix_filter_directive(flt)
    return flt.accept(tvdb_visitor) if flt else ""


class TencentVectorDB(VectorStore):
    """Tencent VectorDB as a vector store.

    In order to use this you need to have a database instance.
    See the following documentation for details:
    https://cloud.tencent.com/document/product/1709/94951
    """

    field_id: str = "id"
    field_vector: str = "vector"
    field_text: str = "text"
    field_metadata: str = "metadata"

    def __init__(
        self,
        embedding: Embeddings,
        connection_params: ConnectionParams,
        index_params: IndexParams = IndexParams(768),
        database_name: str = "LangChainDatabase",
        collection_name: str = "LangChainCollection",
        drop_old: Optional[bool] = False,
        collection_description: Optional[str] = "Collection for LangChain",
        meta_fields: Optional[List[MetaField]] = None,
        t_vdb_embedding: Optional[str] = "bge-base-zh",
    ):
        self.document = guard_import("tcvectordb.model.document")
        tcvectordb = guard_import("tcvectordb")
        tcollection = guard_import("tcvectordb.model.collection")
        enum = guard_import("tcvectordb.model.enum")

        if t_vdb_embedding:
            embedding_model = [
                model
                for model in enum.EmbeddingModel
                if t_vdb_embedding == model.model_name
            ]
            if not any(embedding_model):
                raise ValueError(
                    f"embedding model `{t_vdb_embedding}` is invalid. "
                    f"choices: {[member.model_name for member in enum.EmbeddingModel]}"
                )
            self.embedding_model = tcollection.Embedding(
                vector_field="vector", field="text", model=embedding_model[0]
            )
        self.embedding_func = embedding
        self.index_params = index_params
        self.collection_description = collection_description
        self.vdb_client = tcvectordb.VectorDBClient(
            url=connection_params.url,
            username=connection_params.username,
            key=connection_params.key,
            timeout=connection_params.timeout,
        )
        self.meta_fields = meta_fields
        db_list = self.vdb_client.list_databases()
        db_exist: bool = False
        for db in db_list:
            if database_name == db.database_name:
                db_exist = True
                break
        if db_exist:
            self.database = self.vdb_client.database(database_name)
        else:
            self.database = self.vdb_client.create_database(database_name)
        try:
            self.collection = self.database.describe_collection(collection_name)
            if drop_old:
                self.database.drop_collection(collection_name)
                self._create_collection(collection_name)
        except tcvectordb.exceptions.VectorDBException:
            self._create_collection(collection_name)

    def _create_collection(self, collection_name: str) -> None:
        enum = guard_import("tcvectordb.model.enum")
        vdb_index = guard_import("tcvectordb.model.index")

        index_type = enum.IndexType.__members__.get(self.index_params.index_type)
        if index_type is None:
            raise ValueError("unsupported index_type")
        metric_type = enum.MetricType.__members__.get(self.index_params.metric_type)
        if metric_type is None:
            raise ValueError("unsupported metric_type")
        params = vdb_index.HNSWParams(
            m=(self.index_params.params or {}).get("M", 16),
            efconstruction=(self.index_params.params or {}).get("efConstruction", 200),
        )

        index = vdb_index.Index(
            vdb_index.FilterIndex(
                self.field_id, enum.FieldType.String, enum.IndexType.PRIMARY_KEY
            ),
            vdb_index.VectorIndex(
                self.field_vector,
                self.index_params.dimension,
                index_type,
                metric_type,
                params,
            ),
            vdb_index.FilterIndex(
                self.field_text, enum.FieldType.String, enum.IndexType.FILTER
            ),
        )
        # Add metadata indexes
        if self.meta_fields is not None:
            index_meta_fields = [field for field in self.meta_fields if field.index]
            for field in index_meta_fields:
                ft_index = vdb_index.FilterIndex(
                    field.name, field.data_type, enum.IndexType.FILTER
                )
                index.add(ft_index)
        else:
            index.add(
                vdb_index.FilterIndex(
                    self.field_metadata, enum.FieldType.String, enum.IndexType.FILTER
                )
            )
        self.collection = self.database.create_collection(
            name=collection_name,
            shard=self.index_params.shard,
            replicas=self.index_params.replicas,
            description=self.collection_description,
            index=index,
            embedding=self.embedding_model,
        )

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_func

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter_expr: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete documents from the collection."""
        delete_attrs = {}
        if ids:
            delete_attrs["ids"] = ids
        if filter_expr:
            delete_attrs["filter"] = self.document.Filter(filter_expr)
        self.collection.delete(**delete_attrs)
        return True

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        connection_params: Optional[ConnectionParams] = None,
        index_params: Optional[IndexParams] = None,
        database_name: str = "LangChainDatabase",
        collection_name: str = "LangChainCollection",
        drop_old: Optional[bool] = False,
        collection_description: Optional[str] = "Collection for LangChain",
        meta_fields: Optional[List[MetaField]] = None,
        t_vdb_embedding: Optional[str] = "bge-base-zh",
        **kwargs: Any,
    ) -> TencentVectorDB:
        """Create a collection, indexes it with HNSW, and insert data."""
        if len(texts) == 0:
            raise ValueError("texts is empty")
        if connection_params is None:
            raise ValueError("connection_params is empty")
        enum = guard_import("tcvectordb.model.enum")
        if embedding is None and t_vdb_embedding is None:
            raise ValueError("embedding and t_vdb_embedding cannot be both None")
        if embedding:
            embeddings = embedding.embed_documents(texts[0:1])
            dimension = len(embeddings[0])
        else:
            embedding_model = [
                model
                for model in enum.EmbeddingModel
                if t_vdb_embedding == model.model_name
            ]
            if not any(embedding_model):
                raise ValueError(
                    f"embedding model `{t_vdb_embedding}` is invalid. "
                    f"choices: {[member.model_name for member in enum.EmbeddingModel]}"
                )
            dimension = embedding_model[0]._EmbeddingModel__dimensions
        if index_params is None:
            index_params = IndexParams(dimension=dimension)
        else:
            index_params.dimension = dimension
        vector_db = cls(
            embedding=embedding,
            connection_params=connection_params,
            index_params=index_params,
            database_name=database_name,
            collection_name=collection_name,
            drop_old=drop_old,
            collection_description=collection_description,
            meta_fields=meta_fields,
            t_vdb_embedding=t_vdb_embedding,
        )
        vector_db.add_texts(texts=texts, metadatas=metadatas)
        return vector_db

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        timeout: Optional[int] = None,
        batch_size: int = 1000,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Insert text data into TencentVectorDB."""
        texts = list(texts)
        if len(texts) == 0:
            logger.debug("Nothing to insert, skipping.")
            return []
        if self.embedding_func:
            embeddings = self.embedding_func.embed_documents(texts)
        else:
            embeddings = []
        pks: list[str] = []
        total_count = len(texts)
        for start in range(0, total_count, batch_size):
            # Grab end index
            docs = []
            end = min(start + batch_size, total_count)
            for id in range(start, end, 1):
                metadata = (
                    self._get_meta(metadatas[id]) if metadatas and metadatas[id] else {}
                )
                doc_id = ids[id] if ids else None
                doc_attrs: Dict[str, Any] = {
                    "id": doc_id
                    or "{}-{}-{}".format(time.time_ns(), hash(texts[id]), id)
                }
                if embeddings:
                    doc_attrs["vector"] = embeddings[id]
                else:
                    doc_attrs["text"] = texts[id]
                doc_attrs.update(metadata)
                doc = self.document.Document(**doc_attrs)
                docs.append(doc)
                pks.append(doc_attrs["id"])
            self.collection.upsert(docs, timeout)
        return pks

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search against the query string."""
        res = self.similarity_search_with_score(
            query=query, k=k, param=param, expr=expr, timeout=timeout, **kwargs
        )
        return [doc for doc, _ in res]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score."""
        # Embed the query text.
        if self.embedding_func:
            embedding = self.embedding_func.embed_query(query)
            return self.similarity_search_with_score_by_vector(
                embedding=embedding,
                k=k,
                param=param,
                expr=expr,
                timeout=timeout,
                **kwargs,
            )
        return self.similarity_search_with_score_by_vector(
            embedding=[],
            k=k,
            param=param,
            expr=expr,
            timeout=timeout,
            query=query,
            **kwargs,
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search against the query string."""
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, param=param, expr=expr, timeout=timeout, **kwargs
        )
        return [doc for doc, _ in docs]

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        filter: Optional[str] = None,
        timeout: Optional[int] = None,
        query: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score."""
        if filter and not expr:
            expr = translate_filter(
                filter, [f.name for f in (self.meta_fields or []) if f.index]
            )
        search_args = {
            "filter": self.document.Filter(expr) if expr else None,
            "params": self.document.HNSWSearchParams(ef=(param or {}).get("ef", 10)),
            "retrieve_vector": False,
            "limit": k,
            "timeout": timeout,
        }
        if query:
            search_args["embeddingItems"] = [query]
            res: List[List[Dict]] = self.collection.searchByText(**search_args).get(
                "documents"
            )
        else:
            search_args["vectors"] = [embedding]
            res = self.collection.search(**search_args)

        ret: List[Tuple[Document, float]] = []
        if res is None or len(res) == 0:
            return ret
        for result in res[0]:
            meta = self._get_meta(result)
            doc = Document(page_content=result.get(self.field_text), metadata=meta)  # type: ignore[arg-type]
            pair = (doc, result.get("score", 0.0))
            ret.append(pair)
        return ret

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a search and return results that are reordered by MMR."""
        if self.embedding_func:
            embedding = self.embedding_func.embed_query(query)
            return self.max_marginal_relevance_search_by_vector(
                embedding=embedding,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                param=param,
                expr=expr,
                timeout=timeout,
                **kwargs,
            )
        # tvdb will do the query embedding
        docs = self.similarity_search_with_score(
            query=query, k=fetch_k, param=param, expr=expr, timeout=timeout, **kwargs
        )
        return [doc for doc, _ in docs]

    def _get_meta(self, result: Dict) -> Dict:
        """Get metadata from the result."""

        if self.meta_fields:
            return {field.name: result.get(field.name) for field in self.meta_fields}
        elif result.get(self.field_metadata):
            raw_meta = result.get(self.field_metadata)
            if raw_meta and isinstance(raw_meta, str):
                return json.loads(raw_meta)
        return {}

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        filter: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a search and return results that are reordered by MMR."""
        if filter and not expr:
            expr = translate_filter(
                filter, [f.name for f in (self.meta_fields or []) if f.index]
            )
        res: List[List[Dict]] = self.collection.search(
            vectors=[embedding],
            filter=self.document.Filter(expr) if expr else None,
            params=self.document.HNSWSearchParams(ef=(param or {}).get("ef", 10)),
            retrieve_vector=True,
            limit=fetch_k,
            timeout=timeout,
        )
        # Organize results.
        documents = []
        ordered_result_embeddings = []
        for result in res[0]:
            meta = self._get_meta(result)
            doc = Document(page_content=result.get(self.field_text), metadata=meta)  # type: ignore[arg-type]
            documents.append(doc)
            ordered_result_embeddings.append(result.get(self.field_vector))
        # Get the new order of results.
        new_ordering = maximal_marginal_relevance(
            np.array(embedding), ordered_result_embeddings, k=k, lambda_mult=lambda_mult
        )
        # Reorder the values and return.
        return [documents[x] for x in new_ordering if x != -1]
