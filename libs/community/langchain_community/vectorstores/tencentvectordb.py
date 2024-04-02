"""Wrapper around the Tencent vector database."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import guard_import
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

logger = logging.getLogger(__name__)


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
        index_params: IndexParams = IndexParams(128),
        database_name: str = "LangChainDatabase",
        collection_name: str = "LangChainCollection",
        drop_old: Optional[bool] = False,
    ):
        self.document = guard_import("tcvectordb.model.document")
        tcvectordb = guard_import("tcvectordb")
        self.embedding_func = embedding
        self.index_params = index_params
        self.vdb_client = tcvectordb.VectorDBClient(
            url=connection_params.url,
            username=connection_params.username,
            key=connection_params.key,
            timeout=connection_params.timeout,
        )
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
        index_type = None
        for k, v in enum.IndexType.__members__.items():
            if k == self.index_params.index_type:
                index_type = v
        if index_type is None:
            raise ValueError("unsupported index_type")
        metric_type = None
        for k, v in enum.MetricType.__members__.items():
            if k == self.index_params.metric_type:
                metric_type = v
        if metric_type is None:
            raise ValueError("unsupported metric_type")
        if self.index_params.params is None:
            params = vdb_index.HNSWParams(m=16, efconstruction=200)
        else:
            params = vdb_index.HNSWParams(
                m=self.index_params.params.get("M", 16),
                efconstruction=self.index_params.params.get("efConstruction", 200),
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
            vdb_index.FilterIndex(
                self.field_metadata, enum.FieldType.String, enum.IndexType.FILTER
            ),
        )
        self.collection = self.database.create_collection(
            name=collection_name,
            shard=self.index_params.shard,
            replicas=self.index_params.replicas,
            description="Collection for LangChain",
            index=index,
        )

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_func

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
        **kwargs: Any,
    ) -> TencentVectorDB:
        """Create a collection, indexes it with HNSW, and insert data."""
        if len(texts) == 0:
            raise ValueError("texts is empty")
        if connection_params is None:
            raise ValueError("connection_params is empty")
        try:
            embeddings = embedding.embed_documents(texts[0:1])
        except NotImplementedError:
            embeddings = [embedding.embed_query(texts[0])]
        dimension = len(embeddings[0])
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
        )
        vector_db.add_texts(texts=texts, metadatas=metadatas)
        return vector_db

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        timeout: Optional[int] = None,
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> List[str]:
        """Insert text data into TencentVectorDB."""
        texts = list(texts)
        try:
            embeddings = self.embedding_func.embed_documents(texts)
        except NotImplementedError:
            embeddings = [self.embedding_func.embed_query(x) for x in texts]
        if len(embeddings) == 0:
            logger.debug("Nothing to insert, skipping.")
            return []
        pks: list[str] = []
        total_count = len(embeddings)
        for start in range(0, total_count, batch_size):
            # Grab end index
            docs = []
            end = min(start + batch_size, total_count)
            for id in range(start, end, 1):
                metadata = "{}"
                if metadatas is not None:
                    metadata = json.dumps(metadatas[id])
                doc = self.document.Document(
                    id="{}-{}-{}".format(time.time_ns(), hash(texts[id]), id),
                    vector=embeddings[id],
                    text=texts[id],
                    metadata=metadata,
                )
                docs.append(doc)
                pks.append(str(id))
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
        embedding = self.embedding_func.embed_query(query)
        res = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, param=param, expr=expr, timeout=timeout, **kwargs
        )
        return res

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
        res = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, param=param, expr=expr, timeout=timeout, **kwargs
        )
        return [doc for doc, _ in res]

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score."""
        filter = None if expr is None else self.document.Filter(expr)
        ef = 10 if param is None else param.get("ef", 10)
        res: List[List[Dict]] = self.collection.search(
            vectors=[embedding],
            filter=filter,
            params=self.document.HNSWSearchParams(ef=ef),
            retrieve_vector=False,
            limit=k,
            timeout=timeout,
        )
        # Organize results.
        ret: List[Tuple[Document, float]] = []
        if res is None or len(res) == 0:
            return ret
        for result in res[0]:
            meta = result.get(self.field_metadata)
            if meta is not None:
                meta = json.loads(meta)
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

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a search and return results that are reordered by MMR."""
        filter = None if expr is None else self.document.Filter(expr)
        ef = 10 if param is None else param.get("ef", 10)
        res: List[List[Dict]] = self.collection.search(
            vectors=[embedding],
            filter=filter,
            params=self.document.HNSWSearchParams(ef=ef),
            retrieve_vector=True,
            limit=fetch_k,
            timeout=timeout,
        )
        # Organize results.
        documents = []
        ordered_result_embeddings = []
        for result in res[0]:
            meta = result.get(self.field_metadata)
            if meta is not None:
                meta = json.loads(meta)
            doc = Document(page_content=result.get(self.field_text), metadata=meta)  # type: ignore[arg-type]
            documents.append(doc)
            ordered_result_embeddings.append(result.get(self.field_vector))
        # Get the new order of results.
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
