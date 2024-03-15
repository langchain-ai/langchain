"""Wrapper around the Baidu vector database."""
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
    """Baidu VectorDB Connection params.

    See the following documentation for details:
    https://cloud.baidu.com/doc/VDB/s/6lrsob0wy

    Attribute:
        endpoint (str) : The access address of the vector database server
            that the client needs to connect to.
        api_key (str): API key for client to access the vector database server,
            which is used for authentication.
        account (str) : Account for client to access the vector database server.
        connection_timeout_in_mills (int) : Request Timeout.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        account: str = "root",
        connection_timeout_in_mills: int = 50 * 1000,
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.account = account
        self.connection_timeout_in_mills = connection_timeout_in_mills


class TableParams:
    """Baidu VectorDB table params.

    See the following documentation for details:
    https://cloud.baidu.com/doc/VDB/s/mlrsob0p6
    """

    def __init__(
        self,
        dimension: int,
        replication: int = 3,
        partition: int = 1,
        index_type: str = "HNSW",
        metric_type: str = "L2",
        params: Optional[Dict] = None,
    ):
        self.dimension = dimension
        self.replication = replication
        self.partition = partition
        self.index_type = index_type
        self.metric_type = metric_type
        self.params = params


class BaiduVectorDB(VectorStore):
    """Baidu VectorDB as a vector store.

    In order to use this you need to have a database instance.
    See the following documentation for details:
    https://cloud.baidu.com/doc/VDB/index.html
    """

    field_id: str = "id"
    field_vector: str = "vector"
    field_text: str = "text"
    field_metadata: str = "metadata"

    index_vector: str = "vector_idx"

    def __init__(
        self,
        embedding: Embeddings,
        connection_params: ConnectionParams,
        table_params: TableParams = TableParams(128),
        database_name: str = "LangChainDatabase",
        table_name: str = "LangChainTable",
        drop_old: Optional[bool] = False,
    ):
        pymochow = guard_import("pymochow")
        configuration = guard_import("pymochow.configuration")
        auth = guard_import("pymochow.auth.bce_credentials")
        self.mochowtable = guard_import("pymochow.model.table")
        self.mochowenum = guard_import("pymochow.model.enum")
        self.embedding_func = embedding
        self.table_params = table_params
        config = configuration.Configuration(
            credentials=auth.BceCredentials(
                connection_params.account, connection_params.api_key
            ),
            endpoint=connection_params.endpoint,
            connection_timeout_in_mills=connection_params.connection_timeout_in_mills,
        )
        self.vdb_client = pymochow.MochowClient(config)
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
            self.table = self.database.describe_table(table_name)
            if drop_old:
                self.database.drop_table(table_name)
                self._create_table(table_name)
        except pymochow.exception.ServerError:
            self._create_table(table_name)

    def _create_table(self, table_name: str) -> None:
        schema = guard_import("pymochow.model.schema")
        index_type = None
        for k, v in self.mochowenum.IndexType.__members__.items():
            if k == self.table_params.index_type:
                index_type = v
        if index_type is None:
            raise ValueError("unsupported index_type")
        metric_type = None
        for k, v in self.mochowenum.MetricType.__members__.items():
            if k == self.table_params.metric_type:
                metric_type = v
        if metric_type is None:
            raise ValueError("unsupported metric_type")
        if self.table_params.params is None:
            params = schema.HNSWParams(m=16, efconstruction=200)
        else:
            params = schema.HNSWParams(
                m=self.table_params.params.get("M", 16),
                efconstruction=self.table_params.params.get("efConstruction", 200),
            )
        fields = []
        fields.append(
            schema.Field(
                self.field_id,
                self.mochowenum.FieldType.STRING,
                primary_key=True,
                partition_key=True,
                auto_increment=False,
                not_null=True,
            )
        )
        fields.append(
            schema.Field(
                self.field_vector,
                self.mochowenum.FieldType.FLOAT_VECTOR,
                dimension=self.table_params.dimension,
            )
        )
        fields.append(schema.Field(self.field_text, self.mochowenum.FieldType.STRING))
        fields.append(
            schema.Field(self.field_metadata, self.mochowenum.FieldType.STRING)
        )
        indexes = []
        indexes.append(
            schema.VectorIndex(
                index_name=self.index_vector,
                index_type=index_type,
                field=self.field_vector,
                metric_type=metric_type,
                params=params,
            )
        )

        self.table = self.database.create_table(
            table_name=table_name,
            replication=self.table_params.replication,
            partition=self.mochowtable.Partition(
                partition_num=self.table_params.partition
            ),
            schema=schema.Schema(fields=fields, indexes=indexes),
        )

        while True:
            time.sleep(1)
            table = self.database.describe_table(table_name)
            if table.state == self.mochowenum.TableState.NORMAL:
                break

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
        table_params: Optional[TableParams] = None,
        database_name: str = "LangChainDatabase",
        table_name: str = "LangChainTable",
        drop_old: Optional[bool] = False,
        **kwargs: Any,
    ) -> BaiduVectorDB:
        """Create a table, indexes it with HNSW, and insert data."""
        if len(texts) == 0:
            raise ValueError("texts is empty")
        if connection_params is None:
            raise ValueError("connection_params is empty")
        try:
            embeddings = embedding.embed_documents(texts[0:1])
        except NotImplementedError:
            embeddings = [embedding.embed_query(texts[0])]
        dimension = len(embeddings[0])
        if table_params is None:
            table_params = TableParams(dimension=dimension)
        else:
            table_params.dimension = dimension
        vector_db = cls(
            embedding=embedding,
            connection_params=connection_params,
            table_params=table_params,
            database_name=database_name,
            table_name=table_name,
            drop_old=drop_old,
        )
        vector_db.add_texts(texts=texts, metadatas=metadatas)
        return vector_db

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> List[str]:
        """Insert text data into Baidu VectorDB."""
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
            rows = []
            end = min(start + batch_size, total_count)
            for id in range(start, end, 1):
                metadata = "{}"
                if metadatas is not None:
                    metadata = json.dumps(metadatas[id])
                row = self.mochowtable.Row(
                    id="{}-{}-{}".format(time.time_ns(), hash(texts[id]), id),
                    vector=[float(num) for num in embeddings[id]],
                    text=texts[id],
                    metadata=metadata,
                )
                rows.append(row)
                pks.append(str(id))
            self.table.upsert(rows=rows)
        # need rebuild vindex after upsert
        self.table.rebuild_index(self.index_vector)
        while True:
            time.sleep(2)
            index = self.table.describe_index(self.index_vector)
            if index.state == self.mochowenum.IndexState.NORMAL:
                break
        return pks

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search against the query string."""
        res = self.similarity_search_with_score(
            query=query, k=k, param=param, expr=expr, **kwargs
        )
        return [doc for doc, _ in res]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score."""
        # Embed the query text.
        embedding = self.embedding_func.embed_query(query)
        res = self._similarity_search_with_score(
            embedding=embedding, k=k, param=param, expr=expr, **kwargs
        )
        return res

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search against the query string."""
        res = self._similarity_search_with_score(
            embedding=embedding, k=k, param=param, expr=expr, **kwargs
        )
        return [doc for doc, _ in res]

    def _similarity_search_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score."""
        ef = 10 if param is None else param.get("ef", 10)

        anns = self.mochowtable.AnnSearch(
            vector_field=self.field_vector,
            vector_floats=[float(num) for num in embedding],
            params=self.mochowtable.HNSWSearchParams(ef=ef, limit=k),
            filter=expr,
        )
        res = self.table.search(anns=anns)

        rows = [[item] for item in res.rows]
        # Organize results.
        ret: List[Tuple[Document, float]] = []
        if rows is None or len(rows) == 0:
            return ret
        for row in rows:
            for result in row:
                row_data = result.get("row", {})
                meta = row_data.get(self.field_metadata)
                if meta is not None:
                    meta = json.loads(meta)
                doc = Document(
                    page_content=row_data.get(self.field_text), metadata=meta
                )
                pair = (doc, result.get("distance", 0.0))
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
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a search and return results that are reordered by MMR."""
        embedding = self.embedding_func.embed_query(query)
        return self._max_marginal_relevance_search(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            param=param,
            expr=expr,
            **kwargs,
        )

    def _max_marginal_relevance_search(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a search and return results that are reordered by MMR."""
        ef = 10 if param is None else param.get("ef", 10)
        anns = self.mochowtable.AnnSearch(
            vector_field=self.field_vector,
            vector_floats=[float(num) for num in embedding],
            params=self.mochowtable.HNSWSearchParams(ef=ef, limit=k),
            filter=expr,
        )
        res = self.table.search(anns=anns, retrieve_vector=True)

        # Organize results.
        documents: List[Document] = []
        ordered_result_embeddings = []
        rows = [[item] for item in res.rows]
        if rows is None or len(rows) == 0:
            return documents
        for row in rows:
            for result in row:
                row_data = result.get("row", {})
                meta = row_data.get(self.field_metadata)
                if meta is not None:
                    meta = json.loads(meta)
                doc = Document(
                    page_content=row_data.get(self.field_text), metadata=meta
                )
                documents.append(doc)
                ordered_result_embeddings.append(row_data.get(self.field_vector))
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
