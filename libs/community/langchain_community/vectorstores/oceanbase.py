import json
import logging
import math
import traceback
import uuid
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from sqlalchemy import JSON, Column, String, Table, func, text
from sqlalchemy.dialects.mysql import LONGTEXT

logger = logging.getLogger(__name__)

DEFAULT_OCEANBASE_CONNECTION = {
    "host": "localhost",
    "port": "2881",
    "user": "root@test",
    "password": "",
    "db_name": "test",
}
DEFAULT_OCEANBASE_VECTOR_TABLE_NAME = "langchain_vector"
DEFAULT_OCEANBASE_HNSW_BUILD_PARAM = {"M": 16, "efConstruction": 256}
DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM = {"efSearch": 64}
OCEANBASE_SUPPORTED_VECTOR_INDEX_TYPE = "HNSW"
DEFAULT_OCEANBASE_VECTOR_METRIC_TYPE = "l2"

DEFAULT_METADATA_FIELD = "metadata"


def _euclidean_similarity(distance: float) -> float:
    return 1.0 - distance / math.sqrt(2)


def _neg_inner_product_similarity(distance: float) -> float:
    return -distance


class OceanBase(VectorStore):
    """`OceanBase` vector store.

    You need to install `pyobvector` and run a standalone observer or OceanBase cluster.

    See the following documentation for how to deploy OceanBase:
    https://github.com/oceanbase/oceanbase-doc/blob/V4.3.1/en-US/400.deploy/500.deploy-oceanbase-database-community-edition/100.deployment-overview.md

    IF USING L2/INNER_PRODUCT metric, IT IS HIGHLY SUGGESTED TO NORMALIZE YOUR DATA.

    Args:
        embedding_function (Embeddings): Function used to embed the text.
        table_name (str): Which table name to use. Defaults to "langchain_vector".
        connection_args (Optional[dict[str, any]]): The connection args used for
            this class comes in the form of a dict. Refer to
            `DEFAULT_OCEANBASE_CONNECTION` for example.
        vidx_metric_type (str): Metric method of distance between vectors.
            This parameter takes values in `l2` and `inner_product`. Defaults to `l2`.
        vidx_algo_params (Optional[dict]): Which index params to use. Now OceanBase
            supports HNSW only. Refer to `DEFAULT_OCEANBASE_HNSW_BUILD_PARAM`
            for example.
        drop_old (bool): Whether to drop the current table. Defaults
            to False.
        primary_field (str): Name of the primary key column. Defaults to "id".
        vector_field (str): Name of the vector column. Defaults to "embedding".
        text_field (str): Name of the text column. Defaults to "document".
        metadata_field (Optional[str]): Name of the metadata column.
            Defaults to "metadata". When `metadata_field` is specified,
            the document's metadata will store as json.
        vidx_name (str): Name of the vector index table.
        partitions (ObPartition): Partition strategy of table. Refer to `pyobvector`'s
            documentation for more examples.
        extra_columns (Optional[List[Column]]): Extra sqlalchemy columns
            to add to the table.

    Example:
        .. code-block:: python

        from langchain_community.vectorstores import OceanBase
        from langchain_community.embeddings import OpenAIEmbeddings

        embedding = OpenAIEmbeddings()
        # Connect to a OceanBase instance on localhost
        ob = OceanBase(
            embedding_function=embedding,
            table_name="langchain_vector",
            connection_args={
                "host": "<YOUR-HOST>",
                "port": "<YOUR-PORT>",
                "user": "<YOUR-USER>",
                "password": "<YOUR-PASSWORD>",
                "db_name": "<YOUR-DBNAME>",
            },
            vidx_metric_type="l2",
            drop_old=True,
        )

    Raises:
        ValueError: If the pyobvector python package is not installed.
    """

    def __init__(  # type: ignore[no-untyped-def]
        self,
        embedding_function: Embeddings,
        table_name: str = DEFAULT_OCEANBASE_VECTOR_TABLE_NAME,
        connection_args: Optional[dict[str, Any]] = None,
        vidx_metric_type: str = DEFAULT_OCEANBASE_VECTOR_METRIC_TYPE,
        vidx_algo_params: Optional[dict] = None,
        drop_old: bool = False,
        *,
        primary_field: str = "id",
        vector_field: str = "embedding",
        text_field: str = "document",
        metadata_field: Optional[str] = DEFAULT_METADATA_FIELD,
        vidx_name: str = "vidx",
        partitions: Optional[Any] = None,
        extra_columns: Optional[List[Column]] = None,
        normalize: bool = False,
        **kwargs,
    ):
        """Initialize the OceanBase vector store."""

        self.embedding_function = embedding_function
        self.table_name = table_name
        self.connection_args = (
            connection_args
            if connection_args is not None
            else DEFAULT_OCEANBASE_CONNECTION
        )
        self.extra_columns = extra_columns
        self.normalize = normalize
        self._create_client(**kwargs)
        assert self.obvector is not None

        self.vidx_metric_type = vidx_metric_type.lower()
        if self.vidx_metric_type not in ("l2", "inner_product"):
            raise ValueError(
                "`vidx_metric_type` should be set in `l2`/`inner_product`."
            )

        self.vidx_algo_params = (
            vidx_algo_params
            if vidx_algo_params is not None
            else DEFAULT_OCEANBASE_HNSW_BUILD_PARAM
        )

        self.drop_old = drop_old
        self.primary_field = primary_field
        self.vector_field = vector_field
        self.text_field = text_field
        self.metadata_field = metadata_field or DEFAULT_METADATA_FIELD
        self.vidx_name = vidx_name
        self.partition = partitions
        self.hnsw_ef_search = -1

        if self.drop_old:
            self.obvector.drop_table_if_exist(self.table_name)

        if self.obvector.check_table_exists(self.table_name):
            self._load_table()

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def _create_client(self, **kwargs):  # type: ignore[no-untyped-def]
        try:
            from pyobvector import ObVecClient
        except ImportError:
            raise ImportError(
                "Could not import pyobvector package. "
                "Please install it with `pip install pyobvector`."
            )

        host = self.connection_args.get("host", "localhost")
        port = self.connection_args.get("port", "2881")
        user = self.connection_args.get("user", "root@test")
        password = self.connection_args.get("password", "")
        db_name = self.connection_args.get("db_name", "test")

        self.obvector = ObVecClient(
            uri=host + ":" + port,
            user=user,
            password=password,
            db_name=db_name,
            **kwargs,
        )

    def _load_table(self) -> None:
        table = Table(
            self.table_name,
            self.obvector.metadata_obj,
            autoload_with=self.obvector.engine,
        )
        column_names = [column.name for column in table.columns]
        optional_len = len(self.extra_columns or []) + 1
        assert len(column_names) == (3 + optional_len)

        logging.info(f"load exist table with {column_names} columns")
        self.primary_field = column_names[0]
        self.vector_field = column_names[1]
        self.text_field = column_names[2]
        self.metadata_field = column_names[3]

    def _create_table_with_index(self, embeddings: list) -> None:
        try:
            from pyobvector import VECTOR
        except ImportError:
            raise ImportError(
                "Could not import pyobvector package. "
                "Please install it with `pip install pyobvector`."
            )

        if self.obvector.check_table_exists(self.table_name):
            self._load_table()
            return

        dim = len(embeddings[0])
        cols = [
            Column(
                self.primary_field, String(4096), primary_key=True, autoincrement=False
            ),
            Column(self.vector_field, VECTOR(dim)),
            Column(self.text_field, LONGTEXT),
            Column(self.metadata_field, JSON),
        ]
        if self.extra_columns is not None:
            cols.extend(self.extra_columns)

        vidx_params = self.obvector.prepare_index_params()
        vidx_params.add_index(
            field_name=self.vector_field,
            index_type=OCEANBASE_SUPPORTED_VECTOR_INDEX_TYPE,
            index_name=self.vidx_name,
            metric_type=self.vidx_metric_type,
            params=self.vidx_algo_params,
        )

        self.obvector.create_table_with_index_params(
            table_name=self.table_name,
            columns=cols,
            indexes=None,
            vidxs=vidx_params,
            partitions=self.partition,
        )

    def _parse_metric_type_str_to_dist_func(self) -> Any:
        if self.vidx_metric_type == "l2":
            return func.l2_distance
        if self.vidx_metric_type == "cosine":
            return func.cosine_distance
        if self.vidx_metric_type == "inner_product":
            return func.negative_inner_product
        raise ValueError(f"Invalid vector index metric type: {self.vidx_metric_type}")

    def _normalize(self, vector: List[float]) -> List[float]:
        arr = np.array(vector)
        norm = np.linalg.norm(arr)
        arr = arr / norm
        return arr.tolist()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 1000,
        *,
        ids: Optional[List[str]] = None,
        extras: Optional[List[dict]] = None,
        partition_name: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Insert text data into OceanBase.

        Inserting data when the table has not be created yet will result
        in creating a new table. The data of the first record decides
        the schema of the new table, the dim is extracted from the first
        embedding.

        Args:
            texts (Iterable[str]): The texts to embed. OceanBase use a `LONGTEXT`
                type column to hold the data.
            metadatas (Optional[List[dict]]): Metadata dicts attached to each of
                the texts. Defaults to None.
            batch_size (int, optional): Batch size to use for insertion.
                Defaults to 1000.
            ids (Optional[List[str]]): List of text ids.
            extras (Optional[List[dict]]): Extra data to store in the table.
            partition_name (Optional[str]): The partition name to insert data into.

        Raises:
            Exception: Failure to add texts

        Returns:
            List[str]: The resulting ids for each inserted element.
        """
        texts = list(texts)

        try:
            embeddings = self.embedding_function.embed_documents(texts)
        except NotImplementedError:
            embeddings = [self.embedding_function.embed_query(x) for x in texts]

        total_count = len(embeddings)
        if total_count == 0:
            return []

        self._create_table_with_index(embeddings)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        extra_data = extras or [{} for _ in texts]

        pks: list[str] = []
        for i in range(0, total_count, batch_size):
            data = [
                {
                    self.primary_field: id,
                    self.vector_field: (
                        embedding if not self.normalize else self._normalize(embedding)
                    ),
                    self.text_field: text,
                    self.metadata_field: metadata,
                    **extra,
                }
                for id, embedding, text, metadata, extra in zip(
                    ids[i : i + batch_size],
                    embeddings[i : i + batch_size],
                    texts[i : i + batch_size],
                    metadatas[i : i + batch_size],
                    extra_data[i : i + batch_size],
                )
            ]
            try:
                self.obvector.insert(
                    table_name=self.table_name,
                    data=data,
                    partition_name=(partition_name or ""),
                )
                pks.extend(ids[i : i + batch_size])
            except Exception:
                traceback.print_exc()
                logger.error(
                    f"Failed to insert batch starting at entity:[{i}, {i + batch_size})"
                )
        return pks

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        table_name: str = DEFAULT_OCEANBASE_VECTOR_TABLE_NAME,
        connection_args: Optional[dict[str, Any]] = None,
        vidx_metric_type: str = DEFAULT_OCEANBASE_VECTOR_METRIC_TYPE,
        vidx_algo_params: Optional[dict] = None,
        drop_old: bool = False,
        *,
        ids: Optional[List[str]] = None,
        extra_columns: Optional[List[Column]] = None,
        normalize: bool = False,
        extras: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "OceanBase":
        """Create a OceanBase table, indexes it with HNSW, and insert data.

        Args:
            texts (List[str]): Text data.
            embedding (Embeddings): Embedding function.
            metadatas (Optional[List[dict]]): Metadata for each text if it exists.
                Defaults to None.
            table_name (str): Table name to use. Defaults to "langchain_vector".
            connection_args (Optional[dict[str, Any]]): Refer to
                `DEFAULT_OCEANBASE_CONNECTION` for example.
            vidx_metric_type (str): Metric method of distance between vectors.
                This parameter takes values in `l2` and `inner_product`.
                Defaults to `l2`.
            vidx_algo_params (Optional[dict]): Which index params to use. Now OceanBase
                supports HNSW only. Refer to `DEFAULT_OCEANBASE_HNSW_BUILD_PARAM`
                for example.
            drop_old (bool): Whether to drop the current table. Defaults
                to False.
            ids (Optional[List[str]]): List of text ids. Defaults to None.
            extra_columns (Optional[List[Column]]): Extra columns to add to the table.
            extras (Optional[List[dict]]): Extra data to insert. Defaults to None.

        Returns:
            OceanBase: OceanBase Vector Store
        """
        oceanbase = cls(
            embedding_function=embedding,
            table_name=table_name,
            connection_args=connection_args,
            vidx_metric_type=vidx_metric_type,
            vidx_algo_params=vidx_algo_params,
            drop_old=drop_old,
            extra_columns=extra_columns,
            normalize=normalize,
            **kwargs,
        )
        oceanbase.add_texts(texts, metadatas, ids=ids, extras=extras)
        return oceanbase

    def delete(  # type: ignore[no-untyped-def]
        self, ids: Optional[List[str]] = None, fltr: Optional[str] = None, **kwargs
    ):
        """Delete by vector ID or boolean expression.

        Args:
            ids (Optional[List[str]]): List of ids to delete.
            fltr (Optional[str]): Boolean filter that specifies the entities to delete.
        """
        self.obvector.delete(
            table_name=self.table_name,
            ids=ids,
            where_clause=([text(fltr)] if fltr is not None else None),
        )

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get entities by vector ID.

        Args:
            ids (Optional[List[str]]): List of ids to get.

        Returns:
            List[Document]: Document results for search.
        """
        res = self.obvector.get(
            table_name=self.table_name,
            ids=ids,
            output_column_names=[self.text_field, self.metadata_field],
        )
        return [
            Document(
                page_content=r[0],
                metadata=json.loads(r[1]),
            )
            for r in res.fetchall()
        ]

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        param: Optional[dict] = None,
        fltr: Optional[str] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Perform a similarity search against the query string.

        Args:
            query (str): The text to search.
            k (int, optional): How many results to return. Defaults to 10.
            param (Optional[dict]): The search params for the index type.
                Defaults to None. Refer to `DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM`
                for example.
            fltr (Optional[str]): Boolean filter. Defaults to None.

        Returns:
            List[Document]: Document results for search.
        """
        if k < 0:
            return []

        query_vector = self.embedding_function.embed_query(query)
        return self.similarity_search_by_vector(
            embedding=query_vector, k=k, param=param, fltr=fltr, **kwargs
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        param: Optional[dict] = None,
        fltr: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        Args:
            query (str): The text being searched.
            k (int, optional): How many results to return. Defaults to 10.
            param (Optional[dict]): The search params for the index type.
                Defaults to None. Refer to `DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM`
                for example.
            fltr (Optional[str]): Boolean filter. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: Document results with score for search.
        """
        if k < 0:
            return []

        query_vector = self.embedding_function.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding=query_vector, k=k, param=param, fltr=fltr, **kwargs
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        param: Optional[dict] = None,
        fltr: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform a similarity search against the query string.

        Args:
            embedding (List[float]): The embedding vector to search.
            k (int, optional): How many results to return. Defaults to 10.
            param (Optional[dict]): The search params for the index type.
                Defaults to None. Refer to `DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM`
                for example.
            fltr (Optional[str]): Boolean filter. Defaults to None.

        Returns:
            List[Document]: Document results for search.
        """
        if k < 0:
            return []

        search_param = (
            param if param is not None else DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM
        )
        ef_search = search_param.get(
            "efSearch", DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM["efSearch"]
        )
        if ef_search != self.hnsw_ef_search:
            self.obvector.set_ob_hnsw_ef_search(ef_search)
            self.hnsw_ef_search = ef_search

        res = self.obvector.ann_search(
            table_name=self.table_name,
            vec_data=(embedding if not self.normalize else self._normalize(embedding)),
            vec_column_name=self.vector_field,
            distance_func=self._parse_metric_type_str_to_dist_func(),
            topk=k,
            output_column_names=[self.text_field, self.metadata_field],
            where_clause=([text(fltr)] if fltr is not None else None),
            **kwargs,
        )
        return [
            Document(
                page_content=r[0],
                metadata=json.loads(r[1]),
            )
            for r in res.fetchall()
        ]

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 10,
        param: Optional[dict] = None,
        fltr: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        Args:
            embedding (List[float]): The embedding vector being searched.
            k (int, optional): The amount of results to return. Defaults to 10.
            param (Optional[dict]): The search params for the index type.
                Defaults to None. Refer to `DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM`
                for example.
            fltr (Optional[str]): Boolean filter. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: Document results with score for search.
        """
        if k < 0:
            return []

        search_param = (
            param if param is not None else DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM
        )
        ef_search = search_param.get(
            "efSearch", DEFAULT_OCEANBASE_HNSW_SEARCH_PARAM["efSearch"]
        )
        if ef_search != self.hnsw_ef_search:
            self.obvector.set_ob_hnsw_ef_search(ef_search)
            self.hnsw_ef_search = ef_search

        res = self.obvector.ann_search(
            table_name=self.table_name,
            vec_data=(embedding if not self.normalize else self._normalize(embedding)),
            vec_column_name=self.vector_field,
            distance_func=self._parse_metric_type_str_to_dist_func(),
            with_dist=True,
            topk=k,
            output_column_names=[self.text_field, self.metadata_field],
            where_clause=([text(fltr)] if fltr is not None else None),
            **kwargs,
        )
        return [
            (
                Document(
                    page_content=r[0],
                    metadata=json.loads(r[1]),
                ),
                r[2],
            )
            for r in res.fetchall()
        ]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        Select the relevance score function based on the distance strategy.
        """
        if self.vidx_metric_type == "inner_product":
            return _neg_inner_product_similarity
        elif self.vidx_metric_type == "l2":
            return _euclidean_similarity
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self.vidx_metric_type}."
            )
