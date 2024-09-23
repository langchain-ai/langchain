import logging
import uuid
import json
from typing import List, Optional, Any, Iterable, Tuple, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from sqlalchemy import Column, String, TEXT, JSON, Table, func, text

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


class OceanBase(VectorStore):
    """`OceanBase` vector store.

    You need to install `pyobvector` and run a standalone observer or OceanBase cluster.
    """

    def __init__(
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
        metadata_field: Optional[str] = None,
        vidx_name: str = "vidx",
        partitions=None,
        **kwargs,
    ):
        try:
            from pyobvector import ObVecClient
        except ImportError:
            raise ImportError(
                "Could not import pyobvector package. "
                "Please install it with `pip install pyobvector`."
            )

        self.embedding_function = embedding_function
        self.table_name = table_name
        self.connection_args = (
            connection_args
            if connection_args is not None
            else DEFAULT_OCEANBASE_CONNECTION
        )

        self.obvector: Optional[ObVecClient] = None
        self._create_client(**kwargs)
        assert self.obvector is not None

        self.vidx_metric_type = vidx_metric_type.lower()
        if self.vidx_metric_type not in ("l2", "cosine", "ip"):
            raise ValueError("`vidx_metric_type` should be set in `l2`/`cosine`/`ip`.")

        self.vidx_algo_params = (
            vidx_algo_params
            if vidx_algo_params is not None
            else DEFAULT_OCEANBASE_HNSW_BUILD_PARAM
        )

        self.drop_old = drop_old
        self.primary_field = primary_field
        self.vector_field = vector_field
        self.text_field = text_field
        self.metadata_field = metadata_field
        self.vidx_name = vidx_name
        self.partition = partitions
        self.hnsw_ef_search = -1

        if self.drop_old:
            self.obvector.drop_table_if_exist(self.table_name)

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def _create_client(self, **kwargs):
        """Create the client to the OceanBase server."""
        try:
            from pyobvector import ObVecClient
        except ImportError:
            raise ImportError(
                "Could not import pyobvector package. "
                "Please install it with `pip install pyobvector`."
            )

        host: str = self.connection_args.get("host", "localhost")
        port: str = self.connection_args.get("port", "2881")
        user: str = self.connection_args.get("user", "root@test")
        password: str = self.connection_args.get("password", "")
        db_name: str = self.connection_args.get("db_name", "test")

        self.obvector = ObVecClient(
            uri=host + ":" + port,
            user=user,
            password=password,
            db_name=db_name,
            **kwargs,
        )

    def _create_table_with_index(self, embeddings: list):
        try:
            from pyobvector import VECTOR
        except ImportError:
            raise ImportError(
                "Could not import pyobvector package. "
                "Please install it with `pip install pyobvector`."
            )

        if self.obvector.check_table_exists(self.table_name):
            table = Table(
                self.table_name,
                self.obvector.metadata_obj,
                autoload_with=self.obvector.engine,
            )
            column_names = [column.name for column in table.columns]
            assert len(column_names) in (3, 4)
            logging.info(f"load exist table with {column_names} columns")
            self.primary_field = column_names[0]
            self.vector_field = column_names[1]
            self.text_field = column_names[2]
            self.metadata_field = None if len(column_names) == 3 else column_names[3]
            return

        dim = len(embeddings[0])
        cols = [
            Column(
                self.primary_field, String(4096), primary_key=True, autoincrement=False
            ),
            Column(self.vector_field, VECTOR(dim)),
            Column(self.text_field, TEXT),
        ]
        if self.metadata_field is not None:
            cols.append(Column(self.metadata_field, JSON))

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
            paritions=self.partition,
        )

    def _parse_metric_type_str_to_dist_func(self):
        if self.vidx_metric_type == "l2":
            return func.l2_distance
        if self.vidx_metric_type == "cosine":
            return func.cosine_distance
        if self.vidx_metric_type == "ip":
            return func.inner_product
        raise ValueError(f"Invalid vector index metric type: {self.vidx_metric_type}")

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 1000,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        texts = list(texts)

        try:
            embeddings = self.embedding_function.embed_documents(texts)
        except NotImplementedError:
            embeddings = [self.embedding_function.embed_query(x) for x in texts]

        total_count = len(embeddings)
        if total_count == 0:
            return []

        if metadatas is not None:
            self.metadata_field = "metadata"
        self._create_table_with_index(embeddings)

        if metadatas is not None and self.metadata_field is None:
            raise ValueError("metadata field is not set in table.")

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]

        pks: list[str] = []
        for i in range(0, total_count, batch_size):
            if self.metadata_field is None:
                data = [
                    {
                        self.primary_field: id,
                        self.vector_field: embedding,
                        self.text_field: text,
                    }
                    for id, embedding, text in zip(
                        ids[i : i + batch_size],
                        embeddings[i : i + batch_size],
                        texts[i : i + batch_size],
                    )
                ]
            else:
                data = [
                    {
                        self.primary_field: id,
                        self.vector_field: embedding,
                        self.text_field: text,
                        self.metadata_field: metadata,
                    }
                    for id, embedding, text, metadata in zip(
                        ids[i : i + batch_size],
                        embeddings[i : i + batch_size],
                        texts[i : i + batch_size],
                        metadatas[i : i + batch_size],
                    )
                ]
            try:
                self.obvector.insert(table_name=self.table_name, data=data)
                pks.extend(ids[i : i + batch_size])
            except Exception as e:
                logger.error(
                    f"Failed to insert batch starting at entity: [{i}, {i + batch_size})"
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
        **kwargs: Any,
    ):
        oceanbase = cls(
            embedding_function=embedding,
            table_name=table_name,
            connection_args=connection_args,
            vidx_metric_type=vidx_metric_type,
            vidx_algo_params=vidx_algo_params,
            drop_old=drop_old,
            **kwargs,
        )
        oceanbase.add_texts(texts, metadatas, ids=ids)
        return oceanbase

    def delete(  # type: ignore[no-untyped-def]
        self, ids: Optional[List[str]] = None, fltr: Optional[str] = None, **kwargs
    ):
        self.obvector.delete(
            table_name=self.table_name,
            ids=ids,
            where_clause=([text(fltr)] if fltr is not None else None),
        )

    def get_by_ids(self, ids: List[str], **kwargs) -> list[Document]:
        res = self.obvector.get(
            table_name=self.table_name,
            ids=ids,
            output_column_name=(
                [self.text_field]
                if self.metadata_field is None
                else [self.text_field, self.metadata_field]
            ),
        )
        return [
            Document(
                page_content=r[0],
                metadata=json.loads(r[1]) if self.metadata_field is not None else None,
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
            vec_data=embedding,
            vec_column_name=self.vector_field,
            distance_func=self._parse_metric_type_str_to_dist_func(),
            topk=k,
            output_column_name=(
                [self.text_field]
                if self.metadata_field is None
                else [self.text_field, self.metadata_field]
            ),
            where_clause=([text(fltr)] if fltr is not None else None),
        )
        return [
            Document(
                page_content=r[0],
                metadata=json.loads(r[1]) if self.metadata_field is not None else None,
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
            vec_data=embedding,
            vec_column_name=self.vector_field,
            distance_func=self._parse_metric_type_str_to_dist_func(),
            with_dist=True,
            topk=k,
            output_column_name=(
                [self.text_field]
                if self.metadata_field is None
                else [self.text_field, self.metadata_field]
            ),
            where_clause=([text(fltr)] if fltr is not None else None),
        )
        return [
            (
                Document(
                    page_content=r[0],
                    metadata=json.loads(r[1]) if self.metadata_field is not None else None,
                ),
                r[2],
            )
            for r in res.fetchall()
        ]
