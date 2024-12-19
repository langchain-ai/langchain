import json
import logging
import uuid
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class TablestoreVectorStore(VectorStore):
    """`Tablestore` vector store.

    To use, you should have the ``tablestore`` python package installed.

    Example:
        .. code-block:: python

            import os

            from langchain_openai import OpenAIEmbeddings
            from langchain_community.vectorstores import TablestoreVectorStore
            import tablestore

            embeddings = OpenAIEmbeddings()
            store = TablestoreVectorStore(
                embeddings,
                endpoint=os.getenv("end_point"),
                instance_name=os.getenv("instance_name"),
                access_key_id=os.getenv("access_key_id"),
                access_key_secret=os.getenv("access_key_secret"),
                vector_dimension=512,
                # metadata mapping is used to filter non-vector fields.
                metadata_mappings=[
                    tablestore.FieldSchema(
                        "type",
                        tablestore.FieldType.KEYWORD,
                        index=True,
                        enable_sort_and_agg=True
                    ),
                    tablestore.FieldSchema(
                        "time",
                        tablestore.FieldType.LONG,
                        index=True,
                        enable_sort_and_agg=True
                    ),
                ]
            )
    """

    def __init__(
        self,
        embedding: Embeddings,
        *,
        endpoint: Optional[str] = None,
        instance_name: Optional[str] = None,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        table_name: Optional[str] = "langchain_vector_store_ots_v1",
        index_name: Optional[str] = "langchain_vector_store_ots_index_v1",
        text_field: Optional[str] = "content",
        vector_field: Optional[str] = "embedding",
        vector_dimension: int = 512,
        vector_metric_type: Optional[str] = "cosine",
        metadata_mappings: Optional[List[Any]] = None,
    ):
        try:
            import tablestore
        except ImportError:
            raise ImportError(
                "Could not import tablestore python package. "
                "Please install it with `pip install tablestore`."
            )
        self.__embedding = embedding
        self.__tablestore_client = tablestore.OTSClient(
            endpoint,
            access_key_id,
            access_key_secret,
            instance_name,
            retry_policy=tablestore.WriteRetryPolicy(),
        )
        self.__table_name = table_name
        self.__index_name = index_name
        self.__vector_dimension = vector_dimension
        self.__vector_field = vector_field
        self.__text_field = text_field
        if vector_metric_type == "cosine":
            self.__vector_metric_type = tablestore.VectorMetricType.VM_COSINE
        elif vector_metric_type == "euclidean":
            self.__vector_metric_type = tablestore.VectorMetricType.VM_EUCLIDEAN
        elif vector_metric_type == "dot_product":
            self.__vector_metric_type = tablestore.VectorMetricType.VM_DOT_PRODUCT
        else:
            raise ValueError(
                f"Unsupported vector_metric_type operator: {vector_metric_type}"
            )

        self.__metadata_mappings = [
            tablestore.FieldSchema(
                self.__text_field,
                tablestore.FieldType.TEXT,
                index=True,
                enable_sort_and_agg=False,
                store=False,
                analyzer=tablestore.AnalyzerType.MAXWORD,
            ),
            tablestore.FieldSchema(
                self.__vector_field,
                tablestore.FieldType.VECTOR,
                vector_options=tablestore.VectorOptions(
                    data_type=tablestore.VectorDataType.VD_FLOAT_32,
                    dimension=self.__vector_dimension,
                    metric_type=self.__vector_metric_type,
                ),
            ),
        ]

        if metadata_mappings:
            for mapping in metadata_mappings:
                if not isinstance(mapping, tablestore.FieldSchema):
                    raise ValueError(
                        f"meta_data mapping should be an "
                        f"instance of tablestore.FieldSchema, "
                        f"bug got {type(mapping)}"
                    )
                if (
                    mapping.field_name == text_field
                    or mapping.field_name == vector_field
                ):
                    continue
                self.__metadata_mappings.append(mapping)

    def create_table_if_not_exist(self) -> None:
        """Create table if not exist."""

        try:
            import tablestore
        except ImportError:
            raise ImportError(
                "Could not import tablestore python package. "
                "Please install it with `pip install tablestore`."
            )
        table_list = self.__tablestore_client.list_table()
        if self.__table_name in table_list:
            logger.info("Tablestore system table[%s] already exists", self.__table_name)
            return None
        logger.info(
            "Tablestore system table[%s] does not exist, try to create the table.",
            self.__table_name,
        )

        schema_of_primary_key = [("id", "STRING")]
        table_meta = tablestore.TableMeta(self.__table_name, schema_of_primary_key)
        table_options = tablestore.TableOptions()
        reserved_throughput = tablestore.ReservedThroughput(
            tablestore.CapacityUnit(0, 0)
        )
        try:
            self.__tablestore_client.create_table(
                table_meta, table_options, reserved_throughput
            )
            logger.info("Tablestore create table[%s] successfully.", self.__table_name)
        except tablestore.OTSClientError as e:
            logger.exception(
                "Tablestore create system table[%s] failed with client error, "
                "http_status:%d, error_message:%s",
                self.__table_name,
                e.get_http_status(),
                e.get_error_message(),
            )
        except tablestore.OTSServiceError as e:
            logger.exception(
                "Tablestore create system table[%s] failed with client error, "
                "http_status:%d, error_code:%s, error_message:%s, request_id:%s",
                self.__table_name,
                e.get_http_status(),
                e.get_error_code(),
                e.get_error_message(),
                e.get_request_id(),
            )

    def create_search_index_if_not_exist(self) -> None:
        """Create search index if not exist."""

        try:
            import tablestore
        except ImportError:
            raise ImportError(
                "Could not import tablestore python package. "
                "Please install it with `pip install tablestore`."
            )
        search_index_list = self.__tablestore_client.list_search_index(
            table_name=self.__table_name
        )
        if self.__index_name in [t[1] for t in search_index_list]:
            logger.info("Tablestore system index[%s] already exists", self.__index_name)
            return None
        index_meta = tablestore.SearchIndexMeta(self.__metadata_mappings)
        self.__tablestore_client.create_search_index(
            self.__table_name, self.__index_name, index_meta
        )
        logger.info(
            "Tablestore create system index[%s] successfully.", self.__index_name
        )

    def delete_table_if_exists(self) -> None:
        """Delete table if exists."""

        search_index_list = self.__tablestore_client.list_search_index(
            table_name=self.__table_name
        )
        for resp_tuple in search_index_list:
            self.__tablestore_client.delete_search_index(resp_tuple[0], resp_tuple[1])
        self.__tablestore_client.delete_table(self.__table_name)

    def delete_search_index(self, table_name: str, index_name: str) -> None:
        """Delete search index."""

        self.__tablestore_client.delete_search_index(table_name, index_name)

    def __write_row(
        self, row_id: str, content: str, embedding_vector: List[float], meta_data: dict
    ) -> None:
        try:
            import tablestore
        except ImportError:
            raise ImportError(
                "Could not import tablestore python package. "
                "Please install it with `pip install tablestore`."
            )
        primary_key = [("id", row_id)]
        attribute_columns = [
            (self.__text_field, content),
            (self.__vector_field, json.dumps(embedding_vector)),
        ]
        for k, v in meta_data.items():
            item = (k, v)
            attribute_columns.append(item)
        row = tablestore.Row(primary_key, attribute_columns)

        try:
            self.__tablestore_client.put_row(self.__table_name, row)
            logger.debug(
                "Tablestore put row successfully. id:%s, content:%s, meta_data:%s",
                row_id,
                content,
                meta_data,
            )
        except tablestore.OTSClientError as e:
            logger.exception(
                "Tablestore put row failed with client error:%s, "
                "id:%s, content:%s, meta_data:%s",
                e,
                row_id,
                content,
                meta_data,
            )
        except tablestore.OTSServiceError as e:
            logger.exception(
                "Tablestore put row failed with client error:%s, id:%s, content:%s, "
                "meta_data:%s, http_status:%d, "
                "error_code:%s, error_message:%s, request_id:%s",
                e,
                row_id,
                content,
                meta_data,
                e.get_http_status(),
                e.get_error_code(),
                e.get_error_message(),
                e.get_request_id(),
            )

    def __delete_row(self, row_id: str) -> None:
        try:
            import tablestore
        except ImportError:
            raise ImportError(
                "Could not import tablestore python package. "
                "Please install it with `pip install tablestore`."
            )
        primary_key = [("id", row_id)]
        try:
            self.__tablestore_client.delete_row(self.__table_name, primary_key, None)
            logger.info("Tablestore delete row successfully. id:%s", row_id)
        except tablestore.OTSClientError as e:
            logger.exception(
                "Tablestore delete row failed with client error:%s, id:%s", e, row_id
            )
        except tablestore.OTSServiceError as e:
            logger.exception(
                "Tablestore delete row failed with client error:%s, "
                "id:%s, http_status:%d, error_code:%s, error_message:%s, request_id:%s",
                e,
                row_id,
                e.get_http_status(),
                e.get_error_code(),
                e.get_error_message(),
                e.get_request_id(),
            )

    def __get_row(self, row_id: str) -> Document:
        try:
            import tablestore
        except ImportError:
            raise ImportError(
                "Could not import tablestore python package. "
                "Please install it with `pip install tablestore`."
            )
        primary_key = [("id", row_id)]
        try:
            _, row, _ = self.__tablestore_client.get_row(
                self.__table_name, primary_key, None, None, 1
            )
            logger.debug("Tablestore get row successfully. id:%s", row_id)
            if row is None:
                raise ValueError("Can't not find row_id:%s in tablestore." % row_id)
            document_id = row.primary_key[0][1]
            meta_data = {}
            text = ""
            for col in row.attribute_columns:
                key = col[0]
                val = col[1]
                if key == self.__text_field:
                    text = val
                    continue
                meta_data[key] = val
            return Document(
                id=document_id,
                page_content=text,
                metadata=meta_data,
            )
        except tablestore.OTSClientError as e:
            logger.exception(
                "Tablestore get row failed with client error:%s, id:%s", e, row_id
            )
            raise e
        except tablestore.OTSServiceError as e:
            logger.exception(
                "Tablestore get row failed with client error:%s, "
                "id:%s, http_status:%d, error_code:%s, error_message:%s, request_id:%s",
                e,
                row_id,
                e.get_http_status(),
                e.get_error_code(),
                e.get_error_message(),
                e.get_request_id(),
            )
            raise e

    def _tablestore_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        tablestore_filter_query: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        try:
            import tablestore
        except ImportError:
            raise ImportError(
                "Could not import tablestore python package. "
                "Please install it with `pip install tablestore`."
            )
        if tablestore_filter_query:
            if not isinstance(tablestore_filter_query, tablestore.Query):
                raise ValueError(
                    f"table_store_filter_query should be "
                    f"an instance of tablestore.Query, "
                    f"bug got {type(tablestore_filter_query)}"
                )
        if "knn_top_k" in kwargs:
            knn_top_k = kwargs["knn_top_k"]
        else:
            knn_top_k = k
        ots_query = tablestore.KnnVectorQuery(
            field_name=self.__vector_field,
            top_k=knn_top_k,
            float32_query_vector=query_embedding,
            filter=tablestore_filter_query,
        )
        sort = tablestore.Sort(
            sorters=[tablestore.ScoreSort(sort_order=tablestore.SortOrder.DESC)]
        )
        search_query = tablestore.SearchQuery(
            ots_query, limit=k, get_total_count=False, sort=sort
        )
        try:
            search_response = self.__tablestore_client.search(
                table_name=self.__table_name,
                index_name=self.__index_name,
                search_query=search_query,
                columns_to_get=tablestore.ColumnsToGet(
                    return_type=tablestore.ColumnReturnType.ALL
                ),
            )
            logger.info(
                "Tablestore search successfully. request_id:%s",
                search_response.request_id,
            )
            tuple_list = []
            for hit in search_response.search_hits:
                row = hit.row
                score = hit.score
                document_id = row[0][0][1]
                meta_data = {}
                text = ""
                for col in row[1]:
                    key = col[0]
                    val = col[1]
                    if key == self.__text_field:
                        text = val
                        continue
                    if key == self.__vector_field:
                        val = json.loads(val)
                    meta_data[key] = val
                doc = Document(
                    id=document_id,
                    page_content=text,
                    metadata=meta_data,
                )
                tuple_list.append((doc, score))
            return tuple_list
        except tablestore.OTSClientError as e:
            logger.exception("Tablestore search failed with client error:%s", e)
            raise e
        except tablestore.OTSServiceError as e:
            logger.exception(
                "Tablestore search failed with client error:%s, "
                "http_status:%d, error_code:%s, error_message:%s, request_id:%s",
                e,
                e.get_http_status(),
                e.get_error_code(),
                e.get_error_message(),
                e.get_request_id(),
            )
            raise e

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        ids = ids or [str(uuid.uuid4().hex) for _ in texts]
        text_list = list(texts)
        embeddings = self.__embedding.embed_documents(text_list)
        for i in range(len(ids)):
            row_id = ids[i]
            text = text_list[i]
            embedding_vector = embeddings[i]
            if len(embedding_vector) != self.__vector_dimension:
                raise RuntimeError(
                    "embedding vector size:%d is not the same as vector store dim:%d"
                    % (len(embedding_vector), self.__vector_dimension)
                )
            metadata = dict()
            if metadatas and metadatas[i]:
                metadata = metadatas[i]
            self.__write_row(
                row_id=row_id,
                content=text,
                embedding_vector=embedding_vector,
                meta_data=metadata,
            )
        return ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if ids:
            for row_id in ids:
                self.__delete_row(row_id)
        return True

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        return [self.__get_row(row_id) for row_id in ids]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        tablestore_filter_query: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return [
            doc
            for (doc, score) in self.similarity_search_with_score(
                query, k=k, tablestore_filter_query=tablestore_filter_query, **kwargs
            )
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        tablestore_filter_query: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        query_embedding = self.__embedding.embed_query(query)
        return self._tablestore_search(
            query_embedding,
            k=k,
            tablestore_filter_query=tablestore_filter_query,
            **kwargs,
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        tablestore_filter_query: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Document]:
        return [
            doc
            for (doc, score) in self._tablestore_search(
                embedding,
                k=k,
                tablestore_filter_query=tablestore_filter_query,
                **kwargs,
            )
        ]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        endpoint: Optional[str] = None,
        instance_name: Optional[str] = None,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        table_name: Optional[str] = "langchain_vector_store_ots_v1",
        index_name: Optional[str] = "langchain_vector_store_ots_index_v1",
        text_field: Optional[str] = "content",
        vector_field: Optional[str] = "embedding",
        vector_dimension: int = 512,
        vector_metric_type: Optional[str] = "cosine",
        metadata_mappings: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> "TablestoreVectorStore":
        store = cls(
            embedding=embedding,
            endpoint=endpoint,
            instance_name=instance_name,
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            table_name=table_name,
            index_name=index_name,
            text_field=text_field,
            vector_field=vector_field,
            vector_dimension=vector_dimension,
            vector_metric_type=vector_metric_type,
            metadata_mappings=metadata_mappings,
        )
        store.create_table_if_not_exist()
        store.create_search_index_if_not_exist()
        store.add_texts(texts, metadatas)
        return store
