"""Vector Store in Google Cloud BigQuery."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime
from functools import partial
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.utils.google import get_client_info
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)

DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.EUCLIDEAN_DISTANCE
DEFAULT_DOC_ID_COLUMN_NAME = "doc_id"  # document id
DEFAULT_TEXT_EMBEDDING_COLUMN_NAME = "text_embedding"  # embeddings vectors
DEFAULT_METADATA_COLUMN_NAME = "metadata"  # document metadata
DEFAULT_CONTENT_COLUMN_NAME = "content"  # text content, do not rename
DEFAULT_TOP_K = 4  # default number of documents returned from similarity search

_MIN_INDEX_ROWS = 5000  # minimal number of rows for creating an index
_INDEX_CHECK_PERIOD_SECONDS = 60  # Do not check for index more often that this.
_vector_table_lock = Lock()  # process-wide BigQueryVectorSearch table lock


@deprecated(
    since="0.0.33",
    removal="1.0",
    alternative_import="langchain_google_community.BigQueryVectorSearch",
)
class BigQueryVectorSearch(VectorStore):
    """Google Cloud BigQuery vector store.

    To use, you need the following packages installed:
        google-cloud-bigquery
    """

    def __init__(
        self,
        embedding: Embeddings,
        project_id: str,
        dataset_name: str,
        table_name: str,
        location: str = "US",
        content_field: str = DEFAULT_CONTENT_COLUMN_NAME,
        metadata_field: str = DEFAULT_METADATA_COLUMN_NAME,
        text_embedding_field: str = DEFAULT_TEXT_EMBEDDING_COLUMN_NAME,
        doc_id_field: str = DEFAULT_DOC_ID_COLUMN_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        credentials: Optional[Any] = None,
    ):
        """Constructor for BigQueryVectorSearch.

        Args:
            embedding (Embeddings): Text Embedding model to use.
            project_id (str): GCP project.
            dataset_name (str): BigQuery dataset to store documents and embeddings.
            table_name (str): BigQuery table name.
            location (str, optional): BigQuery region. Defaults to
                                      `US`(multi-region).
            content_field (str): Specifies the column to store the content.
                                 Defaults to `content`.
            metadata_field (str): Specifies the column to store the metadata.
                                  Defaults to `metadata`.
            text_embedding_field (str): Specifies the column to store
                                        the embeddings vector.
                                        Defaults to `text_embedding`.
            doc_id_field (str): Specifies the column to store the document id.
                                Defaults to `doc_id`.
            distance_strategy (DistanceStrategy, optional):
                Determines the strategy employed for calculating
                the distance between vectors in the embedding space.
                Defaults to EUCLIDEAN_DISTANCE.
                Available options are:
                - COSINE: Measures the similarity between two vectors of an inner
                    product space.
                - EUCLIDEAN_DISTANCE: Computes the Euclidean distance between
                    two vectors. This metric considers the geometric distance in
                    the vector space, and might be more suitable for embeddings
                    that rely on spatial relationships. This is the default behavior
            credentials (Credentials, optional): Custom Google Cloud credentials
                to use. Defaults to None.
        """
        try:
            from google.cloud import bigquery

            client_info = get_client_info(module="bigquery-vector-search")
            self.bq_client = bigquery.Client(
                project=project_id,
                location=location,
                credentials=credentials,
                client_info=client_info,
            )
        except ModuleNotFoundError:
            raise ImportError(
                "Please, install or upgrade the google-cloud-bigquery library: "
                "pip install google-cloud-bigquery"
            )
        self._logger = logging.getLogger(__name__)
        self._creating_index = False
        self._have_index = False
        self.embedding_model = embedding
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.table_name = table_name
        self.location = location
        self.content_field = content_field
        self.metadata_field = metadata_field
        self.text_embedding_field = text_embedding_field
        self.doc_id_field = doc_id_field
        self.distance_strategy = distance_strategy
        self._full_table_id = f"{self.project_id}.{self.dataset_name}.{self.table_name}"
        self._logger.debug("Using table `%s`", self.full_table_id)
        with _vector_table_lock:
            self.vectors_table = self._initialize_table()
        self._last_index_check = datetime.min
        self._initialize_vector_index()

    def _initialize_table(self) -> Any:
        """Validates or creates the BigQuery table."""
        from google.cloud import bigquery

        table_ref = bigquery.TableReference.from_string(self._full_table_id)
        table = self.bq_client.create_table(table_ref, exists_ok=True)
        changed_schema = False
        schema = table.schema.copy()
        columns = {c.name: c for c in schema}
        if self.doc_id_field not in columns:
            changed_schema = True
            schema.append(
                bigquery.SchemaField(name=self.doc_id_field, field_type="STRING")
            )
        elif (
            columns[self.doc_id_field].field_type != "STRING"
            or columns[self.doc_id_field].mode == "REPEATED"
        ):
            raise ValueError(f"Column {self.doc_id_field} must be of STRING type")
        if self.metadata_field not in columns:
            changed_schema = True
            schema.append(
                bigquery.SchemaField(name=self.metadata_field, field_type="JSON")
            )
        elif (
            columns[self.metadata_field].field_type not in ["JSON", "STRING"]
            or columns[self.metadata_field].mode == "REPEATED"
        ):
            raise ValueError(
                f"Column {self.metadata_field} must be of STRING or JSON type"
            )
        if self.content_field not in columns:
            changed_schema = True
            schema.append(
                bigquery.SchemaField(name=self.content_field, field_type="STRING")
            )
        elif (
            columns[self.content_field].field_type != "STRING"
            or columns[self.content_field].mode == "REPEATED"
        ):
            raise ValueError(f"Column {self.content_field} must be of STRING type")
        if self.text_embedding_field not in columns:
            changed_schema = True
            schema.append(
                bigquery.SchemaField(
                    name=self.text_embedding_field,
                    field_type="FLOAT64",
                    mode="REPEATED",
                )
            )
        elif (
            columns[self.text_embedding_field].field_type not in ("FLOAT", "FLOAT64")
            or columns[self.text_embedding_field].mode != "REPEATED"
        ):
            raise ValueError(
                f"Column {self.text_embedding_field} must be of ARRAY<FLOAT64> type"
            )
        if changed_schema:
            self._logger.debug("Updated table `%s` schema.", self.full_table_id)
            table.schema = schema
            table = self.bq_client.update_table(table, fields=["schema"])
        return table

    def _initialize_vector_index(self) -> Any:
        """
        A vector index in BigQuery table enables efficient
        approximate vector search.
        """
        from google.cloud import bigquery

        if self._have_index or self._creating_index:
            # Already have an index or in the process of creating one.
            return
        table = self.bq_client.get_table(self.vectors_table)
        if (table.num_rows or 0) < _MIN_INDEX_ROWS:
            # Not enough rows to create index.
            self._logger.debug("Not enough rows to create a vector index.")
            return
        if (
            datetime.utcnow() - self._last_index_check
        ).total_seconds() < _INDEX_CHECK_PERIOD_SECONDS:
            return
        with _vector_table_lock:
            if self._creating_index or self._have_index:
                return
            self._last_index_check = datetime.utcnow()
            # Check if index exists, create if necessary
            check_query = (
                f"SELECT 1 FROM `{self.project_id}.{self.dataset_name}"
                ".INFORMATION_SCHEMA.VECTOR_INDEXES` WHERE"
                f" table_name = '{self.table_name}'"
            )
            job = self.bq_client.query(
                check_query, api_method=bigquery.enums.QueryApiMethod.QUERY
            )
            if job.result().total_rows == 0:
                # Need to create an index. Make it in a separate thread.
                self._create_index_in_background()
            else:
                self._logger.debug("Vector index already exists.")
                self._have_index = True

    def _create_index_in_background(self):  # type: ignore[no-untyped-def]
        if self._have_index or self._creating_index:
            # Already have an index or in the process of creating one.
            return
        self._creating_index = True
        self._logger.debug("Trying to create a vector index.")
        thread = Thread(target=self._create_index, daemon=True)
        thread.start()

    def _create_index(self):  # type: ignore[no-untyped-def]
        from google.api_core.exceptions import ClientError

        table = self.bq_client.get_table(self.vectors_table)
        if (table.num_rows or 0) < _MIN_INDEX_ROWS:
            # Not enough rows to create index.
            return
        if self.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            distance_type = "EUCLIDEAN"
        elif self.distance_strategy == DistanceStrategy.COSINE:
            distance_type = "COSINE"
        # Default to EUCLIDEAN_DISTANCE
        else:
            distance_type = "EUCLIDEAN"
        index_name = f"{self.table_name}_langchain_index"
        try:
            sql = f"""
                CREATE VECTOR INDEX IF NOT EXISTS
                `{index_name}`
                ON `{self.full_table_id}`({self.text_embedding_field})
                OPTIONS(distance_type="{distance_type}", index_type="IVF")
            """
            self.bq_client.query(sql).result()
            self._have_index = True
        except ClientError as ex:
            self._logger.debug("Vector index creation failed (%s).", ex.args[0])
        finally:
            self._creating_index = False

    def _persist(self, data: Dict[str, Any]) -> None:
        """Saves documents and embeddings to BigQuery."""
        from google.cloud import bigquery

        data_len = len(data[list(data.keys())[0]])
        if data_len == 0:
            return

        list_of_dicts = [dict(zip(data, t)) for t in zip(*data.values())]

        job_config = bigquery.LoadJobConfig()
        job_config.schema = self.vectors_table.schema
        job_config.schema_update_options = (
            bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
        )
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
        job = self.bq_client.load_table_from_json(
            list_of_dicts, self.vectors_table, job_config=job_config
        )
        job.result()

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding_model

    @property
    def full_table_id(self) -> str:
        return self._full_table_id

    def add_texts(  # type: ignore[override]
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: List of strings to add to the vectorstore.
            metadatas: Optional list of metadata associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embs = self.embedding_model.embed_documents(texts)
        return self.add_texts_with_embeddings(texts, embs, metadatas, **kwargs)

    def add_texts_with_embeddings(
        self,
        texts: List[str],
        embs: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: List of strings to add to the vectorstore.
            embs: List of lists of floats with text embeddings for texts.
            metadatas: Optional list of metadata associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        ids = [uuid.uuid4().hex for _ in texts]
        values_dict: Dict[str, List[Any]] = {
            self.content_field: texts,
            self.doc_id_field: ids,
        }
        if not metadatas:
            metadatas = []
        len_diff = len(ids) - len(metadatas)
        add_meta = [None for _ in range(0, len_diff)]
        metadatas = [m if m is not None else {} for m in metadatas + add_meta]
        values_dict[self.metadata_field] = metadatas
        values_dict[self.text_embedding_field] = embs
        self._persist(values_dict)
        return ids

    def get_documents(
        self, ids: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search documents by their ids or metadata values.

        Args:
            ids: List of ids of documents to retrieve from the vectorstore.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if ids and len(ids) > 0:
            from google.cloud import bigquery

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("ids", "STRING", ids),
                ]
            )
            id_expr = f"{self.doc_id_field} IN UNNEST(@ids)"
        else:
            job_config = None
            id_expr = "TRUE"
        if filter:
            filter_expressions = []
            for i in filter.items():
                if isinstance(i[1], float):
                    expr = (
                        "ABS(CAST(JSON_VALUE("
                        f"`{self.metadata_field}`,'$.{i[0]}') "
                        f"AS FLOAT64) - {i[1]}) "
                        f"<= {sys.float_info.epsilon}"
                    )
                else:
                    val = str(i[1]).replace('"', '\\"')
                    expr = f"JSON_VALUE(`{self.metadata_field}`,'$.{i[0]}') = \"{val}\""
                filter_expressions.append(expr)
            filter_expression_str = " AND ".join(filter_expressions)
            where_filter_expr = f" AND ({filter_expression_str})"
        else:
            where_filter_expr = ""

        job = self.bq_client.query(
            f"""
                    SELECT * FROM `{self.full_table_id}` WHERE {id_expr}
                    {where_filter_expr}
                    """,
            job_config=job_config,
        )
        docs: List[Document] = []
        for row in job:
            metadata = None
            if self.metadata_field:
                metadata = row[self.metadata_field]
            if metadata:
                if not isinstance(metadata, dict):
                    metadata = json.loads(metadata)
            else:
                metadata = {}
            metadata["__id"] = row[self.doc_id_field]
            doc = Document(page_content=row[self.content_field], metadata=metadata)
            docs.append(doc)
        return docs

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        if not ids or len(ids) == 0:
            return True
        from google.cloud import bigquery

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("ids", "STRING", ids),
            ]
        )
        self.bq_client.query(
            f"""
                    DELETE FROM `{self.full_table_id}` WHERE {self.doc_id_field}
                    IN UNNEST(@ids)
                    """,
            job_config=job_config,
        ).result()
        return True

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self.delete, **kwargs), ids
        )

    def _search_with_score_and_embeddings_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_TOP_K,
        filter: Optional[Dict[str, Any]] = None,
        brute_force: bool = False,
        fraction_lists_to_search: Optional[float] = None,
    ) -> List[Tuple[Document, List[float], float]]:
        from google.cloud import bigquery

        # Create an index if no index exists.
        if not self._have_index and not self._creating_index:
            self._initialize_vector_index()
        # Prepare filter
        filter_expr = "TRUE"
        if filter:
            filter_expressions = []
            for i in filter.items():
                if isinstance(i[1], float):
                    expr = (
                        "ABS(CAST(JSON_VALUE("
                        f"base.`{self.metadata_field}`,'$.{i[0]}') "
                        f"AS FLOAT64) - {i[1]}) "
                        f"<= {sys.float_info.epsilon}"
                    )
                else:
                    val = str(i[1]).replace('"', '\\"')
                    expr = (
                        f"JSON_VALUE(base.`{self.metadata_field}`,'$.{i[0]}')"
                        f' = "{val}"'
                    )
                filter_expressions.append(expr)
            filter_expression_str = " AND ".join(filter_expressions)
            filter_expr += f" AND ({filter_expression_str})"
        # Configure and run a query job.
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("v", "FLOAT64", embedding),
            ],
            use_query_cache=False,
            priority=bigquery.QueryPriority.BATCH,
        )
        if self.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            distance_type = "EUCLIDEAN"
        elif self.distance_strategy == DistanceStrategy.COSINE:
            distance_type = "COSINE"
        # Default to EUCLIDEAN_DISTANCE
        else:
            distance_type = "EUCLIDEAN"
        if brute_force:
            options_string = ",options => '{\"use_brute_force\":true}'"
        elif fraction_lists_to_search:
            if fraction_lists_to_search == 0 or fraction_lists_to_search >= 1.0:
                raise ValueError(
                    "`fraction_lists_to_search` must be between 0.0 and 1.0"
                )
            options_string = (
                ',options => \'{"fraction_lists_to_search":'
                f"{fraction_lists_to_search}}}'"
            )
        else:
            options_string = ""
        query = f"""
            SELECT
                base.*,
                distance AS _vector_search_distance
            FROM VECTOR_SEARCH(
                TABLE `{self.full_table_id}`,
                "{self.text_embedding_field}",
                (SELECT @v AS {self.text_embedding_field}),
                distance_type => "{distance_type}",
                top_k => {k}
                {options_string}
            )
            WHERE {filter_expr}
            LIMIT {k}
        """
        document_tuples: List[Tuple[Document, List[float], float]] = []
        # TODO(vladkol): Use jobCreationMode=JOB_CREATION_OPTIONAL when available.
        job = self.bq_client.query(
            query, job_config=job_config, api_method=bigquery.enums.QueryApiMethod.QUERY
        )
        # Process job results.
        for row in job:
            metadata = row[self.metadata_field]
            if metadata:
                if not isinstance(metadata, dict):
                    metadata = json.loads(metadata)
            else:
                metadata = {}
            metadata["__id"] = row[self.doc_id_field]
            metadata["__job_id"] = job.job_id
            doc = Document(page_content=row[self.content_field], metadata=metadata)
            document_tuples.append(
                (
                    doc,
                    row[self.text_embedding_field],
                    row["_vector_search_distance"],
                )
            )
        return document_tuples

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_TOP_K,
        filter: Optional[Dict[str, Any]] = None,
        brute_force: bool = False,
        fraction_lists_to_search: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
            brute_force: Whether to use brute force search. Defaults to False.
            fraction_lists_to_search: Optional percentage of lists to search,
                must be in range 0.0 and 1.0, exclusive.
                If Node, uses service's default which is 0.05.

        Returns:
            List of Documents most similar to the query vector with distance.
        """
        del kwargs
        document_tuples = self._search_with_score_and_embeddings_by_vector(
            embedding, k, filter, brute_force, fraction_lists_to_search
        )
        return [(doc, distance) for doc, _, distance in document_tuples]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_TOP_K,
        filter: Optional[Dict[str, Any]] = None,
        brute_force: bool = False,
        fraction_lists_to_search: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
            brute_force: Whether to use brute force search. Defaults to False.
            fraction_lists_to_search: Optional percentage of lists to search,
                must be in range 0.0 and 1.0, exclusive.
                If Node, uses service's default which is 0.05.

        Returns:
            List of Documents most similar to the query vector.
        """
        tuples = self.similarity_search_with_score_by_vector(
            embedding, k, filter, brute_force, fraction_lists_to_search, **kwargs
        )
        return [i[0] for i in tuples]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        filter: Optional[Dict[str, Any]] = None,
        brute_force: bool = False,
        fraction_lists_to_search: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with score.

        Args:
            query: search query text.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
            brute_force: Whether to use brute force search. Defaults to False.
            fraction_lists_to_search: Optional percentage of lists to search,
                must be in range 0.0 and 1.0, exclusive.
                If Node, uses service's default which is 0.05.

        Returns:
            List of Documents most similar to the query vector, with similarity scores.
        """
        emb = self.embedding_model.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            emb, k, filter, brute_force, fraction_lists_to_search, **kwargs
        )

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        filter: Optional[Dict[str, Any]] = None,
        brute_force: bool = False,
        fraction_lists_to_search: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search.

        Args:
            query: search query text.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
            brute_force: Whether to use brute force search. Defaults to False.
            fraction_lists_to_search: Optional percentage of lists to search,
                must be in range 0.0 and 1.0, exclusive.
                If Node, uses service's default which is 0.05.

        Returns:
            List of Documents most similar to the query vector.
        """
        tuples = self.similarity_search_with_score(
            query, k, filter, brute_force, fraction_lists_to_search, **kwargs
        )
        return [i[0] for i in tuples]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self.distance_strategy == DistanceStrategy.COSINE:
            return BigQueryVectorSearch._cosine_relevance_score_fn
        else:
            raise ValueError(
                "Relevance score is not supported "
                f"for `{self.distance_strategy}` distance."
            )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        fetch_k: int = DEFAULT_TOP_K * 5,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        brute_force: bool = False,
        fraction_lists_to_search: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: search query text.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
            brute_force: Whether to use brute force search. Defaults to False.
            fraction_lists_to_search: Optional percentage of lists to search,
                must be in range 0.0 and 1.0, exclusive.
                If Node, uses service's default which is 0.05.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        query_embedding = self.embedding_model.embed_query(query)
        doc_tuples = self._search_with_score_and_embeddings_by_vector(
            query_embedding, fetch_k, filter, brute_force, fraction_lists_to_search
        )
        doc_embeddings = [d[1] for d in doc_tuples]
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(query_embedding), doc_embeddings, lambda_mult=lambda_mult, k=k
        )
        return [doc_tuples[i][0] for i in mmr_doc_indexes]

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_TOP_K,
        fetch_k: int = DEFAULT_TOP_K * 5,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        brute_force: bool = False,
        fraction_lists_to_search: Optional[float] = None,
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
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
            brute_force: Whether to use brute force search. Defaults to False.
            fraction_lists_to_search: Optional percentage of lists to search,
                must be in range 0.0 and 1.0, exclusive.
                If Node, uses service's default which is 0.05.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        doc_tuples = self._search_with_score_and_embeddings_by_vector(
            embedding, fetch_k, filter, brute_force, fraction_lists_to_search
        )
        doc_embeddings = [d[1] for d in doc_tuples]
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(embedding), doc_embeddings, lambda_mult=lambda_mult, k=k
        )
        return [doc_tuples[i][0] for i in mmr_doc_indexes]

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        fetch_k: int = DEFAULT_TOP_K * 5,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        brute_force: bool = False,
        fraction_lists_to_search: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""

        func = partial(
            self.max_marginal_relevance_search,
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            brute_force=brute_force,
            fraction_lists_to_search=fraction_lists_to_search,
            **kwargs,
        )
        return await asyncio.get_event_loop().run_in_executor(None, func)

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_TOP_K,
        fetch_k: int = DEFAULT_TOP_K * 5,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        brute_force: bool = False,
        fraction_lists_to_search: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        return await asyncio.get_running_loop().run_in_executor(
            None,
            partial(self.max_marginal_relevance_search_by_vector, **kwargs),
            embedding,
            k,
            fetch_k,
            lambda_mult,
            filter,
            brute_force,
            fraction_lists_to_search,
        )

    @classmethod
    def from_texts(
        cls: Type["BigQueryVectorSearch"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "BigQueryVectorSearch":
        """Return VectorStore initialized from texts and embeddings."""
        vs_obj = BigQueryVectorSearch(embedding=embedding, **kwargs)
        vs_obj.add_texts(texts, metadatas)
        return vs_obj

    def explore_job_stats(self, job_id: str) -> Dict:
        """Return the statistics for a single job execution.

        Args:
            job_id: The BigQuery Job id.

        Returns:
            A dictionary of job statistics for a given job.
        """
        return self.bq_client.get_job(job_id)._properties["statistics"]
