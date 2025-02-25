from __future__ import annotations

import json
import logging
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

import numpy as np
from langchain_core._api import deprecated, warn_deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST, VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    from databricks.vector_search.client import VectorSearchIndex

logger = logging.getLogger(__name__)


@deprecated(
    since="0.3.3",
    removal="1.0",
    alternative_import="databricks_langchain.DatabricksVectorSearch",
)
class DatabricksVectorSearch(VectorStore):
    """`Databricks Vector Search` vector store.

    To use, you should have the ``databricks-vectorsearch`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import DatabricksVectorSearch
            from databricks.vector_search.client import VectorSearchClient

            vs_client = VectorSearchClient()
            vs_index = vs_client.get_index(
              endpoint_name="vs_endpoint",
              index_name="ml.llm.index"
            )
            vectorstore = DatabricksVectorSearch(vs_index)

    Args:
        index: A Databricks Vector Search index object.
        embedding: The embedding model.
                  Required for direct-access index or delta-sync index
                  with self-managed embeddings.
        text_column: The name of the text column to use for the embeddings.
                    Required for direct-access index or delta-sync index
                    with self-managed embeddings.
                    Make sure the text column specified is in the index.
        columns: The list of column names to get when doing the search.
                Defaults to ``[primary_key, text_column]``.

    Delta-sync index with Databricks-managed embeddings manages the ingestion, deletion,
    and embedding for you.
    Manually ingestion/deletion of the documents/texts is not supported for delta-sync
    index.

    If you want to use a delta-sync index with self-managed embeddings, you need to
    provide the embedding model and text column name to use for the embeddings.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import DatabricksVectorSearch
            from databricks.vector_search.client import VectorSearchClient
            from langchain_community.embeddings.openai import OpenAIEmbeddings

            vs_client = VectorSearchClient()
            vs_index = vs_client.get_index(
              endpoint_name="vs_endpoint",
              index_name="ml.llm.index"
            )
            vectorstore = DatabricksVectorSearch(
              index=vs_index,
              embedding=OpenAIEmbeddings(),
              text_column="document_content"
            )

    If you want to manage the documents ingestion/deletion yourself, you can use a
    direct-access index.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import DatabricksVectorSearch
            from databricks.vector_search.client import VectorSearchClient
            from langchain_community.embeddings.openai import OpenAIEmbeddings

            vs_client = VectorSearchClient()
            vs_index = vs_client.get_index(
              endpoint_name="vs_endpoint",
              index_name="ml.llm.index"
            )
            vectorstore = DatabricksVectorSearch(
              index=vs_index,
              embedding=OpenAIEmbeddings(),
              text_column="document_content"
            )
            vectorstore.add_texts(
              texts=["text1", "text2"]
            )

    For more information on Databricks Vector Search, see `Databricks Vector Search
    documentation: https://docs.databricks.com/en/generative-ai/vector-search.html.

    """

    def __init__(
        self,
        index: VectorSearchIndex,
        *,
        embedding: Optional[Embeddings] = None,
        text_column: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ):
        try:
            from databricks.vector_search.client import VectorSearchIndex
        except ImportError as e:
            raise ImportError(
                "Could not import databricks-vectorsearch python package. "
                "Please install it with `pip install databricks-vectorsearch`."
            ) from e
        # index
        self.index = index
        if not isinstance(index, VectorSearchIndex):
            raise TypeError("index must be of type VectorSearchIndex.")

        # index_details
        index_details = self.index.describe()
        self.primary_key = index_details["primary_key"]
        self.index_type = index_details.get("index_type")
        self._delta_sync_index_spec = index_details.get("delta_sync_index_spec", dict())
        self._direct_access_index_spec = index_details.get(
            "direct_access_index_spec", dict()
        )

        # text_column
        if self._is_databricks_managed_embeddings():
            index_source_column = self._embedding_source_column_name()
            # check if input text column matches the source column of the index
            if text_column is not None and text_column != index_source_column:
                raise ValueError(
                    f"text_column '{text_column}' does not match with the "
                    f"source column of the index: '{index_source_column}'."
                )
            self.text_column = index_source_column
        else:
            self._require_arg(text_column, "text_column")
            self.text_column = text_column

        # columns
        self.columns = columns or []
        # add primary key column and source column if not in columns
        if self.primary_key not in self.columns:
            self.columns.append(self.primary_key)
        if self.text_column and self.text_column not in self.columns:
            self.columns.append(self.text_column)

        # Validate specified columns are in the index
        if self._is_direct_access_index():
            index_schema = self._index_schema()
            if index_schema:
                for col in self.columns:
                    if col not in index_schema:
                        raise ValueError(
                            f"column '{col}' is not in the index's schema."
                        )

        # embedding model
        if not self._is_databricks_managed_embeddings():
            # embedding model is required for direct-access index
            # or delta-sync index with self-managed embedding
            self._require_arg(embedding, "embedding")
            self._embedding = embedding
            # validate dimension matches
            index_embedding_dimension = self._embedding_vector_column_dimension()
            if index_embedding_dimension is not None:
                inferred_embedding_dimension = self._infer_embedding_dimension()
                if inferred_embedding_dimension != index_embedding_dimension:
                    raise ValueError(
                        f"embedding model's dimension '{inferred_embedding_dimension}' "
                        f"does not match with the index's dimension "
                        f"'{index_embedding_dimension}'."
                    )
        else:
            if embedding is not None:
                logger.warning(
                    "embedding model is not used in delta-sync index with "
                    "Databricks-managed embeddings."
                )
            self._embedding = None

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict]] = None,
        **kwargs: Any,
    ) -> VST:
        raise NotImplementedError(
            "`from_texts` is not supported. "
            "Use `add_texts` to add to existing direct-access index."
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the index.

        Only support direct-access index.

        Args:
            texts: List of texts to add.
            metadatas: List of metadata for each text. Defaults to None.
            ids: List of ids for each text. Defaults to None.
                If not provided, a random uuid will be generated for each text.

        Returns:
            List of ids from adding the texts into the index.
        """
        self._op_require_direct_access_index("add_texts")
        assert self.embeddings is not None, "embedding model is required."
        # Wrap to list if input texts is a single string
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        vectors = self.embeddings.embed_documents(texts)
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        metadatas = metadatas or [{} for _ in texts]

        updates = [
            {
                self.primary_key: id_,
                self.text_column: text,
                self._embedding_vector_column_name(): vector,
                **metadata,
            }
            for text, vector, id_, metadata in zip(texts, vectors, ids, metadatas)
        ]

        upsert_resp = self.index.upsert(updates)
        if upsert_resp.get("status") in ("PARTIAL_SUCCESS", "FAILURE"):
            failed_ids = upsert_resp.get("result", dict()).get(
                "failed_primary_keys", []
            )
            if upsert_resp.get("status") == "FAILURE":
                logger.error("Failed to add texts to the index.")
            else:
                logger.warning("Some texts failed to be added to the index.")
            return [id_ for id_ in ids if id_ not in failed_ids]

        return ids

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the query embedding object if available."""
        return self._embedding

    def delete(self, ids: Optional[List[Any]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents from the index.

        Only support direct-access index.

        Args:
            ids: List of ids of documents to delete.

        Returns:
            True if successful.
        """
        self._op_require_direct_access_index("delete")
        if ids is None:
            raise ValueError("ids must be provided.")
        self.index.delete(ids)
        return True

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        *,
        query_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filters to apply to the query. Defaults to None.
            query_type: The type of this query. Supported values are "ANN" and "HYBRID".

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_with_score = self.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            query_type=query_type,
            **kwargs,
        )
        return [doc for doc, _ in docs_with_score]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        *,
        query_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filters to apply to the query. Defaults to None.
            query_type: The type of this query. Supported values are "ANN" and "HYBRID".

        Returns:
            List of Documents most similar to the embedding and score for each.
        """
        if self._is_databricks_managed_embeddings():
            query_text = query
            query_vector = None
        else:
            assert self.embeddings is not None, "embedding model is required."
            # The value for `query_text` needs to be specified only for hybrid search.
            if query_type is not None and query_type.upper() == "HYBRID":
                query_text = query
            else:
                query_text = None
            query_vector = self.embeddings.embed_query(query)
        search_resp = self.index.similarity_search(
            columns=self.columns,
            query_text=query_text,
            query_vector=query_vector,
            filters=filter or _alias_filters(kwargs),
            num_results=k,
            query_type=query_type,
        )
        return self._parse_search_response(search_resp)

    @staticmethod
    def _identity_fn(score: float) -> float:
        return score

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        Databricks Vector search uses a normalized score 1/(1+d) where d
        is the L2 distance. Hence, we simply return the identity function.
        """

        return self._identity_fn

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        *,
        query_type: Optional[str] = None,
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
            filter: Filters to apply to the query. Defaults to None.
            query_type: The type of this query. Supported values are "ANN" and "HYBRID".
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        if not self._is_databricks_managed_embeddings():
            assert self.embeddings is not None, "embedding model is required."
            query_vector = self.embeddings.embed_query(query)
        else:
            raise ValueError(
                "`max_marginal_relevance_search` is not supported for index with "
                "Databricks-managed embeddings."
            )

        docs = self.max_marginal_relevance_search_by_vector(
            query_vector,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            filter=filter or _alias_filters(kwargs),
            query_type=query_type,
        )
        return docs

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Any] = None,
        *,
        query_type: Optional[str] = None,
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
            filter: Filters to apply to the query. Defaults to None.
            query_type: The type of this query. Supported values are "ANN" and "HYBRID".
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        if not self._is_databricks_managed_embeddings():
            embedding_column = self._embedding_vector_column_name()
        else:
            raise ValueError(
                "`max_marginal_relevance_search` is not supported for index with "
                "Databricks-managed embeddings."
            )
        search_resp = self.index.similarity_search(
            columns=list(set(self.columns + [embedding_column])),
            query_text=None,
            query_vector=embedding,
            filters=filter or _alias_filters(kwargs),
            num_results=fetch_k,
            query_type=query_type,
        )

        embeddings_result_index = (
            search_resp.get("manifest").get("columns").index({"name": embedding_column})
        )
        embeddings = [
            doc[embeddings_result_index]
            for doc in search_resp.get("result").get("data_array")
        ]

        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )

        ignore_cols: List = (
            [embedding_column] if embedding_column not in self.columns else []
        )
        candidates = self._parse_search_response(search_resp, ignore_cols=ignore_cols)
        selected_results = [r[0] for i, r in enumerate(candidates) if i in mmr_selected]
        return selected_results

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Any] = None,
        *,
        query_type: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filters to apply to the query. Defaults to None.
            query_type: The type of this query. Supported values are "ANN" and "HYBRID".

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_with_score = self.similarity_search_by_vector_with_score(
            embedding=embedding,
            k=k,
            filter=filter,
            query_type=query_type,
            query=query,
            **kwargs,
        )
        return [doc for doc, _ in docs_with_score]

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Any] = None,
        *,
        query_type: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector, along with scores.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filters to apply to the query. Defaults to None.
            query_type: The type of this query. Supported values are "ANN" and "HYBRID".

        Returns:
            List of Documents most similar to the embedding and score for each.
        """
        if self._is_databricks_managed_embeddings():
            raise ValueError(
                "`similarity_search_by_vector` is not supported for index with "
                "Databricks-managed embeddings."
            )
        if query_type is not None and query_type.upper() == "HYBRID":
            if query is None:
                raise ValueError(
                    "A value for `query` must be specified for hybrid search."
                )
            query_text = query
        else:
            if query is not None:
                raise ValueError(
                    (
                        "Cannot specify both `embedding` and "
                        '`query` unless `query_type="HYBRID"'
                    )
                )
            query_text = None
        search_resp = self.index.similarity_search(
            columns=self.columns,
            query_vector=embedding,
            query_text=query_text,
            filters=filter or _alias_filters(kwargs),
            num_results=k,
            query_type=query_type,
        )
        return self._parse_search_response(search_resp)

    def _parse_search_response(
        self, search_resp: Dict, ignore_cols: Optional[List[str]] = None
    ) -> List[Tuple[Document, float]]:
        """Parse the search response into a list of Documents with score."""
        if ignore_cols is None:
            ignore_cols = []

        columns = [
            col["name"]
            for col in search_resp.get("manifest", dict()).get("columns", [])
        ]
        docs_with_score = []
        for result in search_resp.get("result", dict()).get("data_array", []):
            doc_id = result[columns.index(self.primary_key)]
            text_content = result[columns.index(self.text_column)]
            metadata = {
                col: value
                for col, value in zip(columns[:-1], result[:-1])
                if col not in ([self.primary_key, self.text_column] + ignore_cols)
            }
            metadata[self.primary_key] = doc_id
            score = result[-1]
            doc = Document(page_content=text_content, metadata=metadata)
            docs_with_score.append((doc, score))
        return docs_with_score

    def _index_schema(self) -> Optional[Dict]:
        """Return the index schema as a dictionary.
        Return None if no schema found.
        """
        if self._is_direct_access_index():
            schema_json = self._direct_access_index_spec.get("schema_json")
            if schema_json is not None:
                return json.loads(schema_json)
        return None

    def _embedding_vector_column_name(self) -> Optional[str]:
        """Return the name of the embedding vector column.
        None if the index is not a self-managed embedding index.
        """
        return self._embedding_vector_column().get("name")

    def _embedding_vector_column_dimension(self) -> Optional[int]:
        """Return the dimension of the embedding vector column.
        None if the index is not a self-managed embedding index.
        """
        return self._embedding_vector_column().get("embedding_dimension")

    def _embedding_vector_column(self) -> Dict:
        """Return the embedding vector column configs as a dictionary.
        Empty if the index is not a self-managed embedding index.
        """
        index_spec = (
            self._delta_sync_index_spec
            if self._is_delta_sync_index()
            else self._direct_access_index_spec
        )
        return next(iter(index_spec.get("embedding_vector_columns") or list()), dict())

    def _embedding_source_column_name(self) -> Optional[str]:
        """Return the name of the embedding source column.
        None if the index is not a Databricks-managed embedding index.
        """
        return self._embedding_source_column().get("name")

    def _embedding_source_column(self) -> Dict:
        """Return the embedding source column configs as a dictionary.
        Empty if the index is not a Databricks-managed embedding index.
        """
        index_spec = self._delta_sync_index_spec
        return next(iter(index_spec.get("embedding_source_columns") or list()), dict())

    def _is_delta_sync_index(self) -> bool:
        """Return True if the index is a delta-sync index."""
        return self.index_type == "DELTA_SYNC"

    def _is_direct_access_index(self) -> bool:
        """Return True if the index is a direct-access index."""
        return self.index_type == "DIRECT_ACCESS"

    def _is_databricks_managed_embeddings(self) -> bool:
        """Return True if the embeddings are managed by Databricks Vector Search."""
        return (
            self._is_delta_sync_index()
            and self._embedding_source_column_name() is not None
        )

    def _infer_embedding_dimension(self) -> int:
        """Infer the embedding dimension from the embedding function."""
        assert self.embeddings is not None, "embedding model is required."
        return len(self.embeddings.embed_query("test"))

    def _op_require_direct_access_index(self, op_name: str) -> None:
        """
        Raise ValueError if the operation is not supported for direct-access index."""
        if not self._is_direct_access_index():
            raise ValueError(f"`{op_name}` is only supported for direct-access index.")

    @staticmethod
    def _require_arg(arg: Any, arg_name: str) -> None:
        """Raise ValueError if the required arg with name `arg_name` is None."""
        if not arg:
            raise ValueError(f"`{arg_name}` is required for this index.")


def _alias_filters(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    The `filters` argument was used in the previous versions. It is now
    replaced with `filter` for consistency with other vector stores, but
    we still support `filters` for backward compatibility.
    """
    if "filters" in kwargs:
        warn_deprecated(
            since="0.2.11",
            removal="1.0",
            message="DatabricksVectorSearch received a key `filters` in search_kwargs. "
            "`filters` was deprecated since langchain-community 0.2.11 and will "
            "be removed in 0.3. Please use `filter` instead.",
        )
    return kwargs.pop("filters", None)
