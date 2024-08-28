from __future__ import annotations

import asyncio
import json
import logging
import uuid
from enum import Enum
from functools import partial
from typing import (
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
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST, VectorStore

from langchain_databricks.utils import maximal_marginal_relevance

logger = logging.getLogger(__name__)


class IndexType(str, Enum):
    DIRECT_ACCESS = "DIRECT_ACCESS"
    DELTA_SYNC = "DELTA_SYNC"


_DIRECT_ACCESS_ONLY_MSG = "`%s` is only supported for direct-access index."
_NON_MANAGED_EMB_ONLY_MSG = (
    "`%s` is not supported for index with Databricks-managed embeddings."
)


class DatabricksVectorSearch(VectorStore):
    """Databricks vector store integration.

    Setup:
        Install ``langchain-databricks`` and ``databricks-vectorsearch`` python packages.

        .. code-block:: bash

            pip install -U langchain-databricks databricks-vectorsearch

        If you don't have a Databricks Vector Search endpoint already, you can create one by following the instructions here: https://docs.databricks.com/en/generative-ai/create-query-vector-search.html

        If you are outside Databricks, set the Databricks workspace
        hostname and personal access token to environment variables:

        .. code-block:: bash

            export DATABRICKS_HOSTNAME="https://your-databricks-workspace"
            export DATABRICKS_TOKEN="your-personal-access-token"

    Key init args — indexing params:

        endpoint: The name of the Databricks Vector Search endpoint.
        index_name: The name of the index to use. Format: "catalog.schema.index".
        embedding: The embedding model.
                  Required for direct-access index or delta-sync index
                  with self-managed embeddings.
        text_column: The name of the text column to use for the embeddings.
                    Required for direct-access index or delta-sync index
                    with self-managed embeddings.
                    Make sure the text column specified is in the index.
        columns: The list of column names to get when doing the search.
                Defaults to ``[primary_key, text_column]``.

    Instantiate:

        `DatabricksVectorSearch` supports two types of indexes:

        * **Delta Sync Index** automatically syncs with a source Delta Table, automatically and incrementally updating the index as the underlying data in the Delta Table changes.

        * **Direct Vector Access Index** supports direct read and write of vectors and metadata. The user is responsible for updating this table using the REST API or the Python SDK.

        Also for delta-sync index, you can choose to use Databricks-managed embeddings or self-managed embeddings (via LangChain embeddings classes).

        If you are using a delta-sync index with Databricks-managed embeddings:

        .. code-block:: python

            from langchain_databricks.vectorstores import DatabricksVectorSearch

            vector_store = DatabricksVectorSearch(
                endpoint="<your-endpoint-name>",
                index_name="<your-index-name>"
            )

        If you are using a direct-access index or a delta-sync index with self-managed embeddings,
        you also need to provide the embedding model and text column in your source table to
        use for the embeddings:

        .. code-block:: python

            from langchain_openai import OpenAIEmbeddings

            vector_store = DatabricksVectorSearch(
                endpoint="<your-endpoint-name>",
                index_name="<your-index-name>",
                embedding=OpenAIEmbeddings(),
                text_column="document_content"
            )

    Add Documents:
        .. code-block:: python
            from langchain_core.documents import Document

            document_1 = Document(page_content="foo", metadata={"baz": "bar"})
            document_2 = Document(page_content="thud", metadata={"bar": "baz"})
            document_3 = Document(page_content="i will be deleted :(")
            documents = [document_1, document_2, document_3]
            ids = ["1", "2", "3"]
            vector_store.add_documents(documents=documents, ids=ids)

    Delete Documents:
        .. code-block:: python
            vector_store.delete(ids=["3"])

        .. note::

            The `delete` method is only supported for direct-access index.

    Search:
        .. code-block:: python
            results = vector_store.similarity_search(query="thud",k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")
        .. code-block:: python
            * thud [{'id': '2'}]

        .. note:

            By default, similarity search only returns the primary key and text column.
            If you want to retrieve the custom metadata associated with the document,
            pass the additional columns in the `columns` parameter when initializing the vector store.

            .. code-block:: python

                vector_store = DatabricksVectorSearch(
                    endpoint="<your-endpoint-name>",
                    index_name="<your-index-name>",
                    columns=["baz", "bar"],
                )

                vector_store.similarity_search(query="thud",k=1)
                # Output: * thud [{'bar': 'baz', 'baz': None, 'id': '2'}]

    Search with filter:
        .. code-block:: python
            results = vector_store.similarity_search(query="thud",k=1,filter={"bar": "baz"})
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")
        .. code-block:: python
            * thud [{'id': '2'}]

    Search with score:
        .. code-block:: python
            results = vector_store.similarity_search_with_score(query="qux",k=1)
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
        .. code-block:: python
            * [SIM=0.748804] foo [{'id': '1'}]

    Async:
        .. code-block:: python
            # add documents
            await vector_store.aadd_documents(documents=documents, ids=ids)
            # delete documents
            await vector_store.adelete(ids=["3"])
            # search
            results = vector_store.asimilarity_search(query="thud",k=1)
            # search with score
            results = await vector_store.asimilarity_search_with_score(query="qux",k=1)
            for doc,score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
        .. code-block:: python
            * [SIM=0.748807] foo [{'id': '1'}]

    Use as Retriever:
        .. code-block:: python
            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
            )
            retriever.invoke("thud")
        .. code-block:: python
            [Document(metadata={'id': '2'}, page_content='thud')]
    """  # noqa: E501

    def __init__(
        self,
        endpoint: str,
        index_name: str,
        embedding: Optional[Embeddings] = None,
        text_column: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ):
        try:
            from databricks.vector_search.client import (  # type: ignore[import]
                VectorSearchClient,
            )
        except ImportError as e:
            raise ImportError(
                "Could not import databricks-vectorsearch python package. "
                "Please install it with `pip install databricks-vectorsearch`."
            ) from e

        self.index = VectorSearchClient().get_index(endpoint, index_name)
        self._index_details = IndexDetails(self.index)

        _validate_embedding(embedding, self._index_details)
        self._embeddings = embedding
        self._text_column = _validate_and_get_text_column(
            text_column, self._index_details
        )
        self._columns = _validate_and_get_return_columns(
            columns or [], self._text_column, self._index_details
        )
        self._primary_key = self._index_details.primary_key

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the query embedding object if available."""
        return self._embeddings

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

        .. note::

            This method is only supported for a direct-access index.

        Args:
            texts: List of texts to add.
            metadatas: List of metadata for each text. Defaults to None.
            ids: List of ids for each text. Defaults to None.
                If not provided, a random uuid will be generated for each text.

        Returns:
            List of ids from adding the texts into the index.
        """
        if self._index_details.is_delta_sync_index():
            raise NotImplementedError(_DIRECT_ACCESS_ONLY_MSG % "add_texts")

        # Wrap to list if input texts is a single string
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        vectors = self._embeddings.embed_documents(texts)  # type: ignore[union-attr]
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        metadatas = metadatas or [{} for _ in texts]

        updates = [
            {
                self._primary_key: id_,
                self._text_column: text,
                self._index_details.embedding_vector_column["name"]: vector,
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

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self.add_texts, **kwargs), texts, metadatas
        )

    def delete(self, ids: Optional[List[Any]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents from the index.

        .. note::

            This method is only supported for a direct-access index.

        Args:
            ids: List of ids of documents to delete.

        Returns:
            True if successful.
        """
        if self._index_details.is_delta_sync_index():
            raise NotImplementedError(_DIRECT_ACCESS_ONLY_MSG % "delete")

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

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        func = partial(self.similarity_search, query, k=k, **kwargs)
        return await asyncio.get_event_loop().run_in_executor(None, func)

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
        if self._index_details.is_databricks_managed_embeddings():
            query_text = query
            query_vector = None
        else:
            # The value for `query_text` needs to be specified only for hybrid search.
            if query_type is not None and query_type.upper() == "HYBRID":
                query_text = query
            else:
                query_text = None
            query_vector = self._embeddings.embed_query(query)  # type: ignore[union-attr]

        search_resp = self.index.similarity_search(
            columns=self._columns,
            query_text=query_text,
            query_vector=query_vector,
            filters=filter,
            num_results=k,
            query_type=query_type,
        )
        return self._parse_search_response(search_resp)

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        Databricks Vector search uses a normalized score 1/(1+d) where d
        is the L2 distance. Hence, we simply return the identity function.
        """
        return lambda score: score

    async def asimilarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        func = partial(self.similarity_search_with_score, *args, **kwargs)
        return await asyncio.get_event_loop().run_in_executor(None, func)

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
        if self._index_details.is_databricks_managed_embeddings():
            raise NotImplementedError(
                _NON_MANAGED_EMB_ONLY_MSG % "similarity_search_by_vector"
            )

        docs_with_score = self.similarity_search_by_vector_with_score(
            embedding=embedding,
            k=k,
            filter=filter,
            query_type=query_type,
            query=query,
            **kwargs,
        )
        return [doc for doc, _ in docs_with_score]

    async def asimilarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        func = partial(self.similarity_search_by_vector, embedding, k=k, **kwargs)
        return await asyncio.get_event_loop().run_in_executor(None, func)

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

        .. note::

            This method is not supported for index with Databricks-managed embeddings.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filters to apply to the query. Defaults to None.
            query_type: The type of this query. Supported values are "ANN" and "HYBRID".

        Returns:
            List of Documents most similar to the embedding and score for each.
        """
        if self._index_details.is_databricks_managed_embeddings():
            raise NotImplementedError(
                _NON_MANAGED_EMB_ONLY_MSG % "similarity_search_by_vector_with_score"
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
            columns=self._columns,
            query_vector=embedding,
            query_text=query_text,
            filters=filter,
            num_results=k,
            query_type=query_type,
        )
        return self._parse_search_response(search_resp)

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

        .. note::

            This method is not supported for index with Databricks-managed embeddings.

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
        if self._index_details.is_databricks_managed_embeddings():
            raise NotImplementedError(
                _NON_MANAGED_EMB_ONLY_MSG % "max_marginal_relevance_search"
            )

        query_vector = self._embeddings.embed_query(query)  # type: ignore[union-attr]
        docs = self.max_marginal_relevance_search_by_vector(
            query_vector,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            query_type=query_type,
        )
        return docs

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        func = partial(
            self.max_marginal_relevance_search,
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )
        return await asyncio.get_event_loop().run_in_executor(None, func)

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

        .. note::

            This method is not supported for index with Databricks-managed embeddings.

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
        if self._index_details.is_databricks_managed_embeddings():
            raise NotImplementedError(
                _NON_MANAGED_EMB_ONLY_MSG % "max_marginal_relevance_search_by_vector"
            )

        embedding_column = self._index_details.embedding_vector_column["name"]
        search_resp = self.index.similarity_search(
            columns=list(set(self._columns + [embedding_column])),
            query_text=None,
            query_vector=embedding,
            filters=filter,
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
            [embedding_column] if embedding_column not in self._columns else []
        )
        candidates = self._parse_search_response(search_resp, ignore_cols=ignore_cols)
        selected_results = [r[0] for i, r in enumerate(candidates) if i in mmr_selected]
        return selected_results

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError

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
            doc_id = result[columns.index(self._primary_key)]
            text_content = result[columns.index(self._text_column)]
            ignore_cols = [self._primary_key, self._text_column] + ignore_cols
            metadata = {
                col: value
                for col, value in zip(columns[:-1], result[:-1])
                if col not in ignore_cols
            }
            metadata[self._primary_key] = doc_id
            score = result[-1]
            doc = Document(page_content=text_content, metadata=metadata)
            docs_with_score.append((doc, score))
        return docs_with_score


def _validate_and_get_text_column(
    text_column: Optional[str], index_details: IndexDetails
) -> str:
    if index_details.is_databricks_managed_embeddings():
        index_source_column: str = index_details.embedding_source_column["name"]
        # check if input text column matches the source column of the index
        if text_column is not None:
            raise ValueError(
                f"The index '{index_details.name}' has the source column configured as "
                f"'{index_source_column}'. Do not pass the `text_column` parameter."
            )
        return index_source_column
    else:
        if text_column is None:
            raise ValueError("The `text_column` parameter is required for this index.")
        return text_column


def _validate_and_get_return_columns(
    columns: List[str], text_column: str, index_details: IndexDetails
) -> List[str]:
    """
    Get a list of columns to retrieve from the index.

    If the index is direct-access index, validate the given columns against the schema.
    """
    # add primary key column and source column if not in columns
    if index_details.primary_key not in columns:
        columns.append(index_details.primary_key)
    if text_column and text_column not in columns:
        columns.append(text_column)

    # Validate specified columns are in the index
    if index_details.is_direct_access_index() and (
        index_schema := index_details.schema
    ):
        if missing_columns := [c for c in columns if c not in index_schema]:
            raise ValueError(
                "Some columns specified in `columns` are not "
                f"in the index schema: {missing_columns}"
            )
    return columns


def _validate_embedding(
    embedding: Optional[Embeddings], index_details: IndexDetails
) -> None:
    if index_details.is_databricks_managed_embeddings():
        if embedding is not None:
            raise ValueError(
                f"The index '{index_details.name}' uses Databricks-managed embeddings. "
                "Do not pass the `embedding` parameter when initializing vector store."
            )
    else:
        if not embedding:
            raise ValueError(
                "The `embedding` parameter is required for a direct-access index "
                "or delta-sync index with self-managed embedding."
            )
        _validate_embedding_dimension(embedding, index_details)


def _validate_embedding_dimension(
    embeddings: Embeddings, index_details: IndexDetails
) -> None:
    """validate if the embedding dimension matches with the index's configuration."""
    if index_embedding_dimension := index_details.embedding_vector_column.get(
        "embedding_dimension"
    ):
        # Infer the embedding dimension from the embedding function."""
        actual_dimension = len(embeddings.embed_query("test"))
        if actual_dimension != index_embedding_dimension:
            raise ValueError(
                f"The specified embedding model's dimension '{actual_dimension}' does "
                f"not match with the index configuration '{index_embedding_dimension}'."
            )


class IndexDetails:
    """An utility class to store the configuration details of an index."""

    def __init__(self, index: Any):
        self._index_details = index.describe()

    @property
    def name(self) -> str:
        return self._index_details["name"]

    @property
    def schema(self) -> Optional[Dict]:
        if self.is_direct_access_index():
            schema_json = self.index_spec.get("schema_json")
            if schema_json is not None:
                return json.loads(schema_json)
        return None

    @property
    def primary_key(self) -> str:
        return self._index_details["primary_key"]

    @property
    def index_spec(self) -> Dict:
        return (
            self._index_details.get("delta_sync_index_spec", {})
            if self.is_delta_sync_index()
            else self._index_details.get("direct_access_index_spec", {})
        )

    @property
    def embedding_vector_column(self) -> Dict:
        if vector_columns := self.index_spec.get("embedding_vector_columns"):
            return vector_columns[0]
        return {}

    @property
    def embedding_source_column(self) -> Dict:
        if source_columns := self.index_spec.get("embedding_source_columns"):
            return source_columns[0]
        return {}

    def is_delta_sync_index(self) -> bool:
        return self._index_details["index_type"] == IndexType.DELTA_SYNC.value

    def is_direct_access_index(self) -> bool:
        return self._index_details["index_type"] == IndexType.DIRECT_ACCESS.value

    def is_databricks_managed_embeddings(self) -> bool:
        return (
            self.is_delta_sync_index()
            and self.embedding_source_column.get("name") is not None
        )
