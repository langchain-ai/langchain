from __future__ import annotations

import base64
import os
import uuid
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Type

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import guard_import
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import maximal_marginal_relevance

DEFAULT_K = 4  # Number of Documents to return.


def import_lancedb() -> Any:
    """Import lancedb package."""
    return guard_import("lancedb")


def to_lance_filter(filter: Dict[str, str]) -> str:
    """Converts a dict filter to a LanceDB filter string."""
    return " AND ".join([f"{k} = '{v}'" for k, v in filter.items()])


class LanceDB(VectorStore):
    """`LanceDB` vector store.

    To use, you should have ``lancedb`` python package installed.
    You can install it with ``pip install lancedb``.

    Args:
        connection: LanceDB connection to use. If not provided, a new connection
                    will be created.
        embedding: Embedding to use for the vectorstore.
        vector_key: Key to use for the vector in the database. Defaults to ``vector``.
        id_key: Key to use for the id in the database. Defaults to ``id``.
        text_key: Key to use for the text in the database. Defaults to ``text``.
        table_name: Name of the table to use. Defaults to ``vectorstore``.
        api_key: API key to use for LanceDB cloud database.
        region: Region to use for LanceDB cloud database.
        mode: Mode to use for adding data to the table. Valid values are
              ``append`` and ``overwrite``. Defaults to ``overwrite``.



    Example:
        .. code-block:: python
            vectorstore = LanceDB(uri='/lancedb', embedding_function)
            vectorstore.add_texts(['text1', 'text2'])
            result = vectorstore.similarity_search('text1')
    """

    def __init__(
        self,
        connection: Optional[Any] = None,
        embedding: Optional[Embeddings] = None,
        uri: Optional[str] = "/tmp/lancedb",
        vector_key: Optional[str] = "vector",
        id_key: Optional[str] = "id",
        text_key: Optional[str] = "text",
        table_name: Optional[str] = "vectorstore",
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        mode: Optional[str] = "overwrite",
        table: Optional[Any] = None,
        distance: Optional[str] = "l2",
        reranker: Optional[Any] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        limit: int = DEFAULT_K,
    ):
        """Initialize with Lance DB vectorstore"""
        lancedb = guard_import("lancedb")
        lancedb.remote.table = guard_import("lancedb.remote.table")
        self._embedding = embedding
        self._vector_key = vector_key
        self._id_key = id_key
        self._text_key = text_key
        self.api_key = api_key or os.getenv("LANCE_API_KEY") if api_key != "" else None
        self.region = region
        self.mode = mode
        self.distance = distance
        self.override_relevance_score_fn = relevance_score_fn
        self.limit = limit
        self._fts_index = None

        if isinstance(reranker, lancedb.rerankers.Reranker):
            self._reranker = reranker
        elif reranker is None:
            self._reranker = None
        else:
            raise ValueError(
                "`reranker` has to be a lancedb.rerankers.Reranker object."
            )

        if isinstance(uri, str) and self.api_key is None:
            if uri.startswith("db://"):
                raise ValueError("API key is required for LanceDB cloud.")

        if self._embedding is None:
            raise ValueError("embedding object should be provided")

        if isinstance(connection, lancedb.db.LanceDBConnection):
            self._connection = connection
        elif isinstance(connection, (str, lancedb.db.LanceTable)):
            raise ValueError(
                "`connection` has to be a lancedb.db.LanceDBConnection object.\
                `lancedb.db.LanceTable` is deprecated."
            )
        else:
            if self.api_key is None:
                self._connection = lancedb.connect(uri)
            else:
                if isinstance(uri, str):
                    if uri.startswith("db://"):
                        self._connection = lancedb.connect(
                            uri, api_key=self.api_key, region=self.region
                        )
                    else:
                        self._connection = lancedb.connect(uri)
                        warnings.warn(
                            "api key provided with local uri.\
                            The data will be stored locally"
                        )
        if table is not None:
            try:
                assert isinstance(
                    table, (lancedb.db.LanceTable, lancedb.remote.table.RemoteTable)
                )
                self._table = table
                self._table_name = (
                    table.name if hasattr(table, "name") else "remote_table"
                )
            except AssertionError:
                raise ValueError(
                    """`table` has to be a lancedb.db.LanceTable or 
                    lancedb.remote.table.RemoteTable object."""
                )
        else:
            self._table = self.get_table(table_name, set_default=True)

    def results_to_docs(self, results: Any, score: bool = False) -> Any:
        columns = results.schema.names

        if "_distance" in columns:
            score_col = "_distance"
        elif "_relevance_score" in columns:
            score_col = "_relevance_score"
        else:
            score_col = None
        # Check if 'metadata' is in the columns
        has_metadata = "metadata" in columns

        if score_col is None or not score:
            return [
                Document(
                    page_content=results[self._text_key][idx].as_py(),
                    metadata=results["metadata"][idx].as_py() if has_metadata else {},
                )
                for idx in range(len(results))
            ]
        elif score_col and score:
            return [
                (
                    Document(
                        page_content=results[self._text_key][idx].as_py(),
                        metadata=results["metadata"][idx].as_py()
                        if has_metadata
                        else {},
                    ),
                    results[score_col][idx].as_py(),
                )
                for idx in range(len(results))
            ]

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Turn texts into embedding and add it to the database

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            ids: Optional list of ids to associate with the texts.

        Returns:
            List of ids of the added texts.
        """
        docs = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        embeddings = self._embedding.embed_documents(list(texts))  # type: ignore[union-attr]
        for idx, text in enumerate(texts):
            embedding = embeddings[idx]
            metadata = metadatas[idx] if metadatas else {"id": ids[idx]}
            docs.append(
                {
                    self._vector_key: embedding,
                    self._id_key: ids[idx],
                    self._text_key: text,
                    "metadata": metadata,
                }
            )

        tbl = self.get_table()

        if tbl is None:
            tbl = self._connection.create_table(self._table_name, data=docs)
            self._table = tbl
        else:
            if self.api_key is None:
                tbl.add(docs, mode=self.mode)
            else:
                tbl.add(docs)

        self._fts_index = None

        return ids

    def get_table(
        self, name: Optional[str] = None, set_default: Optional[bool] = False
    ) -> Any:
        """
        Fetches a table object from the database.

        Args:
            name (str, optional): The name of the table to fetch. Defaults to None
                                    and fetches current table object.
            set_default (bool, optional): Sets fetched table as the default table.
                                        Defaults to False.

        Returns:
            Any: The fetched table object.

        Raises:
            ValueError: If the specified table is not found in the database.

        """
        if name is not None:
            if set_default:
                self._table_name = name
                _name = self._table_name
            else:
                _name = name
        else:
            _name = self._table_name

        try:
            return self._connection.open_table(_name)
        except Exception:
            return None

    def create_index(
        self,
        col_name: Optional[str] = None,
        vector_col: Optional[str] = None,
        num_partitions: Optional[int] = 256,
        num_sub_vectors: Optional[int] = 96,
        index_cache_size: Optional[int] = None,
        metric: Optional[str] = "L2",
        name: Optional[str] = None,
    ) -> None:
        """
        Create a scalar(for non-vector cols) or a vector index on a table.
        Make sure your vector column has enough data before creating an index on it.

        Args:
            vector_col: Provide if you want to create index on a vector column.
            col_name: Provide if you want to create index on a non-vector column.
            metric: Provide the metric to use for vector index. Defaults to 'L2'
                    choice of metrics: 'L2', 'dot', 'cosine'
            num_partitions: Number of partitions to use for the index. Defaults to 256.
            num_sub_vectors: Number of sub-vectors to use for the index. Defaults to 96.
            index_cache_size: Size of the index cache. Defaults to None.
            name: Name of the table to create index on. Defaults to None.

        Returns:
            None
        """
        tbl = self.get_table(name)

        if vector_col:
            tbl.create_index(
                metric=metric,
                vector_column_name=vector_col,
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
                index_cache_size=index_cache_size,
            )
        elif col_name:
            tbl.create_scalar_index(col_name)
        else:
            raise ValueError("Provide either vector_col or col_name")

    def encode_image(self, uri: str) -> str:
        """Get base64 string from image URI."""
        with open(uri, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def add_images(
        self,
        uris: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more images through the embeddings and add to the vectorstore.

        Args:
            uris List[str]: File path to the image.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added images.
        """
        tbl = self.get_table()

        # Map from uris to b64 encoded strings
        b64_texts = [self.encode_image(uri=uri) for uri in uris]
        # Populate IDs
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in uris]
        embeddings = None
        # Set embeddings
        if self._embedding is not None and hasattr(self._embedding, "embed_image"):
            embeddings = self._embedding.embed_image(uris=uris)
        else:
            raise ValueError(
                "embedding object should be provided and must have embed_image method."
            )

        data = []
        for idx, emb in enumerate(embeddings):
            metadata = metadatas[idx] if metadatas else {"id": ids[idx]}
            data.append(
                {
                    self._vector_key: emb,
                    self._id_key: ids[idx],
                    self._text_key: b64_texts[idx],
                    "metadata": metadata,
                }
            )
        if tbl is None:
            tbl = self._connection.create_table(self._table_name, data=data)
            self._table = tbl
        else:
            tbl.add(data)

        return ids

    def _query(
        self,
        query: Any,
        k: Optional[int] = None,
        filter: Optional[Any] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        if k is None:
            k = self.limit
        tbl = self.get_table(name)
        if isinstance(filter, dict):
            filter = to_lance_filter(filter)

        prefilter = kwargs.get("prefilter", False)
        query_type = kwargs.get("query_type", "vector")

        if metrics := kwargs.get("metrics"):
            lance_query = (
                tbl.search(query=query, vector_column_name=self._vector_key)
                .limit(k)
                .metric(metrics)
                .where(filter, prefilter=prefilter)
            )
        else:
            lance_query = (
                tbl.search(query=query, vector_column_name=self._vector_key)
                .limit(k)
                .where(filter, prefilter=prefilter)
            )
        if query_type == "hybrid" and self._reranker is not None:
            lance_query.rerank(reranker=self._reranker)

        docs = lance_query.to_arrow()
        if len(docs) == 0:
            warnings.warn("No results found for the query.")
        return docs

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn:
            return self.override_relevance_score_fn

        if self.distance == "cosine":
            return self._cosine_relevance_score_fn
        elif self.distance == "l2":
            return self._euclidean_relevance_score_fn
        elif self.distance == "ip":
            return self._max_inner_product_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance metric of type: {self.distance}."
                "Consider providing relevance_score_fn to Chroma constructor."
            )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        filter: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Return documents most similar to the query vector.
        """
        if k is None:
            k = self.limit

        res = self._query(embedding, k, filter=filter, name=name, **kwargs)
        return self.results_to_docs(res, score=kwargs.pop("score", False))

    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        filter: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Return documents most similar to the query vector with relevance scores.
        """
        if k is None:
            k = self.limit

        relevance_score_fn = self._select_relevance_score_fn()
        docs_and_scores = self.similarity_search_by_vector(
            embedding, k, score=True, **kwargs
        )
        return [
            (doc, relevance_score_fn(float(score))) for doc, score in docs_and_scores
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Return documents most similar to the query with relevance scores."""
        if k is None:
            k = self.limit

        score = kwargs.get("score", True)
        name = kwargs.get("name", None)
        query_type = kwargs.get("query_type", "vector")

        if self._embedding is None:
            raise ValueError("search needs an emmbedding function to be specified.")

        if query_type == "fts" or query_type == "hybrid":
            if self.api_key is None and self._fts_index is None:
                tbl = self.get_table(name)
                self._fts_index = tbl.create_fts_index(self._text_key, replace=True)

                if query_type == "hybrid":
                    embedding = self._embedding.embed_query(query)
                    _query = (embedding, query)
                else:
                    _query = query  # type: ignore[assignment]

                res = self._query(_query, k, filter=filter, name=name, **kwargs)
                return self.results_to_docs(res, score=score)
            else:
                raise NotImplementedError(
                    "Full text/ Hybrid search is not supported in LanceDB Cloud yet."
                )
        else:
            embedding = self._embedding.embed_query(query)
            res = self._query(embedding, k, filter=filter, **kwargs)
            return self.results_to_docs(res, score=score)

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        name: Optional[str] = None,
        filter: Optional[Any] = None,
        fts: Optional[bool] = False,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents most similar to the query

        Args:
            query: String to query the vectorstore with.
            k: Number of documents to return.
            filter (Optional[Dict]): Optional filter arguments
                sql_filter(Optional[string]): SQL filter to apply to the query.
                prefilter(Optional[bool]): Whether to apply the filter prior
                                             to the vector search.
        Raises:
            ValueError: If the specified table is not found in the database.

        Returns:
            List of documents most similar to the query.
        """
        res = self.similarity_search_with_score(
            query=query, k=k, name=name, filter=filter, fts=fts, score=False, **kwargs
        )
        return res

    def max_marginal_relevance_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
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
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        if k is None:
            k = self.limit

        if self._embedding is None:
            raise ValueError(
                "For MMR search, you must specify an embedding function oncreation."
            )

        embedding = self._embedding.embed_query(query)
        docs = self.max_marginal_relevance_search_by_vector(
            embedding,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
        )
        return docs

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
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
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """

        results = self._query(
            query=embedding,
            k=fetch_k,
            filter=filter,
            **kwargs,
        )
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            results["vector"].to_pylist(),
            k=k or self.limit,
            lambda_mult=lambda_mult,
        )

        candidates = self.results_to_docs(results)

        selected_results = [r for i, r in enumerate(candidates) if i in mmr_selected]
        return selected_results

    @classmethod
    def from_texts(
        cls: Type[LanceDB],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        connection: Optional[Any] = None,
        vector_key: Optional[str] = "vector",
        id_key: Optional[str] = "id",
        text_key: Optional[str] = "text",
        table_name: Optional[str] = "vectorstore",
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        mode: Optional[str] = "overwrite",
        distance: Optional[str] = "l2",
        reranker: Optional[Any] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        **kwargs: Any,
    ) -> LanceDB:
        instance = LanceDB(
            connection=connection,
            embedding=embedding,
            vector_key=vector_key,
            id_key=id_key,
            text_key=text_key,
            table_name=table_name,
            api_key=api_key,
            region=region,
            mode=mode,
            distance=distance,
            reranker=reranker,
            relevance_score_fn=relevance_score_fn,
            **kwargs,
        )
        instance.add_texts(texts, metadatas=metadatas)

        return instance

    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        filter: Optional[str] = None,
        drop_columns: Optional[List[str]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Allows deleting rows by filtering, by ids or drop columns from the table.

        Args:
            filter: Provide a string SQL expression -  "{col} {operation} {value}".
            ids: Provide list of ids to delete from the table.
            drop_columns: Provide list of columns to drop from the table.
            delete_all: If True, delete all rows from the table.
        """
        tbl = self.get_table(name)
        if filter:
            tbl.delete(filter)
        elif ids:
            tbl.delete(f"{self._id_key} in ('{{}}')".format(",".join(ids)))
        elif drop_columns:
            if self.api_key is not None:
                raise NotImplementedError(
                    "Column operations currently not supported in LanceDB Cloud."
                )
            else:
                tbl.drop_columns(drop_columns)
        elif delete_all:
            tbl.delete("true")
        else:
            raise ValueError("Provide either filter, ids, drop_columns or delete_all")
