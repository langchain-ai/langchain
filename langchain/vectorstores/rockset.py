"""Wrapper around Rockset vector database."""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Iterable, List, Optional, Tuple

# Make sure you have rockset client installed. If not, you can
# install it with `pip install rockset`.
from rockset import RocksetClient, ApiException
from rockset.models import QueryResponse

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

logger = logging.getLogger(__name__)


class Rockset(VectorStore):
    """Wrapper arpund Rockset vector database.

    To use, you should have the `rockset` python package installed. Note that to use this,
    the collection being used must already exist in your Rockset instance.
    You must also ensure you use a Rockset ingest transformation to apply `VECTOR_ENFORCE`
    on the column being used to store `embedding_key` in the collection.
    See: https://rockset.com/blog/introducing-vector-search-on-rockset/ for more details

    Example:
        .. code-block:: python

            from langchain.vectorstores import Rockset
            from langchain.embeddings.openai import OpenAIEmbeddings
            import rockset

            # Make sure you use the right host (region) for your Rockset instance, and APIKEY has
            # both read-write access to your collection.
            # Defining the host is optional and defaults to *https://api.use1a1.rockset.com*

            rs = rockset.Rockset.Client(host=rockset.Regions.use1a1, api_key="***")
            collection_name = "langchain_demo"
            embeddings = OpenAIEmbeddings()
            vectorstore = Rockset(rs, collection_name, embeddings, "description", "description_embedding")

    """

    def __init__(
        self,
        client: Any,
        collection_name: str,
        embeddings: Embeddings,
        text_key: str,
        embedding_key: str,
    ):
        """Initialize with Rockset client.
        Args:
            client: Rockset client object
            collection: Rockset collection to insert docs / query
            embeddings: Langchain Embeddings object to use to generate embedding for given text.
            text_key: column in Rockset collection to use to store the text
            embedding_key: column in Rockset collection to use to store the embedding.
                           Note: We must apply `VECTOR_ENFORCE()` on this column via
                           Rockset ingest transformation.

        """
        if not isinstance(client, RocksetClient):
            raise ValueError(
                f"client should be an instance of rockset.RocksetClient, "
                f"got {type(client)}"
            )
        self._client = client
        self._collection_name = collection_name
        self._embeddings = embeddings
        self._text_key = text_key
        self._embedding_key = embedding_key

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore

                Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            batch_size: Send documents in batches to rockset.

        Returns:
            List of ids from adding the texts into the vectorstore.

        """
        batch = []
        for i, text in enumerate(texts):
            if i % batch_size == 0:
                # send the current batch
                self._client.Documents.add_documents(
                    collection=self._collection, data=batch
                )
                batch = []
            doc = metadatas[i] if metadatas else {}
            if ids:
                doc["_id"] = ids[i]
            doc[self._text_key] = text
            doc[self._embedding_key] = self._embedding_function(text)
            batch.append(doc)

    # Rockset supports these vector distance functions.
    class DistanceFunction(Enum):
        COSINE_SIM = "COSINE_SIM"
        EUCLIDEAN_DIST = "EUCLIDEAN_DIST"
        DOT_PRODUCT = "DOT_PRODUCT"

    def _build_query_sql(
        self,
        query_embedding: List[float],
        distance_func: DistanceFunction,
        k: int = 4,
        where_str: Optional[str] = None,
    ) -> str:
        """Builds Rockset SQL query to query similar vectors to query_vector"""

        q_embedding_str = ",".join(map(str, query_embedding))
        distance_str = f"""{distance_func.value}({self._embedding_key}, [{q_embedding_str}]) as dist"""
        q_str = f"""
            SELECT * EXCEPT ({self._embedding_key}), {distance_str}
            FROM {self._collection_name}
            WHERE {where_str}
            ORDER BY dist DESC
            LIMIT {str(k)}
            """
        return q_str

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        distance_func: DistanceFunction,
        k: int = 4,
        where_str: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a similarity search with Rockset

        Args:
            query (str): Text to look up documents similar to.
            distance_func (DistanceFunction): how to compute distance between two vectors in Rockset.
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.
            where_str (Optional[str], optional): Metadata filters supplied as a SQL `where`
                condition string. Defaults to None.
                eg. "price<=70.0 AND brand='Nintendo'"

            NOTE: Please do not let end-user to fill this and always be aware
                  of SQL injection.
            TODO:

        Returns:
            List[Tuple[Document, float]]: List of documents with their relevance score
        """
        return self.similarity_search_by_vector(
            self._embeddings.embed_query(query),
            distance_func,
            k,
            where_str,
            **kwargs,
        )

    def similarity_search(
        self,
        query: str,
        distance_func: DistanceFunction,
        k: int = 4,
        where_str: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Same as `similarity_search_with_relevance_scores` but doesn't return the scores."""
        docs_and_scores = self.similarity_search_with_score(
            query, distance_func, k, where_str, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_by_vector(
        self,
        query_embedding: List[float],
        distance_func: DistanceFunction,
        k: int = 4,
        where_str: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Accepts a query_embedding (vector), and returns documents whose similar embeddings."""

        q_str = self._build_query_sql(query_embedding, distance_func, k, where_str)
        try:
            results = self._client.Queries.query(sql={"query": q_str})
        except ApiException as e:
            logger.error("Exception when querying Rockset: %s\n", e)
            return []
        finalResult = []
        for r in results:
            metadata = {}
            assert isinstance(
                r, QueryResponse
            ), "Rockset query result must be of type `QueryResponse`"
            for k, v in r.attribute_map["results"]:
                if k == self._text_key:
                    assert isinstance(
                        v, str
                    ), "page content stored in column `{}` must be of type `str`. \
                        But found: `{}`".format(
                        self._text_key, type(v)
                    )
                    page_content = v
                elif k == "dist":
                    assert isinstance(
                        v, float
                    ), "Computed distance between vectors must of type `float`. But found {}".format(
                        type(v)
                    )
                    score = v
                else:
                    metadata[k] = v
            finalResult.append(
                tuple(Document(page_content=page_content, metadata=metadata), score)
            )
        return finalResult

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        embeddings: Embeddings,
        metadatas: Optional[List[dict]] = None,
        client: Any = None,
        collection_name: str = "",
        text_key: str = "",
        embedding_key: str = "",
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
    ) -> Rockset:
        """Create Rockset wrapper with existing texts.
        This is intended as a quicker way to get started.
        """

        # Sanitize imputs
        assert client is not None
        assert collection_name != ""
        assert text_key != ""
        assert embedding_key != ""

        rockset = cls(client, collection_name, embeddings, text_key, embedding_key)
        rockset.add_texts(texts, metadatas, ids, batch_size)
        return rockset
