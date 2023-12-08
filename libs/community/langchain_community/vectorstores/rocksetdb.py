from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class Rockset(VectorStore):
    """`Rockset` vector store.

    To use, you should have the `rockset` python package installed. Note that to use
    this, the collection being used must already exist in your Rockset instance.
    You must also ensure you use a Rockset ingest transformation to apply
    `VECTOR_ENFORCE` on the column being used to store `embedding_key` in the
    collection.
    See: https://rockset.com/blog/introducing-vector-search-on-rockset/ for more details

    Everything below assumes `commons` Rockset workspace.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import Rockset
            from langchain_community.embeddings.openai import OpenAIEmbeddings
            import rockset

            # Make sure you use the right host (region) for your Rockset instance
            # and APIKEY has both read-write access to your collection.

            rs = rockset.RocksetClient(host=rockset.Regions.use1a1, api_key="***")
            collection_name = "langchain_demo"
            embeddings = OpenAIEmbeddings()
            vectorstore = Rockset(rs, collection_name, embeddings,
                "description", "description_embedding")

    """

    def __init__(
        self,
        client: Any,
        embeddings: Embeddings,
        collection_name: str,
        text_key: str,
        embedding_key: str,
        workspace: str = "commons",
    ):
        """Initialize with Rockset client.
        Args:
            client: Rockset client object
            collection: Rockset collection to insert docs / query
            embeddings: Langchain Embeddings object to use to generate
                        embedding for given text.
            text_key: column in Rockset collection to use to store the text
            embedding_key: column in Rockset collection to use to store the embedding.
                           Note: We must apply `VECTOR_ENFORCE()` on this column via
                           Rockset ingest transformation.

        """
        try:
            from rockset import RocksetClient
        except ImportError:
            raise ImportError(
                "Could not import rockset client python package. "
                "Please install it with `pip install rockset`."
            )

        if not isinstance(client, RocksetClient):
            raise ValueError(
                f"client should be an instance of rockset.RocksetClient, "
                f"got {type(client)}"
            )
        # TODO: check that `collection_name` exists in rockset. Create if not.
        self._client = client
        self._collection_name = collection_name
        self._embeddings = embeddings
        self._text_key = text_key
        self._embedding_key = embedding_key
        self._workspace = workspace

        try:
            self._client.set_application("langchain")
        except AttributeError:
            # ignore
            pass

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        **kwargs: Any,
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
        batch: list[dict] = []
        stored_ids = []

        for i, text in enumerate(texts):
            if len(batch) == batch_size:
                stored_ids += self._write_documents_to_rockset(batch)
                batch = []
            doc = {}
            if metadatas and len(metadatas) > i:
                doc = metadatas[i]
            if ids and len(ids) > i:
                doc["_id"] = ids[i]
            doc[self._text_key] = text
            doc[self._embedding_key] = self._embeddings.embed_query(text)
            batch.append(doc)
        if len(batch) > 0:
            stored_ids += self._write_documents_to_rockset(batch)
            batch = []
        return stored_ids

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        client: Any = None,
        collection_name: str = "",
        text_key: str = "",
        embedding_key: str = "",
        ids: Optional[List[str]] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> Rockset:
        """Create Rockset wrapper with existing texts.
        This is intended as a quicker way to get started.
        """

        # Sanitize inputs
        assert client is not None, "Rockset Client cannot be None"
        assert collection_name, "Collection name cannot be empty"
        assert text_key, "Text key name cannot be empty"
        assert embedding_key, "Embedding key cannot be empty"

        rockset = cls(client, embedding, collection_name, text_key, embedding_key)
        rockset.add_texts(texts, metadatas, ids, batch_size)
        return rockset

    # Rockset supports these vector distance functions.
    class DistanceFunction(Enum):
        COSINE_SIM = "COSINE_SIM"
        EUCLIDEAN_DIST = "EUCLIDEAN_DIST"
        DOT_PRODUCT = "DOT_PRODUCT"

        # how to sort results for "similarity"
        def order_by(self) -> str:
            if self.value == "EUCLIDEAN_DIST":
                return "ASC"
            return "DESC"

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        distance_func: DistanceFunction = DistanceFunction.COSINE_SIM,
        where_str: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a similarity search with Rockset

        Args:
            query (str): Text to look up documents similar to.
            distance_func (DistanceFunction): how to compute distance between two
                vectors in Rockset.
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.
            where_str (Optional[str], optional): Metadata filters supplied as a
                SQL `where` condition string. Defaults to None.
                eg. "price<=70.0 AND brand='Nintendo'"

            NOTE: Please do not let end-user to fill this and always be aware
                  of SQL injection.

        Returns:
            List[Tuple[Document, float]]: List of documents with their relevance score
        """
        return self.similarity_search_by_vector_with_relevance_scores(
            self._embeddings.embed_query(query),
            k,
            distance_func,
            where_str,
            **kwargs,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        distance_func: DistanceFunction = DistanceFunction.COSINE_SIM,
        where_str: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Same as `similarity_search_with_relevance_scores` but
        doesn't return the scores.
        """
        return self.similarity_search_by_vector(
            self._embeddings.embed_query(query),
            k,
            distance_func,
            where_str,
            **kwargs,
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        distance_func: DistanceFunction = DistanceFunction.COSINE_SIM,
        where_str: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Accepts a query_embedding (vector), and returns documents with
        similar embeddings."""

        docs_and_scores = self.similarity_search_by_vector_with_relevance_scores(
            embedding, k, distance_func, where_str, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = 4,
        distance_func: DistanceFunction = DistanceFunction.COSINE_SIM,
        where_str: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Accepts a query_embedding (vector), and returns documents with
        similar embeddings along with their relevance scores."""

        q_str = self._build_query_sql(embedding, distance_func, k, where_str)
        try:
            query_response = self._client.Queries.query(sql={"query": q_str})
        except Exception as e:
            logger.error("Exception when querying Rockset: %s\n", e)
            return []
        finalResult: list[Tuple[Document, float]] = []
        for document in query_response.results:
            metadata = {}
            assert isinstance(
                document, dict
            ), "document should be of type `dict[str,Any]`. But found: `{}`".format(
                type(document)
            )
            for k, v in document.items():
                if k == self._text_key:
                    assert isinstance(v, str), (
                        "page content stored in column `{}` must be of type `str`. "
                        "But found: `{}`"
                    ).format(self._text_key, type(v))
                    page_content = v
                elif k == "dist":
                    assert isinstance(v, float), (
                        "Computed distance between vectors must of type `float`. "
                        "But found {}"
                    ).format(type(v))
                    score = v
                elif k not in ["_id", "_event_time", "_meta"]:
                    # These columns are populated by Rockset when documents are
                    # inserted. No need to return them in metadata dict.
                    metadata[k] = v
            finalResult.append(
                (Document(page_content=page_content, metadata=metadata), score)
            )
        return finalResult

    # Helper functions

    def _build_query_sql(
        self,
        query_embedding: List[float],
        distance_func: DistanceFunction,
        k: int = 4,
        where_str: Optional[str] = None,
    ) -> str:
        """Builds Rockset SQL query to query similar vectors to query_vector"""

        q_embedding_str = ",".join(map(str, query_embedding))
        distance_str = f"""{distance_func.value}({self._embedding_key}, \
[{q_embedding_str}]) as dist"""
        where_str = f"WHERE {where_str}\n" if where_str else ""
        return f"""\
SELECT * EXCEPT({self._embedding_key}), {distance_str}
FROM {self._workspace}.{self._collection_name}
{where_str}\
ORDER BY dist {distance_func.order_by()}
LIMIT {str(k)}
"""

    def _write_documents_to_rockset(self, batch: List[dict]) -> List[str]:
        add_doc_res = self._client.Documents.add_documents(
            collection=self._collection_name, data=batch, workspace=self._workspace
        )
        return [doc_status._id for doc_status in add_doc_res.data]

    def delete_texts(self, ids: List[str]) -> None:
        """Delete a list of docs from the Rockset collection"""
        try:
            from rockset.models import DeleteDocumentsRequestData
        except ImportError:
            raise ImportError(
                "Could not import rockset client python package. "
                "Please install it with `pip install rockset`."
            )

        self._client.Documents.delete_documents(
            collection=self._collection_name,
            data=[DeleteDocumentsRequestData(id=i) for i in ids],
            workspace=self._workspace,
        )
