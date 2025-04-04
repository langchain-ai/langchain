from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from meilisearch import Client


def _create_client(
    client: Optional[Client] = None,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Client:
    try:
        import meilisearch
    except ImportError:
        raise ImportError(
            "Could not import meilisearch python package. "
            "Please install it with `pip install meilisearch`."
        )
    if not client:
        url = url or get_from_env("url", "MEILI_HTTP_ADDR")
        try:
            api_key = api_key or get_from_env("api_key", "MEILI_MASTER_KEY")
        except Exception:
            pass
        client = meilisearch.Client(url=url, api_key=api_key)
    elif not isinstance(client, meilisearch.Client):
        raise ValueError(
            f"client should be an instance of meilisearch.Client, got {type(client)}"
        )
    try:
        client.version()
    except ValueError as e:
        raise ValueError(f"Failed to connect to Meilisearch: {e}")
    return client


class Meilisearch(VectorStore):
    """`Meilisearch` vector store.

    To use this, you need to have `meilisearch` python package installed,
    and a running Meilisearch instance.

    To learn more about Meilisearch Python, refer to the in-depth
    Meilisearch Python documentation: https://meilisearch.github.io/meilisearch-python/.

    See the following documentation for how to run a Meilisearch instance:
    https://www.meilisearch.com/docs/learn/getting_started/quick_start.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import Meilisearch
            from langchain_community.embeddings.openai import OpenAIEmbeddings
            import meilisearch

            # api_key is optional; provide it if your meilisearch instance requires it
            client = meilisearch.Client(url='http://127.0.0.1:7700', api_key='***')
            embeddings = OpenAIEmbeddings()
            embedders = {
                "theEmbedderName": {
                    "source": "userProvided",
                    "dimensions": "1536"
                }
            }
            vectorstore = Meilisearch(
                embedding=embeddings,
                embedders=embedders,
                client=client,
                index_name='langchain_demo',
                text_key='text')
    """

    def __init__(
        self,
        embedding: Embeddings,
        client: Optional[Client] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        index_name: str = "langchain-demo",
        text_key: str = "text",
        metadata_key: str = "metadata",
        *,
        embedders: Optional[Dict[str, Any]] = None,
    ):
        """Initialize with Meilisearch client."""
        client = _create_client(client=client, url=url, api_key=api_key)

        self._client = client
        self._index_name = index_name
        self._embedding = embedding
        self._text_key = text_key
        self._metadata_key = metadata_key
        self._embedders = embedders
        self._embedders_settings = self._client.index(
            str(self._index_name)
        ).update_embedders(embedders)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        embedder_name: Optional[str] = "default",
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embedding and add them to the vector store.

        Args:
            texts (Iterable[str]): Iterable of strings/text to add to the vectorstore.
            embedder_name: Name of the embedder. Defaults to "default".
            metadatas (Optional[List[dict]]): Optional list of metadata.
                Defaults to None.
            ids Optional[List[str]]: Optional list of IDs.
                Defaults to None.

        Returns:
            List[str]: List of IDs of the texts added to the vectorstore.
        """
        texts = list(texts)

        # Embed and create the documents
        docs = []
        if ids is None:
            ids = [uuid.uuid4().hex for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]
        embedding_vectors = self._embedding.embed_documents(texts)

        for i, text in enumerate(texts):
            id = ids[i]
            metadata = metadatas[i]
            metadata[self._text_key] = text
            embedding = embedding_vectors[i]
            docs.append(
                {
                    "id": id,
                    "_vectors": {f"{embedder_name}": embedding},
                    f"{self._metadata_key}": metadata,
                }
            )

        # Send to Meilisearch
        self._client.index(str(self._index_name)).add_documents(docs)
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        embedder_name: Optional[str] = "default",
        **kwargs: Any,
    ) -> List[Document]:
        """Return meilisearch documents most similar to the query.

        Args:
            query (str): Query text for which to find similar documents.
            embedder_name: Name of the embedder to be used. Defaults to "default".
            k (int): Number of documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata.
                Defaults to None.

        Returns:
            List[Document]: List of Documents most similar to the query
            text and score for each.
        """
        docs_and_scores = self.similarity_search_with_score(
            query=query,
            embedder_name=embedder_name,
            k=k,
            filter=filter,
            kwargs=kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        embedder_name: Optional[str] = "default",
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return meilisearch documents most similar to the query, along with scores.

        Args:
            query (str): Query text for which to find similar documents.
            embedder_name: Name of the embedder to be used. Defaults to "default".
            k (int): Number of documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata.
                Defaults to None.

        Returns:
            List[Document]: List of Documents most similar to the query
            text and score for each.
        """
        _query = self._embedding.embed_query(query)

        docs = self.similarity_search_by_vector_with_scores(
            embedding=_query,
            embedder_name=embedder_name,
            k=k,
            filter=filter,
            kwargs=kwargs,
        )
        return docs

    def similarity_search_by_vector_with_scores(
        self,
        embedding: List[float],
        embedder_name: Optional[str] = "default",
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return meilisearch documents most similar to embedding vector.

        Args:
            embedding (List[float]): Embedding to look up similar documents.
            embedder_name: Name of the embedder to be used. Defaults to "default".
            k (int): Number of documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata.
                Defaults to None.

        Returns:
            List[Document]: List of Documents most similar to the query
                vector and score for each.
        """
        docs = []
        results = self._client.index(str(self._index_name)).search(
            "",
            {
                "vector": embedding,
                "hybrid": {"semanticRatio": 1.0, "embedder": embedder_name},
                "limit": k,
                "filter": filter,
                "showRankingScore": True,
            },
        )

        for result in results["hits"]:
            metadata = result[self._metadata_key]
            if self._text_key in metadata:
                text = metadata.pop(self._text_key)
                semantic_score = result["_rankingScore"]
                docs.append(
                    (
                        Document(page_content=text, metadata=metadata),
                        semantic_score,
                    )
                )

        return docs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        embedder_name: Optional[str] = "default",
        **kwargs: Any,
    ) -> List[Document]:
        """Return meilisearch documents most similar to embedding vector.

        Args:
            embedding (List[float]): Embedding to look up similar documents.
            embedder_name: Name of the embedder to be used. Defaults to "default".
            k (int): Number of documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata.
                Defaults to None.

        Returns:
            List[Document]: List of Documents most similar to the query
                vector and score for each.
        """
        docs = self.similarity_search_by_vector_with_scores(
            embedding=embedding,
            embedder_name=embedder_name,
            k=k,
            filter=filter,
            kwargs=kwargs,
        )
        return [doc for doc, _ in docs]

    @classmethod
    def from_texts(
        cls: Type[Meilisearch],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        client: Optional[Client] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        index_name: str = "langchain-demo",
        ids: Optional[List[str]] = None,
        text_key: Optional[str] = "text",
        metadata_key: Optional[str] = "metadata",
        embedders: Dict[str, Any] = {},
        embedder_name: Optional[str] = "default",
        **kwargs: Any,
    ) -> Meilisearch:
        """Construct Meilisearch wrapper from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Adds the documents to a provided Meilisearch index.

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import Meilisearch
                from langchain_community.embeddings import OpenAIEmbeddings
                import meilisearch

                # The environment should be the one specified next to the API key
                # in your Meilisearch console
                client = meilisearch.Client(url='http://127.0.0.1:7700', api_key='***')
                embedding = OpenAIEmbeddings()
                embedders: Embedders index setting.
                embedder_name: Name of the embedder. Defaults to "default".
                docsearch = Meilisearch.from_texts(
                    client=client,
                    embedding=embedding,
                )
        """
        client = _create_client(client=client, url=url, api_key=api_key)

        vectorstore = cls(
            embedding=embedding,
            embedders=embedders,
            client=client,
            index_name=index_name,
        )
        vectorstore.add_texts(
            texts=texts,
            embedder_name=embedder_name,
            metadatas=metadatas,
            ids=ids,
            text_key=text_key,
            metadata_key=metadata_key,
        )
        return vectorstore
