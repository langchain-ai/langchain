import asyncio
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class SurrealDBStore(VectorStore):
    """
    SurrealDB as Vector Store.

    To use, you should have the ``surrealdb`` python package installed.

    Args:
        embedding_function: Embedding function to use.
        dburl: SurrealDB connection url
        ns: surrealdb namespace for the vector store. (default: "langchain")
        db: surrealdb database for the vector store. (default: "database")
        collection: surrealdb collection for the vector store.
            (default: "documents")

        (optional) db_user and db_pass: surrealdb credentials

    Example:
        .. code-block:: python

            from langchain.vectorstores.surrealdb import SurrealDBStore
            from langchain.embeddings import HuggingFaceEmbeddings

            embedding_function = HuggingFaceEmbeddings()
            dburl = "ws://localhost:8000/rpc"
            ns = "langchain"
            db = "docstore"
            collection = "documents"
            db_user = "root"
            db_pass = "root"

            sdb = SurrealDBStore.from_texts(
                    texts=texts,
                    embedding=embedding_function,
                    dburl,
                    ns, db, collection,
                    db_user=db_user, db_pass=db_pass)
    """

    def __init__(
        self,
        embedding_function: Embeddings,
        **kwargs: Any,
    ) -> None:
        from surrealdb import Surreal

        self.collection = kwargs.pop("collection", "documents")
        self.ns = kwargs.pop("ns", "langchain")
        self.db = kwargs.pop("db", "database")
        self.dburl = kwargs.pop("dburl", "ws://localhost:8000/rpc")
        self.embedding_function = embedding_function
        self.sdb = Surreal(self.dburl)
        self.kwargs = kwargs

    async def initialize(self) -> None:
        """
        Initialize connection to surrealdb database
        and authenticate if credentials are provided
        """
        await self.sdb.connect(self.dburl)
        if "db_user" in self.kwargs and "db_pass" in self.kwargs:
            user = self.kwargs.get("db_user")
            password = self.kwargs.get("db_pass")
            await self.sdb.signin({"user": user, "pass": password})

        await self.sdb.use(self.ns, self.db)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return (
            self.embedding_function
            if isinstance(self.embedding_function, Embeddings)
            else None
        )

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add list of text along with embeddings to the vector store asynchronously

        Args:
            texts (Iterable[str]): collection of text to add to the database

        Returns:
            List of ids for the newly inserted documents
        """
        embeddings = self.embedding_function.embed_documents(list(texts))
        ids = []
        for idx, text in enumerate(texts):
            data = {"text": text, "embedding": embeddings[idx]}
            if metadatas is not None and idx < len(metadatas):
                data["metadata"] = metadatas[idx]
            record = await self.sdb.create(
                self.collection,
                data,
            )
            ids.append(record[0]["id"])
        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add list of text along with embeddings to the vector store

        Args:
            texts (Iterable[str]): collection of text to add to the database

        Returns:
            List of ids for the newly inserted documents
        """

        async def _add_texts(
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
        ) -> List[str]:
            await self.initialize()
            return await self.aadd_texts(texts, metadatas, **kwargs)

        return asyncio.run(_add_texts(texts, metadatas, **kwargs))

    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete by document ID asynchronously.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise.
        """

        if ids is None:
            await self.sdb.delete(self.collection)
            return True
        else:
            if isinstance(ids, str):
                await self.sdb.delete(ids)
                return True
            else:
                if isinstance(ids, list) and len(ids) > 0:
                    _ = [await self.sdb.delete(id) for id in ids]
                    return True
        return False

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete by document ID.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise.
        """

        async def _delete(ids: Optional[List[str]], **kwargs: Any) -> Optional[bool]:
            await self.initialize()
            return await self.adelete(ids=ids, **kwargs)

        return asyncio.run(_delete(ids, **kwargs))

    async def _asimilarity_search_by_vector_with_score(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search for query embedding asynchronously
        and return documents and scores

        Args:
            embedding (List[float]): Query embedding.
            k (int): Number of results to return. Defaults to 4.

        Returns:
            List of Documents most similar along with scores
        """
        args = {
            "collection": self.collection,
            "embedding": embedding,
            "k": k,
            "score_threshold": kwargs.get("score_threshold", 0),
        }
        query = """select id, text, metadata,
        vector::similarity::cosine(embedding,{embedding}) as similarity
        from {collection}
        where vector::similarity::cosine(embedding,{embedding}) >= {score_threshold}
        order by similarity desc LIMIT {k}
        """.format(**args)
        results = await self.sdb.query(query)

        if len(results) == 0:
            return []

        return [
            (
                Document(
                    page_content=result["text"],
                    metadata={"id": result["id"], **result["metadata"]},
                ),
                result["similarity"],
            )
            for result in results[0]["result"]
        ]

    async def asimilarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search asynchronously and return relevance scores

        Args:
            query (str): Query
            k (int): Number of results to return. Defaults to 4.

        Returns:
            List of Documents most similar along with relevance scores
        """
        query_embedding = self.embedding_function.embed_query(query)
        return [
            (document, similarity)
            for document, similarity in (
                await self._asimilarity_search_by_vector_with_score(
                    query_embedding, k, **kwargs
                )
            )
        ]

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search synchronously and return relevance scores

        Args:
            query (str): Query
            k (int): Number of results to return. Defaults to 4.

        Returns:
            List of Documents most similar along with relevance scores
        """

        async def _similarity_search_with_relevance_scores() -> (
            List[Tuple[Document, float]]
        ):
            await self.initialize()
            return await self.asimilarity_search_with_relevance_scores(
                query, k, **kwargs
            )

        return asyncio.run(_similarity_search_with_relevance_scores())

    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search asynchronously and return distance scores

        Args:
            query (str): Query
            k (int): Number of results to return. Defaults to 4.

        Returns:
            List of Documents most similar along with relevance distance scores
        """
        query_embedding = self.embedding_function.embed_query(query)
        return [
            (document, similarity)
            for document, similarity in (
                await self._asimilarity_search_by_vector_with_score(
                    query_embedding, k, **kwargs
                )
            )
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search synchronously and return distance scores

        Args:
            query (str): Query
            k (int): Number of results to return. Defaults to 4.

        Returns:
            List of Documents most similar along with relevance distance scores
        """

        async def _similarity_search_with_score() -> List[Tuple[Document, float]]:
            await self.initialize()
            return await self.asimilarity_search_with_score(query, k, **kwargs)

        return asyncio.run(_similarity_search_with_score())

    async def asimilarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Run similarity search on query embedding asynchronously

        Args:
            embedding (List[float]): Query embedding
            k (int): Number of results to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query
        """
        return [
            document
            for document, _ in await self._asimilarity_search_by_vector_with_score(
                embedding, k, **kwargs
            )
        ]

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Run similarity search on query embedding

        Args:
            embedding (List[float]): Query embedding
            k (int): Number of results to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query
        """

        async def _similarity_search_by_vector() -> List[Document]:
            await self.initialize()
            return await self.asimilarity_search_by_vector(embedding, k, **kwargs)

        return asyncio.run(_similarity_search_by_vector())

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Run similarity search on query asynchronously

        Args:
            query (str): Query
            k (int): Number of results to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query
        """
        query_embedding = self.embedding_function.embed_query(query)
        return await self.asimilarity_search_by_vector(query_embedding, k, **kwargs)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Run similarity search on query

        Args:
            query (str): Query
            k (int): Number of results to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query
        """

        async def _similarity_search() -> List[Document]:
            await self.initialize()
            return await self.asimilarity_search(query, k, **kwargs)

        return asyncio.run(_similarity_search())

    @classmethod
    async def afrom_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "SurrealDBStore":
        """Create SurrealDBStore from list of text asynchronously

        Args:
            texts (List[str]): list of text to vectorize and store
            embedding (Optional[Embeddings]): Embedding function.
            dburl (str): SurrealDB connection url
                (default: "ws://localhost:8000/rpc")
            ns (str): surrealdb namespace for the vector store.
                (default: "langchain")
            db (str): surrealdb database for the vector store.
                (default: "database")
            collection (str): surrealdb collection for the vector store.
                (default: "documents")

            (optional) db_user and db_pass: surrealdb credentials

        Returns:
            SurrealDBStore object initialized and ready for use."""

        sdb = cls(embedding, **kwargs)
        await sdb.initialize()
        await sdb.aadd_texts(texts, metadatas, **kwargs)
        return sdb

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "SurrealDBStore":
        """Create SurrealDBStore from list of text

        Args:
            texts (List[str]): list of text to vectorize and store
            embedding (Optional[Embeddings]): Embedding function.
            dburl (str): SurrealDB connection url
            ns (str): surrealdb namespace for the vector store.
                (default: "langchain")
            db (str): surrealdb database for the vector store.
                (default: "database")
            collection (str): surrealdb collection for the vector store.
                (default: "documents")

            (optional) db_user and db_pass: surrealdb credentials

        Returns:
            SurrealDBStore object initialized and ready for use."""
        sdb = asyncio.run(cls.afrom_texts(texts, embedding, metadatas, **kwargs))
        return sdb
