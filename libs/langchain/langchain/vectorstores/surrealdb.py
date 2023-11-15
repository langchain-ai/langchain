import asyncio
from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from surrealdb import Surreal

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)


class SurrealDBStore(VectorStore):
    """
    SurrealDB as Vector Store.

    To use, you should have the ``surrealdb`` python package installed.

    Args:
        dburl: SurrealDB connection url
        embedding_function: Embedding function to use.
        ns: surrealdb namespace for the vector store. (default: "langchain")
        db: surrealdb database for the vector store. (default: "database")
        collection: surrealdb collection for the vector store. (default: "documents")

        (optional) db_user and db_pass: surrealdb credentials

    Example:
        .. code-block:: python

            from langchain.vectorstores.surrealdb import SurrealDBStore
            from langchain.embeddings import HuggingFaceEmbeddings

            dburl = "ws://localhost:8000/rpc"
            embedding_function = HuggingFaceEmbeddings()
            ns = "langchain"
            db = "docstore"
            collection = "documents"
            db_user = "root"
            db_pass = "root"

            sdb = SurrealDBStore.from_texts(
                    dburl,
                    texts=texts,
                    embedding_function=embedding_function,
                    ns, db, collection,
                    db_user=db_user, db_pass=db_pass)
    """

    def __init__(self, dburl: str,
                 embedding_function: Optional[Embeddings] = None,
                 ns: str = "langchain",
                 db: str = "database",
                 collection: str = "documents",
                 **kwargs: Any) -> None:
        self.collection = collection
        self.ns = ns
        self.db = db
        self.dburl = dburl
        self.embedding_function = embedding_function
        self.sdb = Surreal()
        self.kwargs = kwargs

    async def initialize(self):
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
            texts (Iterable[str]: collection of text to add to the database

        Returns:
            List of ids for the newly inserted documents
        """
        embeddings = self.embedding_function.embed_documents(texts)
        ids = []
        for idx, text in enumerate(texts):
            record = await self.sdb.create(
                self.collection,
                {
                    "text": text,
                    "embedding": embeddings[idx]
                }
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
            texts (Iterable[str]: collection of text to add to the database

        Returns:
            List of ids for the newly inserted documents
        """
        return asyncio.run(self.aadd_texts(texts, metadatas, **kwargs))

    async def _asimilarity_search_by_vector_with_score(
            self, embeddings: List[float], k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search for query embeddings asynchronously
        and return documents and scores

        Args:
            embeddings (Lost[float]): Query embeddings.
            k (int): Number of results to return. Defaults to 4.

        Returns:
            List of Documents most similar along with scores
        """
        args = {
            "collection": self.collection,
            "embedding": embeddings,
            "k": k,
            "score_threshold": kwargs.get("score_threshold", 0)
        }
        query = '''select id, text,
        vector::similarity::cosine(embedding,{embedding}) as similarity
        from {collection}
        where vector::similarity::cosine(embedding,{embedding}) >= {score_threshold}
        order by similarity desc LIMIT {k}
        '''.format(**args)

        results = await self.sdb.query(query)

        return [
            (Document(
                page_content=result["text"],
                metadata={"id": result["id"]}
            ), result["similarity"]) for result in results[0]["result"]
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
        return [(document, similarity) for document, similarity in
                await self._asimilarity_search_by_vector_with_score(
                    query_embedding, k, **kwargs)]

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
        async def _similarity_search_with_relevance_scores():
            await self.initialize()
            return await self.asimilarity_search_with_relevance_scores(
                    query, k, **kwargs)
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
        return [(document, similarity) for document, similarity in
                await self._asimilarity_search_by_vector_with_score(
                    query_embedding, k, **kwargs
                )]

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs):
        """Run similarity search synchronously and return distance scores

        Args:
            query (str): Query
            k (int): Number of results to return. Defaults to 4.

        Returns:
            List of Documents most similar along with relevance distance scores
        """
        async def _similarity_search_with_score():
            await self.initialize()
            return await self.asimilarity_search_with_score(query, k, **kwargs)
        return asyncio.run(_similarity_search_with_score())

    async def asimilarity_search_by_vector(
            self, embeddings: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Run similarity search on query embeddings asynchronously

        Args:
            embeddings (List[float]): Query embeddings
            k (int): Number of results to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query
        """
        return [document for document, _ in
                await self._asimilarity_search_by_vector_with_score(
                    embeddings, k, **kwargs
                )]

    def similarity_search_by_vector(
            self, embeddings: List[float], k: int = 4, **kwargs: Any):
        """Run similarity search on query embeddings

        Args:
            embeddings (List[float]): Query embeddings
            k (int): Number of results to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query
        """
        async def _similarity_search_by_vector():
            await self.initialize()
            return await self.asimilarity_search_by_vector(
                    embeddings, k, **kwargs)
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
        return await self.asimilarity_search_by_vector(
                query_embedding, k, **kwargs)

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
        async def _similarity_search():
            await self.initialize()
            return await self.asimilarity_search(query, k, **kwargs)
        return asyncio.run(_similarity_search())

    @classmethod
    async def afrom_texts(
        cls,
        dburl: str,
        texts: List[str],
        embedding_function: Optional[Embeddings],
        ns: str = "langchain",
        db: str = "database",
        collection: str = "documents",
        **kwargs: Any,
    ) -> 'SurrealDBStore':
        """Create SurrealDBStore from list of text asynchronously

        Args:
            dburl (str): SurrealDB connection url
            texts (List[str]): list of text to vectorize and store
            embedding_function (Optional[Embeddings]): Embedding function to use.
            ns (str): surrealdb namespace for the vector store. (default: "langchain")
            db (str): surrealdb database for the vector store. (default: "database")
            collection (str): surrealdb collection for the vector store. (default: "documents")

            (optional) db_user and db_pass: surrealdb credentials

        Returns:
            SurrealDBStore object initialized and ready for use."""
        sdb = cls(dburl, embedding_function, ns, db, collection, **kwargs)
        await sdb.initialize()
        await sdb.aadd_texts(texts)
        return sdb

    @classmethod
    def from_texts(
        cls,
        dburl: str,
        texts: List[str],
        embedding_function: Optional[Embeddings],
        ns: str = "langchain",
        db: str = "database",
        collection: str = "documents",
        **kwargs: Any,
    ) -> 'SurrealDBStore':
        """Create SurrealDBStore from list of text

        Args:
            dburl (str): SurrealDB connection url
            texts (List[str]): list of text to vectorize and store
            embedding_function (Optional[Embeddings]): Embedding function to use.
            ns (str): surrealdb namespace for the vector store. (default: "langchain")
            db (str): surrealdb database for the vector store. (default: "database")
            collection (str): surrealdb collection for the vector store. (default: "documents")

            (optional) db_user and db_pass: surrealdb credentials

        Returns:
            SurrealDBStore object initialized and ready for use."""
        sdb = asyncio.run(cls.afrom_texts(dburl, texts, embedding_function,
                                          ns, db, collection, **kwargs))
        return sdb
