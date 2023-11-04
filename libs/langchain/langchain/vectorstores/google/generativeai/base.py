"""Google Generative AI Vector Store.

The GenAI Semantic Retriever API is a managed end-to-end service that allows
developers to create a corpus of documents to perform semantic search on
related passages given a user query. For more information visit:
https://developers.generativeai.google/guide
"""

import asyncio
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
    TypeVar,
    cast,
)

from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore

VST = TypeVar("VST", bound="GoogleVectorStore")
_import_err_msg = (
    "`google.generativeai` package not found, "
    "please run `pip install google-generativeai`"
)


class ServerSideEmbedding(Embeddings):
    """Do nothing embedding model where the embedding is done by the server."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return []


class DoesNotExistsException(Exception):
    def __init__(self, *, corpus_id: str, document_id: Optional[str] = None) -> None:
        if document_id is None:
            message = f"No such corpus {corpus_id}"
        else:
            message = f"No such document {document_id} under corpus {corpus_id}"
        super().__init__(message)


class GoogleVectorStore(VectorStore):
    """Google GenerativeAI Vector Store.

    Currently, it computes the embedding vectors on the server side.

    Example: Add texts to an existing corpus.

        store = GoogleVectorStore(corpus_id="123")
        store.add_documents(documents, document_id="456")

    Example: Create a new corpus.

        store = GoogleVectorStore.create_corpus(
            corpus_id="123", display_name="My Google corpus")

    Example: Query the corpus.

        store.as_retriever() \
            .get_relevant_documents("Who caught the gingerbread man?")

    You can also operate at Google's Document level.

    Example: Add texts to an existing Google Vector Store Document.

        doc_store = GoogleVectorStore(corpus_id="123", document_id="456")
        doc_store.add_documents(documents)

    Example: Create a new Google Vector Store Document.

        doc_store = GoogleVectorStore.create_document(
            corpus_id="123", document_id="456", display_name="My Google document")

    Example: Query the Google document.

        doc_store.as_retriever() \
            .get_relevant_documents("Who caught the gingerbread man?")

    For more details, see the class's methods.
    """

    _retriever: Any

    def __init__(
        self, *, corpus_id: str, document_id: Optional[str] = None, **kwargs: Any
    ):
        """Returns an existing Google Semantic Retriever corpus or document.

        If just the corpus ID is provided, the vector store operates over all
        documents within that corpus.

        If the document ID is provided, the vector store operates over just that
        document.

        Raises:
            DoesNotExistsException if the IDs do not match to anything on Google
                server. In this case, consider using `create_corpus` or
                `create_document` to create one.
        """
        try:
            from .semantic_retriever import SemanticRetriever
        except ImportError:
            raise ImportError(_import_err_msg)

        super().__init__(**kwargs)
        self._retriever = SemanticRetriever.from_ids(corpus_id, document_id)

    @classmethod
    def create_corpus(
        cls: Type[VST],
        corpus_id: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> VST:
        """Create a Google Semantic Retriever corpus.

        Args:
            corpus_id: The ID to use to create the new corpus. If not provided,
                Google server will provide one.
            display_name: The title of the new corpus. If not provided, Google
                server will provide one.

        Returns:
            An instance of vector store that points to the newly created corpus.
        """
        try:
            import langchain.vectorstores.google.generativeai.genai_extension as genaix
        except ImportError:
            raise ImportError(_import_err_msg)

        client = genaix.build_semantic_retriever()
        corpus = genaix.create_corpus(
            corpus_id=corpus_id, display_name=display_name, client=client
        )

        n = genaix.EntityName.from_str(corpus.name)
        return cls(corpus_id=n.corpus_id)

    @classmethod
    def create_document(
        cls: Type[VST],
        corpus_id: str,
        document_id: Optional[str] = None,
        display_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VST:
        """Create a Google Semantic Retriever document.

        Args:
            corpus_id: ID of an existing corpus.
            document_id: The ID to use to create the new Google Semantic
                Retriever document. If not provided, Google server will provide
                one.
            display_name: The title of the new document. If not provided, Google
                server will provide one.

        Returns:
            An instance of vector store that points to the newly created
            document.
        """
        try:
            import langchain.vectorstores.google.generativeai.genai_extension as genaix
        except ImportError:
            raise ImportError(_import_err_msg)

        client = genaix.build_semantic_retriever()
        document = genaix.create_document(
            corpus_id=corpus_id,
            document_id=document_id,
            display_name=display_name,
            metadata=metadata,
            client=client,
        )

        assert document.name is not None
        d = genaix.EntityName.from_str(document.name)
        return cls(corpus_id=d.corpus_id, document_id=d.document_id)

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict[str, Any]]] = None,
        *,
        corpus_id: Optional[str] = None,  # str required
        document_id: Optional[str] = None,  # str required
        **kwargs: Any,
    ) -> VST:
        """Returns a vector store of an existing document with the specified text.

        Args:
            corpus_id: REQUIRED. Must be an existing corpus.
            document_id: REQUIRED. Must be an existing document.
            texts: Texts to be loaded into the vector store.

        Returns:
            A vector store pointing to the specified Google Semantic Retriever
            Document.

        Raises:
            DoesNotExistsException if the IDs do not match to anything at
                Google server.
        """
        if corpus_id is None or document_id is None:
            raise NotImplementedError(
                "Must provide an existing corpus ID and document ID"
            )

        doc_store = cls(corpus_id=corpus_id, document_id=document_id, **kwargs)
        doc_store.add_texts(texts, metadatas)

        return doc_store

    @property
    def name(self) -> str:
        """Returns the name of the Google entity.

        You shouldn't need to care about this unless you want to access your
        corpus or document via Google Generative AI API.
        """
        try:
            from .semantic_retriever import SemanticRetriever
        except ImportError:
            raise ImportError(_import_err_msg)

        internal = cast(SemanticRetriever, self._retriever)
        return str(internal.name)

    @property
    def corpus_id(self) -> str:
        """Returns the corpus ID managed by this vector store."""
        try:
            from .semantic_retriever import SemanticRetriever
        except ImportError:
            raise ImportError(_import_err_msg)

        internal = cast(SemanticRetriever, self._retriever)
        id: str = internal.name.corpus_id
        return id

    @property
    def document_id(self) -> Optional[str]:
        """Returns the document ID managed by this vector store."""
        try:
            from .semantic_retriever import SemanticRetriever
        except ImportError:
            raise ImportError(_import_err_msg)

        internal = cast(SemanticRetriever, self._retriever)
        id: Optional[str] = internal.name.document_id
        return id

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        *,
        document_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store.

        If the vector store points to a corpus (instead of a document), you must
        also provide a `document_id`.

        Returns:
            Chunk's names created on Google servers.
        """
        try:
            from .semantic_retriever import SemanticRetriever
        except ImportError:
            raise ImportError(_import_err_msg)

        internal = cast(SemanticRetriever, self._retriever)
        return internal.add_texts(texts, metadatas, document_id)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search the vector store for relevant texts."""
        return [
            document
            for document, _ in self.similarity_search_with_score(
                query, k, filter, **kwargs
            )
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance."""
        try:
            from .semantic_retriever import SemanticRetriever
        except ImportError:
            raise ImportError(_import_err_msg)

        internal = cast(SemanticRetriever, self._retriever)
        return [
            (Document(page_content=text), score)
            for text, score in internal.similarity_search(query, k, filter)
        ]

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete chunnks.

        Note that the "ids" are not corpus ID or document ID. Rather, these
        are the entity names returned by `add_texts`.

        Returns:
            True if successful. Otherwise, you should get an exception anyway.
        """
        try:
            from .semantic_retriever import SemanticRetriever
        except ImportError:
            raise ImportError(_import_err_msg)

        internal = cast(SemanticRetriever, self._retriever)
        return internal.delete(ids)

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self.delete, **kwargs), ids
        )

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        TODO: Check with the team about this!
        The underlying vector store already returns a "score proper",
        i.e. one in [0, 1] where higher means more *similar*.
        """
        return lambda score: score
