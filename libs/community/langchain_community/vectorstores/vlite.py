from __future__ import annotations

# Standard library imports
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

# LangChain imports
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class VLite(VectorStore):
    """VLite is a simple and fast vector database for semantic search."""

    def __init__(
        self,
        embedding_function: Embeddings,
        collection: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.embedding_function = embedding_function
        self.collection = collection or f"vlite_{uuid4().hex}"
        # Third-party imports
        try:
            from vlite import VLite
        except ImportError:
            raise ImportError(
                "Could not import vlite python package. "
                "Please install it with `pip install vlite`."
            )
        self.vlite = VLite(collection=self.collection, **kwargs)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        texts = list(texts)
        ids = kwargs.pop("ids", [str(uuid4()) for _ in texts])
        embeddings = self.embedding_function.embed_documents(texts)
        if not metadatas:
            metadatas = [{} for _ in texts]
        data_points = [
            {"text": text, "metadata": metadata, "id": id, "embedding": embedding}
            for text, metadata, id, embedding in zip(texts, metadatas, ids, embeddings)
        ]
        results = self.vlite.add(data_points)
        return [result[0] for result in results]

    def add_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """Add a list of documents to the vectorstore.

        Args:
            documents: List of documents to add to the vectorstore.
            kwargs: vectorstore specific parameters such as "file_path" for processing
                    directly with vlite.

        Returns:
            List of ids from adding the documents into the vectorstore.
        """
        ids = kwargs.pop("ids", [str(uuid4()) for _ in documents])
        texts = []
        metadatas = []
        for doc, id in zip(documents, ids):
            if "file_path" in kwargs:
                # Third-party imports
                try:
                    from vlite.utils import process_file
                except ImportError:
                    raise ImportError(
                        "Could not import vlite python package. "
                        "Please install it with `pip install vlite`."
                    )
                processed_data = process_file(kwargs["file_path"])
                texts.extend(processed_data)
                metadatas.extend([doc.metadata] * len(processed_data))
                ids.extend([f"{id}_{i}" for i in range(len(processed_data))])
            else:
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)
        return self.add_texts(texts, metadatas, ids=ids)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(query, k=k)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.

        Returns:
            List of Tuples of (doc, score), where score is the similarity score.
        """
        metadata = filter or {}
        embedding = self.embedding_function.embed_query(query)
        results = self.vlite.retrieve(
            text=query,
            top_k=k,
            metadata=metadata,
            return_scores=True,
            embedding=embedding,
        )
        documents_with_scores = [
            (Document(page_content=text, metadata=metadata), score)
            for text, score, metadata in results
        ]
        return documents_with_scores

    def update_document(self, document_id: str, document: Document) -> None:
        """Update an existing document in the vectorstore."""
        self.vlite.update(
            document_id, text=document.page_content, metadata=document.metadata
        )

    def get(self, ids: List[str]) -> List[Document]:
        """Get documents by their IDs."""
        results = self.vlite.get(ids)
        documents = [
            Document(page_content=text, metadata=metadata) for text, metadata in results
        ]
        return documents

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by ids."""
        if ids is not None:
            self.vlite.delete(ids, **kwargs)
            return True
        return None

    @classmethod
    def from_existing_index(
        cls,
        embedding: Embeddings,
        collection: str,
        **kwargs: Any,
    ) -> VLite:
        """Load an existing VLite index.

        Args:
            embedding: Embedding function
            collection: Name of the collection to load.

        Returns:
            VLite vector store.
        """
        vlite = cls(embedding_function=embedding, collection=collection, **kwargs)
        return vlite

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        collection: Optional[str] = None,
        **kwargs: Any,
    ) -> VLite:
        """Construct VLite wrapper from raw documents.

        This is a user-friendly interface that:
        1. Embeds documents.
        2. Adds the documents to the vectorstore.

        This is intended to be a quick way to get started.

        Example:
        .. code-block:: python

            from langchain import VLite
            from langchain.embeddings import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings()
            vlite = VLite.from_texts(texts, embeddings)
        """
        vlite = cls(embedding_function=embedding, collection=collection, **kwargs)
        vlite.add_texts(texts, metadatas, **kwargs)
        return vlite

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        collection: Optional[str] = None,
        **kwargs: Any,
    ) -> VLite:
        """Construct VLite wrapper from a list of documents.

        This is a user-friendly interface that:
        1. Embeds documents.
        2. Adds the documents to the vectorstore.

        This is intended to be a quick way to get started.

        Example:
        .. code-block:: python

            from langchain import VLite
            from langchain.embeddings import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings()
            vlite = VLite.from_documents(documents, embeddings)
        """
        vlite = cls(embedding_function=embedding, collection=collection, **kwargs)
        vlite.add_documents(documents, **kwargs)
        return vlite
