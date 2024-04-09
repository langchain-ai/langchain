from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
from uuid import uuid4
from langchain.docstore.document import Document
from langchain_core.vectorstores import VectorStore
from vlite import VLite as Vlite
from vlite.utils import process_file

class VLite(VectorStore):
    """
    VLite is a simple and fast vector database for semantic search.

    Methods:
    - add_texts: Add text data to the vectorstore.
    - add_documents: Add documents to the vectorstore.
    - similarity_search: Perform a similarity search on the vectorstore.
    - similarity_search_with_score: Perform a similarity search with scores.
    - max_marginal_relevance_search: Perform a max marginal relevance search.
    - delete: Delete documents from the vectorstore.
    - update_document: Update a document in the vectorstore.
    - get: Retrieve documents from the vectorstore based on IDs and/or metadata.
    """
    def __init__(
        self,
        collection: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.collection = collection
        self.vlite = Vlite(collection=collection, **kwargs)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if ids is None:
            ids = [str(uuid4()) for _ in texts]
        data = [{"text": text, "metadata": metadata, "id": id}
                for text, metadata, id in zip(texts, metadatas, ids)]
        results = self.vlite.add(data, **kwargs)
        return [result[0] for result in results]

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if ids is None:
            ids = [str(uuid4()) for _ in documents]
        data = []
        for doc, id in zip(documents, ids):
            if doc.path is not None:
                processed_data = process_file(doc.path)
                data.extend([{"text": chunk, "metadata": doc.metadata, "id": f"{id}_{i}"}
                             for i, chunk in enumerate(processed_data)])
            else:
                data.append({"text": doc.page_content, "metadata": doc.metadata, "id": id})
        results = self.vlite.add(data, **kwargs)
        return [result[0] for result in results]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        metadata = filter or {}
        results = self.vlite.retrieve(text=query, top_k=k, metadata=metadata, **kwargs)
        documents = [Document(page_content=text, metadata=metadata)
                     for text, _, metadata in results]
        return documents

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        metadata = filter or {}
        results = self.vlite.retrieve(text=query, top_k=k, metadata=metadata, return_scores=True, **kwargs)
        documents_with_scores = [(Document(page_content=text, metadata=metadata), score)
                                 for text, score, metadata in results]
        return documents_with_scores

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        metadata = filter or {}
        results = self.vlite.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, metadata=metadata, **kwargs
        )
        documents = [Document(page_content=text, metadata=metadata) for text, metadata in results]
        return documents

    def delete(self, ids: List[str], **kwargs: Any):
        self.vlite.delete(ids, **kwargs)

    def update_document(self, document_id: str, document: Document, **kwargs: Any):
        metadata = document.metadata or {}
        self.vlite.update(document_id, text=document.page_content, metadata=metadata, **kwargs)

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        results = self.vlite.get(ids=ids, where=where, **kwargs)
        documents = [Document(page_content=text, metadata=metadata) for text, metadata in results]
        return documents

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.similarity_search(query)

    def get_relevant_documents_with_score(self, query: str) -> List[Tuple[Document, float]]:
        return self.similarity_search_with_score(query)

    def as_retriever(self, **kwargs: Any) -> VLite:
        return self

    @classmethod
    def from_texts(
        cls: Type["VLite"],
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection: Optional[str] = None,
        **kwargs: Any,
    ) -> "VLite":
        vlite = cls(collection=collection, **kwargs)
        vlite.add_texts(texts, metadatas=metadatas, ids=ids, **kwargs)
        return vlite

    @classmethod
    def from_documents(
        cls: Type["VLite"],
        documents: List[Document],
        ids: Optional[List[str]] = None,
        collection: Optional[str] = None,
        **kwargs: Any,
    ) -> "VLite":
        vlite = cls(collection=collection, **kwargs)
        vlite.add_documents(documents, ids=ids, **kwargs)
        return vlite

    @classmethod
    def from_existing_index(
        cls: Type["VLite"],
        collection: str,
        **kwargs: Any,
    ) -> "VLite":
        vlite = cls(collection=collection, **kwargs)
        return vlite