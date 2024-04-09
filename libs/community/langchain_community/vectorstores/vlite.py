from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

class VLite(VectorStore):
    def __init__(
        self,
        collection: Optional[str] = None,
        embedding_function: Optional[Embeddings] = None,
        model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
        **kwargs: Any,
    ):
        super().__init__()
        self.collection = collection
        self.embedding_function = embedding_function
        self.model_name = model_name
        self.vlite = VLite(collection=collection, model_name=model_name, **kwargs)

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
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas=metadatas, ids=ids, **kwargs)

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

    @classmethod
    def from_texts(
        cls: Type["VLite"],
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection: Optional[str] = None,
        model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
        **kwargs: Any,
    ) -> "VLite":
        vlite = cls(collection=collection, embedding_function=embedding, model_name=model_name, **kwargs)
        vlite.add_texts(texts, metadatas=metadatas, ids=ids, **kwargs)
        return vlite

    @classmethod
    def from_documents(
        cls: Type["VLite"],
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        collection: Optional[str] = None,
        model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
        **kwargs: Any,
    ) -> "VLite":
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts, embedding=embedding, metadatas=metadatas, ids=ids,
            collection=collection, model_name=model_name, **kwargs,
        )