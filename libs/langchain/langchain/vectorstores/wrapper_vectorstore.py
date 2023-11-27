from typing import Any, Iterable, List, Optional, Tuple, Type, TypeVar

# Note: Import directly from langchain_core is not stable and generate some errors
# from langchain_core.documents import Document
# from langchain_core.embeddings import Embeddings
# from langchain_core.vectorstores import VectorStore
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore

VST = TypeVar("VST", bound="WrapperVectorStore")
T = TypeVar("T")


class WrapperVectorStore(VectorStore):
    """
    A generic wrapper for vectorstore.
    Can be known by SelfQueryRetriever to generate the target request.
    """

    def __init__(self, *, vectorstore: VectorStore):
        self.vectorstore = vectorstore

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.vectorstore.embeddings

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        return self.vectorstore.add_texts(texts=texts, metadatas=metadatas, **kwargs)

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        return await self.vectorstore.aadd_texts(
            texts=texts, metadatas=metadatas, **kwargs
        )

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        return self.delete(ids=ids, **kwargs)

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        return await self.vectorstore.adelete(ids=ids, **kwargs)

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        return self.vectorstore.add_documents(documents=documents, **kwargs)

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        return await self.vectorstore.aadd_documents(documents=documents, **kwargs)

    def search(self, query: str, search_type: str, **kwargs: Any) -> List[Document]:
        return self.vectorstore.search(query=query, search_type=search_type, **kwargs)

    async def asearch(
        self, query: str, search_type: str, **kwargs: Any
    ) -> List[Document]:
        return await self.vectorstore.asearch(
            query=query, search_type=search_type, **kwargs
        )

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return self.vectorstore.similarity_search(query=query, k=k, **kwargs)

    def similarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        return self.vectorstore.similarity_search_with_score(*args, **kwargs)

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        return self.vectorstore.similarity_search_with_relevance_scores(
            query=query, k=k, **kwargs
        )

    async def asimilarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        return await self.vectorstore.asimilarity_search_with_relevance_scores(
            query=query, k=k, **kwargs
        )

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return await self.vectorstore.asimilarity_search(query=query, k=k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return self.vectorstore.similarity_search_by_vector(
            embedding=embedding, k=k, **kwargs
        )

    async def asimilarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return await self.vectorstore.asimilarity_search_by_vector(
            embedding=embedding, k=k, **kwargs
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        return self.vectorstore.max_marginal_relevance_search(
            query=query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        return await self.vectorstore.amax_marginal_relevance_search(
            query=query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        return self.vectorstore.max_marginal_relevance_search_by_vector(
            embedding=embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        return await self.vectorstore.amax_marginal_relevance_search_by_vector(
            embedding=embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VST:
        vectorstore_cls = kwargs.pop("vectorstore_cls")
        return cls(
            vectorstore=vectorstore_cls.from_texts(
                texts=texts, embedding=embedding, metadatas=metadatas, **kwargs
            )
        )


if __name__.startswith("langchain_"):

    def _hack() -> None:
        """
        This is a hack, to dynamically path the _get_builtin_translator()
        if we use the component before accepting the pull-request
        """
        from langchain.retrievers.self_query import base as to_patch

        old_get_builtin_translator = to_patch._get_builtin_translator

        def patch_get_builtin_translator(vectorstore: VectorStore) -> Any:
            if isinstance(vectorstore, WrapperVectorStore):
                return patch_get_builtin_translator(vectorstore.vectorstore)
            return old_get_builtin_translator(vectorstore)

        to_patch._get_builtin_translator = patch_get_builtin_translator

    _hack()
