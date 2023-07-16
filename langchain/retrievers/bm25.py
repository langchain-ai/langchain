"""
BM25 Retriever without elastic search
"""


from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Callable

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)

from langchain.schema import BaseRetriever, Document


class BM25Retriever(BaseRetriever):
    vectorizer: Any
    docs: List[Document]
    k: int = 4
    preprocess_func: Callable[[str], str] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Optional[Callable[[str], str]] = None,
        **kwargs: Any,
    ) -> BM25Retriever:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "Could not import rank_bm25, please install with `pip install "
                "rank_bm25`."
            )
            
        if preprocess_func:
            texts_processed = [preprocess_func(t) for t in texts]
        else:
            texts_processed = [t.split() for t in texts]
            
        
        bm25_params = bm25_params or {}
        vectorizer = BM25Okapi(texts_processed, **bm25_params)
        metadatas = metadatas or ({} for _ in texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        return cls(vectorizer=vectorizer, docs=docs, preprocess_func=preprocess_func, **kwargs)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Optional[Callable[[str], str]] = None,
        **kwargs: Any,
    ) -> BM25Retriever:
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        return cls.from_texts(
            texts=texts, bm25_params=bm25_params, metadatas=metadatas,preprocess_func=preprocess_func,  **kwargs
        )

    def _get_relevant_documents(self, query: str, * ,run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        if self.preprocess_func:
            processed_query = self.preprocess_func(query)
        else:
            processed_query = query.split()
            
        return_docs = self.vectorizer.get_top_n(processed_query,self.docs, n=self.k)
        return return_docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        raise NotImplementedError
