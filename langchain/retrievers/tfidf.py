"""TF-IDF Retriever.

Largely based on
https://github.com/asvskartheek/Text-Retrieval/blob/master/TF-IDF%20Search%20Engine%20(SKLEARN).ipynb"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel

from langchain.schema import BaseRetriever, Document


class TFIDFRetriever(BaseRetriever, BaseModel):
    vectorizer: Any
    docs: List[Document]
    tfidf_array: Any
    k: int = 4

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        tfidf_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TFIDFRetriever:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise ImportError(
                "Could not import scikit-learn, please install with `pip install "
                "scikit-learn`."
            )

        tfidf_params = tfidf_params or {}
        vectorizer = TfidfVectorizer(**tfidf_params)
        tfidf_array = vectorizer.fit_transform(texts)
        metadatas = metadatas or ({} for _ in texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        return cls(vectorizer=vectorizer, docs=docs, tfidf_array=tfidf_array, **kwargs)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        tfidf_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TFIDFRetriever:
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        return cls.from_texts(
            texts=texts, tfidf_params=tfidf_params, metadatas=metadatas, **kwargs
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.vectorizer.transform(
            [query]
        )  # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
        results = cosine_similarity(self.tfidf_array, query_vec).reshape(
            (-1,)
        )  # Op -- (n_docs,1) -- Cosine Sim with each doc
        return_docs = [self.docs[i] for i in results.argsort()[-self.k :][::-1]]
        return return_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError
