"""BM25 Retriever.

Based on
https://github.com/asvskartheek/Text-Retrieval/blob/master/TF-IDF%20Search%20Engine%20(SKLEARN).ipynb
"""
from __future__ import annotations

from functools import partial
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel

from langchain.schema import BaseRetriever, Document


class BM25Retriever(BaseRetriever, BaseModel):
    dictionary: Any
    bm25_index: Any
    query_model: Any
    docs: List[Document]
    preprocessor: Any
    k: int = 4

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_filters: Optional[Iterable()] = None,
        **kwargs: Any,
    ) -> BM25Retriever:
        try:
            from gensim.corpora import Dictionary
            from gensim.models import TfidfModel, OkapiBM25Model
            from gensim.parsing.preprocessing import (preprocess_string, 
                                                      strip_punctuation,
                                                      remove_stopwords)
            from gensim.similarities import SparseMatrixSimilarity
        except ImportError:
            raise ImportError(
                "Could not import gensim, please install with "
                "`pip install gensim`."
            )

        # Default tokenizer
        default_filters = [lambda x: x.lower(), strip_punctuation]
        filters = preprocess_filters or default_filters
        preprocessor = partial(preprocess_string, filters=filters)
        gensim_texts = [preprocessor(text) for text in texts]    
        dictionary = Dictionary(gensim_texts)
        bow_corpus = [dictionary.doc2bow(text) for text in gensim_texts]
        bm25_params = bm25_params or {}
        document_model = OkapiBM25Model(dictionary=dictionary, **bm25_params)
        document_corpus = document_model[bow_corpus]
        bm25_index = SparseMatrixSimilarity(
            document_corpus, 
            num_docs=len(texts), 
            num_terms=len(dictionary),
            normalize_queries=False, 
            normalize_documents=False
        )
        query_model = TfidfModel(dictionary=dictionary, smartirs='bnn')  # enforce binary weights

        metadatas = metadatas or ({} for _ in texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        return cls(dictionary=dictionary, bm25_index=bm25_index, query_model=query_model, 
                   docs=docs, preprocessor=preprocessor, **kwargs)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BM25Retriever:
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        return cls.from_texts(
            texts=texts, metadatas=metadatas, bm25_params=bm25_params, **kwargs
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        gensim_query = self.tokenizer(query)    
        bow_query = self.dictionary.doc2bow(gensim_query)
        bm25_query = self.query_model[bow_query]
        similarities = self.bm25_index[bm25_query]
        relevent_docs = [self.docs[i] for i in similarities.argsort()[-self.k :][::-1]]

        return relevent_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError
