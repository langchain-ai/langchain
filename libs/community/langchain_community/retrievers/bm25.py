from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field


def default_preprocessing_func(text: str) -> List[str]:
    return text.split()

def create_bm25_vectorizer(corpus, **bm25_params):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError(
            "Could not import rank_bm25, please install with `pip install "
            "rank_bm25`."
        )
    
    class BM25Vectorizer(BM25Okapi):
        def __init__(self, corpus, **bm25_params):            
            super().__init__(corpus, **bm25_params)
            self.vocabulary = list(self.idf.keys())
            self.word_to_id = {word: i for i, word in enumerate(self.vocabulary)}
        
        def transform(self, queries: list[list[str]]) -> scipy.sparse.csr_matrix:
            try:
                from scipy.sparse import csr_matrix
            except ImportError:
                raise ImportError(
                    "Could not import scipy, please install with `pip install "
                    "scipy`."
                )
            
            rows = []
            cols = []
            data = []
            
            for i, query in enumerate(queries):
                query_len = len(query)
                
                for word in set(query):
                    if word in self.word_to_id:
                        word_id = self.word_to_id[word]
                        tf = query.count(word)
                        idf = self.idf.get(word, 0)
                        
                        # BM25 scoring formula
                        numerator = idf * tf * (self.k1 + 1)
                        denominator = tf + self.k1 * (1 - self.b + self.b * query_len / self.avgdl)
                        
                        score = numerator / denominator
                        
                        rows.append(i)
                        cols.append(word_id)
                        data.append(score)
            
            return csr_matrix((data, (rows, cols)), shape=(len(queries), len(self.vocabulary)))
        
        def count_transform(self, queries: list[list[str]]) -> scipy.sparse.csr_matrix:
            try:
                from scipy.sparse import csr_matrix
            except ImportError:
                raise ImportError(
                    "Could not import scipy, please install with `pip install "
                    "scipy`."
                )
            
            rows = []
            cols = []
            data = []
            
            for i, query in enumerate(queries):
                for word in query:
                    if word in self.word_to_id:
                        word_id = self.word_to_id[word]
                        rows.append(i)
                        cols.append(word_id)
                        data.append(1)  # Count is always 1 for each occurrence
            
            return csr_matrix((data, (rows, cols)), shape=(len(queries), len(self.vocabulary)))
    
    return BM25Vectorizer(corpus, **bm25_params)

class BM25Retriever(BaseRetriever):
    """`BM25` retriever without Elasticsearch."""

    vectorizer: Any = None
    """ BM25 vectorizer."""
    docs: List[Document] = Field(repr=False)
    """ List of documents."""
    bm25_array: Any = None
    """BM25 array."""
    k: int = 4
    """ Number of documents to return."""
    preprocess_func: Callable[[str], List[str]] = default_preprocessing_func
    """ Preprocessing function to use on the text before BM25 vectorization."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> BM25Retriever:
        """
        Create a BM25Retriever from a list of texts.
        Args:
            texts: A list of texts to vectorize.
            metadatas: A list of metadata dicts to associate with each text.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25Retriever instance.
        """
        texts_processed = [preprocess_func(t) for t in texts]
        bm25_params = bm25_params or {}
        vectorizer = create_bm25_vectorizer(texts_processed, **bm25_params)
        bm25_array = vectorizer.transform(texts_processed)
        metadatas = metadatas or ({} for _ in texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        return cls(
            vectorizer=vectorizer, docs=docs, bm25_array=bm25_array, preprocess_func=preprocess_func, **kwargs
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> BM25Retriever:
        """
        Create a BM25Retriever from a list of Documents.
        Args:
            documents: A list of Documents to vectorize.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25Retriever instance.
        """
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        return cls.from_texts(
            texts=texts,
            bm25_params=bm25_params,
            metadatas=metadatas,
            preprocess_func=preprocess_func,
            **kwargs,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        processed_query = self.preprocess_func(query)
        query_vec = self.vectorizer.count_transform([processed_query])
        results = query_vec.dot(self.bm25_array.T).toarray()[0]
        return_docs = [self.docs[i] for i in results.argsort()[-self.k :][::-1]]
        return return_docs