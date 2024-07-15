from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever


class BM25SRetriever(BaseRetriever):
    """`BM25S` retriever."""

    retriever: Any
    """ BM25 retriever."""
    docs: List[Document] = Field(repr=False)
    """ List of documents."""
    k: int = 4
    """ Number of documents to return."""
    stopwords: str = "en"
    """ Stopwords to use. Defaults to 'en'."""
    stemmer: Optional[Any] = None
    """ Stemmer to use. Defaults to None."""
    n_threads: int = 0
    """ Number of threads to use. Defaults to 0."""
    chunksize: int = 50
    """ Chunksize to use. Defaults to 50."""
    

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        bm25s_params: Optional[Dict[str, Any]] = None,
        stopwords: Optional[str | list[str]] = "en",
        stemmer_language: Optional[str] = "english",
        **kwargs: Any,
    ) -> BM25SRetriever:
        """
        Create a BM25SRetriever from a list of texts.
        Args:
            texts: A list of texts to vectorize.
            metadatas: A list of metadata dicts to associate with each text.
            bm25s_params: Parameters to pass to the BM25S vectorizer.
            stopwords: Stopwords to use. Defaults to 'en'.
            stemmer_language: Stemmer to use. Defaults to 'english'.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25SRetriever instance.
        """
        try:
            import bm25s
        except ImportError:
            raise ImportError(
                "Could not import bm25s, please install with `pip install "
                "bm25s`."
            )
        try:
            import Stemmer
        except ImportError:
            raise ImportError(
                "Could not import Stemmer, please install with `pip install "
                "PyStemmer`."
            )
        
        # Initialize the BM25S vectorizer
        bm25s_params = bm25s_params or {}
        metadatas = metadatas or ({} for _ in texts)
        
        if stemmer_language is not None:
            # Initialize the stemmer
            stemmer = Stemmer.Stemmer(stemmer_language)
        else:
            stemmer = None
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]

        # Tokenize the corpus and only keep the ids (faster and saves memory)
        corpus_tokenized = bm25s.tokenize(texts, stopwords=stopwords, stemmer=stemmer)
        retriever = bm25s.BM25(**bm25s_params)
        retriever.index(corpus_tokenized)
        return cls(
            retriever=retriever, docs=docs, stopwords=stopwords, stemmer=stemmer, **kwargs
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25s_params: Optional[Dict[str, Any]] = None,
        stopwords: Optional[str | list[str]] = "en",
        stemmer_language: Optional[str] = "english",
        **kwargs: Any,
    ) -> BM25SRetriever:
        """
        Create a BM25SRetriever from a list of Documents.
        Args:
            documents: A list of Documents to vectorize.
            bm25s_params: Parameters to pass to the BM25S vectorizer.
            stopwords: Stopwords to use. Defaults to 'en'.
            stemmer_language: Stemmer to use. Defaults to 'english'.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25SRetriever instance.
        """
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        return cls.from_texts(
            texts=texts,
            bm25s_params=bm25s_params,
            metadatas=metadatas,
            stopwords=stopwords,
            stemmer_language=stemmer_language,
            **kwargs,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        try:
            import bm25s
        except ImportError:
            raise ImportError(
                "Could not import bm25s, please install with `pip install "
                "bm25s`."
            )
        queries_tokenized = bm25s.tokenize(query, stopwords=self.stopwords, stemmer=self.stemmer)
        return_docs = self.retriever.retrieve(queries_tokenized, k=self.k, n_threads=self.n_threads, chunksize=self.chunksize)
        return_docs = [self.docs[idx] for idx in return_docs.documents[0]]
        return return_docs
