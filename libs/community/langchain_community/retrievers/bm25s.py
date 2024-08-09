from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

# from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import guard_import


class BM25SRetriever(BaseRetriever):
    """`BM25S` retriever."""

    retriever: Any
    """ BM25 retriever."""
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
        persist_directory: Optional[str] = None,
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
            persist_directory: Directory to persist the retriever.
            bm25s_params: Parameters to pass to the BM25S vectorizer.
            stopwords: Stopwords to use. Defaults to 'en'.
            stemmer_language: Stemmer to use. Defaults to 'english'.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25SRetriever instance.
        """
        bm25s = guard_import("bm25s")
        Stemmer = guard_import("Stemmer", pip_name="PyStemmer")

        # Initialize the BM25S vectorizer
        bm25s_params = bm25s_params or {}
        metadatas = metadatas or ({} for _ in texts)

        if stemmer_language is not None:
            # Initialize the stemmer
            stemmer = Stemmer.Stemmer(stemmer_language)
        else:
            stemmer = None
        corpus_json = [{"text": t, "metadata": m} for t, m in zip(texts, metadatas)]

        # Tokenize the corpus and only keep the ids (faster and saves memory)
        corpus_tokenized = bm25s.tokenize(texts, stopwords=stopwords, stemmer=stemmer)
        retriever = bm25s.BM25(corpus=corpus_json, **bm25s_params)
        retriever.index(corpus_tokenized)
        if persist_directory:
            retriever.save(persist_directory, corpus=corpus_json)
        return cls(
            retriever=retriever,
            stopwords=stopwords,
            stemmer=stemmer,
            **kwargs,
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        persist_directory: Optional[str] = None,
        bm25s_params: Optional[Dict[str, Any]] = None,
        stopwords: Optional[str | list[str]] = "en",
        stemmer_language: Optional[str] = "english",
        **kwargs: Any,
    ) -> BM25SRetriever:
        """
        Create a BM25SRetriever from a list of Documents.
        Args:
            documents: A list of Documents to vectorize.
            persist_directory: Directory to persist the retriever.
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
            metadatas=metadatas,
            persist_directory=persist_directory,
            bm25s_params=bm25s_params,
            stopwords=stopwords,
            stemmer_language=stemmer_language,
            **kwargs,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        bm25s = guard_import("bm25s")

        queries_tokenized = bm25s.tokenize(
            query, stopwords=self.stopwords, stemmer=self.stemmer
        )
        result = self.retriever.retrieve(
            queries_tokenized,
            k=self.k,
            n_threads=self.n_threads,
            chunksize=self.chunksize,
        )
        return_docs = [
            Document(page_content=doc.get("text"), metadata=doc.get("metadata"))
            for doc in result.documents[0]
        ]
        return return_docs

    @classmethod
    def load(
        cls,
        persist_directory: str,
        stopwords: Optional[str | list[str]] = "en",
        stemmer_language: Optional[str] = "english",
        mmap: Optional[bool] = False,
        **kwargs: Any,
    ) -> BM25SRetriever:
        """Load the retriever from local storage.

        Args:
            persist_directory: Folder path to load from.

        Returns:
            BM25SRetriever: Loaded retriever.
        """
        bm25s = guard_import("bm25s")
        Stemmer = guard_import("Stemmer", pip_name="PyStemmer")

        if stemmer_language is not None:
            # Initialize the stemmer
            stemmer = Stemmer.Stemmer(stemmer_language)
        else:
            stemmer = None

        retriever = bm25s.BM25.load(persist_directory, load_corpus=True, mmap=mmap)
        return cls(
            retriever=retriever,
            stopwords=stopwords,
            stemmer=stemmer,
            **kwargs,
        )
