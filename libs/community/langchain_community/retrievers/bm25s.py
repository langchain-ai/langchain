from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

DEFAULT_PERSISTENCE_DIRECTORY = "./bm25s_index"
CORPUS_PERSISTENCE_FILE = "corpus.jsonl"


class BM25SRetriever(BaseRetriever):
    """`BM25` retriever with `bm25s` backend"""

    vectorizer: Any
    """ BM25S vectorizer."""
    docs: List[Document] = Field(repr=False)
    """List of documents to retrieve from."""
    k: int = 4
    """Number of top results to return"""
    activate_numba: bool = False
    """Accelerate backend"""

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        stopwords: Union[str, List[str]] = "en",
        stemmer: Optional[Callable[[List[str]], List[str]]] = None,
        persist_directory: Optional[str] = None,
        **kwargs: Any,
    ) -> BM25SRetriever:
        """
        Create a BM25Retriever from a list of texts.
        Args:
            texts:
                A list of texts to vectorize.
            metadatas:
                A list of metadata dicts to associate with each text.
            bm25_params:
                Parameters to pass to the BM25s vectorizer.
            stopwords:
                The list of stopwords to remove from the text. Defaults to "en".
            stemmer:
                The stemmer to use for stemming the tokens. It is recommended to
                use the PyStemmer library for stemming, but you can also any
                callable that takes a list of strings and returns a list of strings.
            persist_directory:
                The directory to save the BM25 index to.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25SRetriever instance.
        """
        try:
            from bm25s import BM25
            from bm25s import tokenize as bm25s_tokenize
        except ImportError:
            raise ImportError(
                "Could not import bm25s, please install with `pip install " "bm25s`."
            )

        bm25_params = bm25_params or {}
        texts_processed = bm25s_tokenize(
            texts=texts,
            stopwords=stopwords,
            stemmer=stemmer,
            return_ids=False,
            show_progress=False,
        )
        vectorizer = BM25(**bm25_params)
        vectorizer.index(texts_processed)

        metadatas = metadatas or ({} for _ in texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]

        persist_directory = persist_directory or DEFAULT_PERSISTENCE_DIRECTORY
        # persist the vectorizer
        vectorizer.save(persist_directory)
        # additionally persist the corpus and the metadata
        with open(f"{persist_directory}/{CORPUS_PERSISTENCE_FILE}", "w") as f:
            for i, d in enumerate(docs):
                entry = {"id": i, "text": d.page_content, "metadata": d.metadata}
                doc_str = json.dumps(entry)
                f.write(doc_str + "\n")

        return cls(vectorizer=vectorizer, docs=docs, **kwargs)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25_params: Optional[Dict[str, Any]] = None,
        stopwords: Union[str, List[str]] = "en",
        stemmer: Optional[Callable[[List[str]], List[str]]] = None,
        persist_directory: Optional[str] = None,
        **kwargs: Any,
    ) -> BM25SRetriever:
        """
        Create a BM25Retriever from a list of Documents.
        Args:
            documents:
                A list of Documents to vectorize.
            bm25_params:
                Parameters to pass to the BM25 vectorizer.
            stopwords:
                The list of stopwords to remove from the text. Defaults to "en".
            stemmer:
                The stemmer to use for stemming the tokens. It is recommended to
                use the PyStemmer library for stemming, but you can also any
                callable that takes a list of strings and returns a list of strings.
            persist_directory:
                The directory to save the BM25 index to.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25Retriever instance.
        """
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        return cls.from_texts(
            texts=texts,
            metadatas=metadatas,
            bm25_params=bm25_params,
            stopwords=stopwords,
            stemmer=stemmer,
            persist_directory=persist_directory,
            **kwargs,
        )

    @classmethod
    def from_persisted_directory(cls, path: str, **kwargs: Any) -> BM25SRetriever:
        from bm25s import BM25

        vectorizer = BM25.load(path)
        with open(f"{path}/{CORPUS_PERSISTENCE_FILE}", "r") as f:
            corpus = [json.loads(line) for line in f]

        docs = [
            Document(page_content=d["text"], metadata=d["metadata"]) for d in corpus
        ]
        return cls(vectorizer=vectorizer, docs=docs, **kwargs)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        from bm25s import tokenize as bm25s_tokenize

        processed_query = bm25s_tokenize(query, return_ids=False)
        if self.activate_numba:
            self.vectorizer.activate_numba_scorer()
            return_docs = self.vectorizer.retrieve(
                processed_query, k=self.k, backend_selection="numba"
            )
            return [self.docs[i] for i in return_docs.documents[0]]
        else:
            return_docs, scores = self.vectorizer.retrieve(
                processed_query, self.docs, k=self.k
            )
            return [return_docs[0, i] for i in range(return_docs.shape[1])]
