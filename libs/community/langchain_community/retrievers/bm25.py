import joblib
import pickle
from pathlib import Path

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field


def default_preprocessing_func(text: str) -> List[str]:
    return text.split()


class BM25Retriever(BaseRetriever):
    """`BM25` retriever without Elasticsearch."""

    vectorizer: Any = None
    """ BM25 vectorizer."""
    docs: List[Document] = Field(repr=False)
    """ List of documents."""
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
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "Could not import rank_bm25, please install with `pip install "
                "rank_bm25`."
            )

        texts_processed = [preprocess_func(t) for t in texts]
        bm25_params = bm25_params or {}
        vectorizer = BM25Okapi(texts_processed, **bm25_params)
        metadatas = metadatas or ({} for _ in texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        return cls(
            vectorizer=vectorizer, docs=docs, preprocess_func=preprocess_func, **kwargs
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
        return_docs = self.vectorizer.get_top_n(processed_query, self.docs, n=self.k)
        return return_docs
    
    def save_local(
        self,
        folder_path: str,
        file_name: str = "bm25_vectorizer",
    ) -> None:
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)

        # Save vectorizer with joblib dump.
        joblib.dump(self.vectorizer, path / f"{file_name}.joblib")

        # Save docs and preprocess_func as pickle.
        with open(path / f"{file_name}.pkl", "wb") as f:
            pickle.dump((self.docs, self.preprocess_func), f)

    @classmethod
    def load_local(
        cls,
        folder_path: str,
        *,
        allow_dangerous_deserialization: bool = False,
        file_name: str = "bm25_vectorizer",
    ) -> BM25Retriever:
        """Load the retriever from local storage with enhancements.

        Args:
            folder_path: Folder path to load from.
            allow_dangerous_deserialization: Whether to allow dangerous deserialization.
            file_name: File name to load from.

        Returns:
            EnhancedBM25Retriever: Loaded retriever with enhancements.
        """
        if not allow_dangerous_deserialization:
            raise ValueError(
                "The de-serialization of this retriever is based on .joblib and "
                ".pkl files."
                "Such files can be modified to deliver a malicious payload that "
                "results in execution of arbitrary code on your machine."
                "You will need to set `allow_dangerous_deserialization` to `True` to "
                "load this retriever. If you do this, make sure you trust the source "
                "of the file, and you are responsible for validating the file "
                "came from a trusted source."
            )

        path = Path(folder_path)

        # Load vectorizer with joblib load.
        vectorizer = joblib.load(path / f"{file_name}.joblib")

        # Load docs and preprocess_func as pickle.
        with open(path / f"{file_name}.pkl", "rb") as f:
            docs, preprocess_func = pickle.load(f)

        return cls(vectorizer=vectorizer, docs=docs, preprocess_func=preprocess_func)
