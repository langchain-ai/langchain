from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain.callbacks.manager import CallbackManagerForRetrieverRun


class TFIDFRetriever(BaseRetriever):
    """`TF-IDF` retriever.

    Largely based on
    https://github.com/asvskartheek/Text-Retrieval/blob/master/TF-IDF%20Search%20Engine%20(SKLEARN).ipynb
    """

    vectorizer: Any
    """TF-IDF vectorizer."""
    docs: List[Document]
    """Documents."""
    tfidf_array: Any
    """TF-IDF array."""
    k: int = 4
    """Number of documents to return."""

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

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.vectorizer.transform(
            [query]
        )  # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
        results = cosine_similarity(self.tfidf_array, query_vec).reshape(
            (-1,)
        )  # Op -- (n_docs,1) -- Cosine Sim with each doc
        return_docs = [self.docs[i] for i in results.argsort()[-self.k :][::-1]]
        return return_docs

    def save_local(
        self,
        folder_path: str,
        file_name: str = "tfidf_vectorizer",
    ) -> None:
        try:
            import joblib
        except ImportError:
            raise ImportError(
                "Could not import joblib, please install with `pip install joblib`."
            )

        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)

        # Save vectorizer with joblib dump.
        joblib.dump(self.vectorizer, path / f"{file_name}.joblib")

        # Save docs and tfidf array as pickle.
        with open(path / f"{file_name}.pkl", "wb") as f:
            pickle.dump((self.docs, self.tfidf_array), f)

    @classmethod
    def load_local(
        cls,
        folder_path: str,
        file_name: str = "tfidf_vectorizer",
    ) -> TFIDFRetriever:
        try:
            import joblib
        except ImportError:
            raise ImportError(
                "Could not import joblib, please install with `pip install joblib`."
            )

        path = Path(folder_path)

        # Load vectorizer with joblib load.
        vectorizer = joblib.load(path / f"{file_name}.joblib")

        # Load docs and tfidf array as pickle.
        with open(path / f"{file_name}.pkl", "rb") as f:
            docs, tfidf_array = pickle.load(f)

        return cls(vectorizer=vectorizer, docs=docs, tfidf_array=tfidf_array)
