"""TF-IDF Retriever.

Largely based on
https://github.com/asvskartheek/Text-Retrieval/blob/master/TF-IDF%20Search%20Engine%20(SKLEARN).ipynb"""
from typing import Any, List

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
    def from_texts(cls, texts: List[str], **kwargs: Any) -> "TFIDFRetriever":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer()
        tfidf_array = vectorizer.fit_transform(texts)
        docs = [Document(page_content=t) for t in texts]
        return cls(vectorizer=vectorizer, docs=docs, tfidf_array=tfidf_array, **kwargs)

    def get_relevant_documents(self, query: str) -> List[Document]:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.vectorizer.transform(
            [query]
        )  # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
        results = cosine_similarity(self.tfidf_array, query_vec).reshape(
            (-1,)
        )  # Op -- (n_docs,1) -- Cosine Sim with each doc
        return_docs = []
        for i in results.argsort()[-self.k :][::-1]:
            return_docs.append(self.docs[i])
        return return_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError
