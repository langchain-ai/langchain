"""Wrapper around in-memory DocArray store."""
from __future__ import annotations

from typing import List, Optional, Any, Type

from docarray.typing import NdArray

from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores.base import VST
from langchain.vectorstores.utils import maximal_marginal_relevance
from langchain.vectorstores.vector_store_from_doc_index import _check_docarray_import, VecStoreFromDocIndex


class InMemory(VecStoreFromDocIndex):
    """Wrapper around in-memory storage.

    To use it, you should have the ``docarray`` package with version >=0.31.0 installed.
    """
    def __init__(
        self,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        metric: str = 'cosine_sim',
    ) -> None:
        """Initialize in-memory store.

        Args:
            texts (List[str]): Text data.
            embedding (Embeddings): Embedding function.
            metadatas (Optional[List[dict]]): Metadata for each text if it exists.
                Defaults to None.
            metric (str): metric for exact nearest-neighbor search.
                Can be one of: 'cosine_sim', 'euclidean_dist' and 'sqeuclidean_dist'.
                Defaults to 'cosine_sim'.

        """
        _check_docarray_import()
        from docarray.index import InMemoryDocIndex

        doc_cls = self._get_doc_cls(metric)
        doc_index = InMemoryDocIndex[doc_cls]()
        super().__init__(doc_index, texts, embedding, metadatas)

    @staticmethod
    def _get_doc_cls(sim_metric: str):
        from docarray import BaseDoc
        from pydantic import Field

        class DocArrayDoc(BaseDoc):
            text: Optional[str]
            embedding: Optional[NdArray] = Field(space=sim_metric)
            metadata: Optional[dict]

        return DocArrayDoc

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        metric: str = 'cosine_sim',
        **kwargs: Any
    ) -> InMemory:
        return cls(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            metric=metric,
        )
    #
    # def add_texts(
    #     self,
    #     texts: Iterable[str],
    #     metadatas: Optional[List[dict]] = None,
    #     **kwargs: Any
    # ) -> List[str]:
    #     """Run more texts through the embeddings and add to the vectorstore.
    #
    #     Args:
    #         texts: Iterable of strings to add to the vectorstore.
    #         metadatas: Optional list of metadatas associated with the texts.
    #
    #     Returns:
    #         List of ids from adding the texts into the vectorstore.
    #     """
    #     if metadatas is None:
    #         metadatas = [{} for _ in range(len(list(texts)))]
    #
    #     ids = []
    #     embeddings = self.embedding.embed_documents(texts)
    #     for t, m, e in zip(texts, metadatas, embeddings):
    #         doc = self.doc_cls(
    #             text=t,
    #             embedding=e,
    #             metadata=m
    #         )
    #         self.docs.append(doc)
    #         ids.append(doc.id)  # TODO return index of self.docs ?
    #
    #     return ids
    #
    # def similarity_search_with_score(
    #     self, query: str, k: int = 4, **kwargs: Any
    # ) -> List[Tuple[Document, float]]:
    #     """Return docs most similar to query.
    #
    #     Args:
    #         query: Text to look up documents similar to.
    #         k: Number of Documents to return. Defaults to 4.
    #
    #     Returns:
    #         List of Documents most similar to the query and score for each.
    #     """
    #     from docarray.utils.find import find  # TODO move import
    #
    #     query_embedding = self.embedding.embed_query(query)
    #     query_doc = self.doc_cls(embedding=query_embedding)
    #     docs, scores = find(index=self.docs, query=query_doc, limit=k, search_field='embedding')
    #
    #     result = [(Document(page_content=doc.text), score) for doc, score in zip(docs, scores)]
    #     return result
    #
    # def similarity_search(
    #     self, query: str, k: int = 4, **kwargs: Any
    # ) -> List[Document]:
    #     """Return docs most similar to query.
    #
    #     Args:
    #         query: Text to look up documents similar to.
    #         k: Number of Documents to return. Defaults to 4.
    #
    #     Returns:
    #         List of Documents most similar to the query.
    #     """
    #     results = self.similarity_search_with_score(query, k)
    #     return list(map(itemgetter(0), results))
    #
    # def _similarity_search_with_relevance_scores(
    #     self,
    #     query: str,
    #     k: int = 4,
    #     **kwargs: Any,
    # ) -> List[Tuple[Document, float]]:
    #     """Return docs and relevance scores, normalized on a scale from 0 to 1.
    #
    #     0 is dissimilar, 1 is most similar.
    #     """
    #     raise NotImplementedError
    #
    # def similarity_search_by_vector(self, embedding: List[float], k: int = 4, **kwargs: Any) -> List[Document]:
    #     """Return docs most similar to embedding vector.
    #
    #     Args:
    #         embedding: Embedding to look up documents similar to.
    #         k: Number of Documents to return. Defaults to 4.
    #
    #     Returns:
    #         List of Documents most similar to the query vector.
    #     """
    #     from docarray.utils.find import find
    #
    #     query_doc = self.doc_cls(embedding=embedding)
    #     result_docs = find(index=self.docs, query=query_doc, limit=k, search_field='embedding').documents
    #
    #     result = [Document(page_content=doc.text) for doc in result_docs]
    #     return result

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, **kwargs: Any
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.

        Returns:
            List of Documents selected by maximal marginal relevance.
                """
        from docarray.utils.find import find

        query_embedding = self.embedding.embed_query(query)
        query_doc = self.doc_cls(embedding=query_embedding)
        find_res = find(self.docs, query_doc, limit=k)

        embeddings = [emb for emb in find_res.documents.emb]
        mmr_selected = maximal_marginal_relevance(query_embedding, embeddings, k=k)
        results = []
        for idx in mmr_selected:
            results.append(Document(page_content=self.docs[idx].text))
        return results

