from typing import TYPE_CHECKING, TypeVar, List, Optional, Type, Iterable, Any, Tuple

from docarray import DocList, BaseDoc
from operator import itemgetter

from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores import VectorStore

from docarray.index.abstract import BaseDocIndex


T_Doc = TypeVar('T_Doc', bound=BaseDocIndex)


def _check_docarray_import():
    try:
        import docarray
        da_version = docarray.__version__.split('.')
        if int(da_version[0]) == 0 and int(da_version[1]) <= 21:
            raise ValueError(
                f'To use the HnswLib VectorStore the docarray version >=0.31.0 is expected, '
                f'received: {docarray.__version__}.'
                f'To upgrade, please run: `pip install -U docarray`.'
            )
    except ImportError:
        raise ImportError(
            "Could not import docarray python package. "
            "Please install it with `pip install -U docarray`."
        )


class VecStoreFromDocIndex(VectorStore):
    doc_index: BaseDocIndex = None
    doc_cls: Type[BaseDoc] = None
    embedding: Embeddings = None

    def __init__(
        self,
        doc_index: T_Doc,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]],
    ):
        self.doc_index = doc_index
        self.doc_cls = doc_index._schema
        self.embedding = embedding

        embeddings = self.embedding.embed_documents(texts)
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]

        docs = DocList[self.doc_cls](
            [
                self.doc_cls(
                    text=t,
                    embedding=e,
                    metadata=m,
                ) for t, m, e in zip(texts, metadatas, embeddings)
            ]
        )
        if len(docs) > 0:
            self.doc_index.index(docs)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if metadatas is None:
            metadatas = [{} for _ in range(len(list(texts)))]

        ids = []
        embeddings = self.embedding.embed_documents(texts)
        for t, m, e in zip(texts, metadatas, embeddings):
            doc = self.doc_cls(
                text=t,
                embedding=e,
                metadata=m
            )
            self.doc_index.index([doc])
            ids.append(doc.id)  # TODO return index of self.docs ?

        return ids

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each.
        """
        query_embedding = self.embedding.embed_query(query)
        query_doc = self.doc_cls(embedding=query_embedding)
        docs, scores = self.doc_index.find(query_doc, search_field='embedding', limit=k)

        result = [(Document(page_content=doc.text), score) for doc, score in zip(docs, scores)]
        return result

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        results = self.similarity_search_with_score(query, k)
        return list(map(itemgetter(0), results))


    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and relevance scores, normalized on a scale from 0 to 1.

        0 is dissimilar, 1 is most similar.
        """
        raise NotImplementedError

    def similarity_search_by_vector(self, embedding: List[float], k: int = 4, **kwargs: Any) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query vector.
        """

        query_doc = self.doc_cls(embedding=embedding)
        docs = self.doc_index.find(query_doc, search_field='embedding', limit=k).documents

        result = [Document(page_content=doc.text) for doc in docs]
        return result

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
        query_embedding = self.embedding.embed_query(query)
        query_doc = self.doc_cls(embedding=query_embedding)

        docs, scores = self.doc_index.find(query_doc, search_field='embedding', limit=fetch_k)

        embeddings = [emb for emb in docs.emb]

        mmr_selected = maximal_marginal_relevance(query_embedding, embeddings, k=k)
        results = [Document(page_content=self.doc_index[idx].text) for idx in mmr_selected]
        return results

