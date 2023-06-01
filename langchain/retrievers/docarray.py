from typing import List

import numpy as np

from langchain.schema import BaseRetriever, Document
from langchain.vectorstores.utils import maximal_marginal_relevance


class DocArrayRetriever(BaseRetriever):
    def __init__(self, index, embeddings, search_field, content_field, search_type='similarity', filters=None, top_k=1):
        self.index = index
        self._schema = self.index._schema
        self._embeddings = embeddings
        self._search_type = search_type
        self._content_field = content_field
        self._search_field = search_field
        self._filters = filters
        self._top_k = top_k

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        query_emb = np.array(self._embeddings.embed_query(query))

        if self._search_type == 'similarity':
            results = self._similarity_search(query_emb)
        elif self._search_type == 'mmr':
            results = self._mrr_search(query_emb)
        else:
            raise ValueError(f"search_type of {self._search_type} not allowed.")

        return results

    def _similarity_search(self, query_emb):
        docs = self.index.find(
            query_emb, search_field=self._search_field, limit=self._top_k
        ).documents

        results = [
            self._docarray_to_langchain_doc(doc) for doc in docs
        ]
        return results

    def _mrr_search(self, query_emb):
        docs = self.index.find(
            query_emb, search_field=self._search_field, limit=20
        ).documents

        mmr_selected = maximal_marginal_relevance(
            query_emb, docs.embedding, k=self._top_k
        )
        results = [
            self._docarray_to_langchain_doc(docs[idx])
            for idx in mmr_selected
        ]
        return results

    def _docarray_to_langchain_doc(self, doc):
        if self._content_field not in doc.__fields__:
            raise ValueError(f"Document {doc} does not contain the content field - {self._content_field}.")
        lc_doc = Document(page_content=getattr(doc, self._content_field))

        for name in doc.__fields__:
            value = getattr(doc, name)
            if isinstance(value, (str, int, float, bool)) and name != self._content_field:
                lc_doc.metadata[name] = value

        return lc_doc

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        pass
