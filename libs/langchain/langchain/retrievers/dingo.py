"""Dingo Retriever"""
from typing import Any, Dict, List, Optional

from pydantic import Extra, root_validator

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores.dingo import Dingo


class DingoRetriever(BaseRetriever):
    """Retriever that uses the Dingo API"""

    embeddings: Embeddings
    """Embeddings model to use."""
    """description"""
    client: Dingo
    """dingo client to use."""
    index_name: str
    """index name"""
    top_k: int = 4
    """Number of documents to return."""

    def add_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        self.client.add_texts(texts, metadatas, ids)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        # convert the question into a vector
        query_emb = self.embeddings.embed_query(query)
        # query dingo with the query parameters
        results = self.client._client.vector_search(
            index_name=self.index_name,
            xq=query_emb,
            top_k=self.top_k,
            search_params={"meta_expr": {"page_content": query}},
        )
        docs = []
            
        if results == []:
            return []

        for res in results[0]["vectorWithDistances"]:
            metadatas = res["scalarData"]
            id = res["id"]
            score = res['distance']
            page_content = metadatas["page_content"]['fields'][0]['data']  
            metadata = {"id": id, "page_content": page_content, 'score': score}
            docs.append(Document(page_content=text, metadata=metadata))
            
        # return search results as json
        return docs
