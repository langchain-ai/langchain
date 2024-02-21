# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The wrapper for Child-Parent retriever based on langchain"""

from enum import Enum
from typing import List
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.pydantic_v1 import Field
from langchain_core.documents import Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.text_splitter import TextSplitter

class SearchType(str, Enum):
    """Enumerator of the types of search to perform."""

    similarity = "similarity"
    """Similarity search."""
    mmr = "mmr"
    """Maximal Marginal Relevance reranking of similarity search."""
    
    
class ChildParentRetriever(BaseRetriever):
    """Retrieve the small child chucks and return their long-context parent chucks.
    1. You may want to have small documents, so that their embeddings can most
        accurately reflect their meaning. If too long, then the embeddings can
        lose meaning.
    2. You want to have long enough documents that the context of each chunk is
        retained.
    
    The child parent retriever can maintain the accuracy and the comprehensiveness of the retrieved results. During 
    the retrieval, it first retrieve the knowledge base with small chuck. Then it will lookup the parent chucks with
    identify id to return longer context.
    
    Differenct forom the ParentDocumentRetriever, ChildParentRetriever supports local database management to save the 
    loading memory and provide fast database initialization.
    
    """
    vectorstore: VectorStore
    parentstore: VectorStore
    child_splitter: TextSplitter
    id_key: str = "identify_id"
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""
    search_type: SearchType = SearchType.similarity
    """Type of search to perform (similarity / mmr)"""

    def add_documents(
            self,
            documents: List[Document],
            ids: Optional[List[str]] = None,
    ) -> None:
        """Adds documents to the parentstore and vectorstores.

        Args:
            documents: List of documents to add
            ids: Optional list of ids for documents. If provided should be the same
                length as the list of documents. 
        """
        if ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in documents]
        else:
            if len(documents) != len(ids):
                raise ValueError(
                    "Got uneven list of documents and ids. "
                    "If `ids` is provided, should be same length as `documents`."
                )
            doc_ids = ids

        docs = []
        parent_docs = []
        for i, doc in enumerate(documents):
            _id = doc_ids[i]
            sub_docs = self.child_splitter.split_documents([doc])
            for _doc in sub_docs:
                _doc.metadata[self.id_key] = _id
            docs.extend(sub_docs)
            doc.metadata["doc_id"] = _id
            parent_docs.append(doc)
        self.vectorstore.add_documents(docs)
        self.parentstore.add_documents(parent_docs)
        

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            The concatation of the retrieved documents and the link
        """
        if self.search_type == SearchType.mmr:
            sub_docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            sub_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)

        ids = []
        for instance in sub_docs:
            if instance.metadata["identify_id"] not in ids:
                ids.append(d.metadata['identify_id'])
        retrieved_documents = self.parentstore.get(ids)
        docs = []
        for i in range(len(retrieved_documents['documents']))
            doc = Document(page_content=retrieved_documents['documents'][i],  \
                           metadata=retrieved_documents['metadatas'][i])
            docs.append(doc)
        return docs