from typing import Dict, List

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_pinecone import PineconeVectorStore

from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.vectorstores.faiss import (
    dependable_faiss_import,
)


class SentenceWindowRetriever(BaseRetriever):
    """`Sentence Window` retriever.

    Sentence Window Retriever dissociates retrieval from generation
    by appending adjacent chunks to the retrieved chunk to provide
    additional context to the generation model

    Currently, it supports 3 backends:
    FAISS, Chroma, Pinecone
    """

    store: VectorStore
    """VectorStore containing the chunks of text. Only supports the 
        above listed databases"""

    k: int = 4
    """Number of top results to return"""

    window_size: int = 2
    """Number of adjacent chunks on each side to be added """

    @root_validator
    def check_database_type(cls, values: Dict) -> Dict:
        k = values.get("k", -1)

        if k < 0:
            raise ValueError("The value of k must be greater than 0")

        return values

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementations for retriever."""

        if type(self.store) == Chroma:
            return self._sentence_window_retriever_chroma(query=query)
        elif type(self.store) == PineconeVectorStore:
            return self._sentence_window_retriever_pinecone(query=query)
        elif type(self.store) == FAISS:
            return self._sentence_window_retriever_faiss(query=query)
        else:
            raise ValueError(
                """Only the following databases are currently supported for 
                the implementation of Sentence Window Retriever : 
                FAISS, Chroma, Pinecone"""
            )

    def _sentence_window_retriever_chroma(self, query: str) -> List[Document]:
        assert isinstance(self.store, Chroma)

        vector = self.store.embeddings.embed_query(query)  # type: ignore

        top_results = self.store._collection.query(
            query_embeddings=vector, n_results=self.k
        )

        doc_list = []

        for id, metadata in zip(top_results["ids"][0], top_results["metadatas"][0]):  # type: ignore
            primary_index = metadata.get("chunk_id")
            source_text = metadata.get("source")
            start_index = max(0, primary_index - self.window_size)
            end_index = primary_index + self.window_size

            if source_text:
                output = self.store._collection.get(
                    where={
                        "$and": [
                            {"source": source_text},
                            {"chunk_id": {"$gte": start_index}},
                            {"chunk_id": {"$lte": end_index}},
                        ]
                    }
                )
            else:
                output = self.store._collection.get(
                    where={
                        "$and": [
                            {"chunk_id": {"$gte": start_index}},
                            {"chunk_id": {"$lte": end_index}},
                        ]
                    }
                )

            page_content = " ".join(output["documents"])  # type: ignore
            pages = [x.get("page") for x in output["metadatas"]]  # type: ignore

            metadata = {
                "primary_index": primary_index,
                "chroma_id": id,
                "page": list(set(pages)),
                "type": "sentence_window",
            }

            if source_text:
                metadata["source"] = source_text

            output_doc = Document(page_content=page_content, metadata=metadata)

            doc_list.append(output_doc)

        return doc_list

    def _sentence_window_retriever_pinecone(self, query: str) -> List[Document]:
        assert isinstance(self.store, PineconeVectorStore)

        top_results = self.store.similarity_search(query, k=self.k)

        vector_dimension = self.store._index.describe_index_stats()["dimension"]

        doc_list = []

        for doc in top_results:
            primary_index = doc.metadata.get("chunk_id")

            if primary_index is None:
                raise ValueError(
                    "chunk_id metadata variable is missing in retrieved documents"
                )

            source_text = doc.metadata.get("source")

            start_index = max(0, primary_index - self.window_size)
            end_index = primary_index + self.window_size

            if source_text:
                filter_value = {
                    "$and": [
                        {"source": source_text},
                        {"chunk_id": {"$gte": start_index}},
                        {"chunk_id": {"$lte": end_index}},
                    ]
                }
            else:
                filter_value = {
                    "$and": [
                        {"chunk_id": {"$gte": start_index}},
                        {"chunk_id": {"$lte": end_index}},
                    ]
                }

            output = self.store._index.query(
                vector=[0] * vector_dimension,
                top_k=2 * self.window_size + 1,
                filter=filter_value,
                include_metadata=True,
            )

            output = sorted(output["matches"], key=lambda x: x["metadata"]["chunk_id"])

            page_content = " ".join([x["metadata"].get("text") for x in output])
            pages = [x["metadata"].get("page") for x in output]

            metadata = {
                "primary_index": primary_index,
                "page": list(set(pages)),
                "type": "sentence_window",
            }

            if source_text:
                metadata["source"] = source_text

            output_doc = Document(page_content=page_content, metadata=metadata)

            doc_list.append(output_doc)

        return doc_list

    def _sentence_window_retriever_faiss(self, query: str) -> List[Document]:
        faiss = dependable_faiss_import()

        assert isinstance(self.store, FAISS)

        vector_ = self.store._embed_query(query)
        vector = np.array([vector_], dtype=np.float32)
        if self.store._normalize_L2:
            faiss.normalize_L2(vector)

        # Retrieve the top k indices that matches the query
        scores, indices = self.store.index.search(vector, self.k)

        doc_list = []
        for primary_index in indices[0]:
            start_index = max(0, primary_index - self.window_size)
            end_index = min(
                primary_index + self.window_size,
                len(self.store.index_to_docstore_id) - 1,
            )
            primary_doc = self.store.docstore.search(
                self.store.index_to_docstore_id[primary_index]
            )

            assert isinstance(primary_doc, Document)
            primary_doc_source = primary_doc.metadata.get("source", None)

            # Retrieving content for neighboring indices
            page_content_list = []
            pages = []
            for index in range(start_index, end_index + 1):
                doc = self.store.docstore.search(self.store.index_to_docstore_id[index])
                assert isinstance(doc, Document)

                if doc.metadata.get("source", None) == primary_doc_source:
                    # We only want to include adjacent indices that are
                    # from the same source text. If source is not provided,
                    # then this condition is relaxed
                    page_content_list.append(doc.page_content)
                    pages.append(doc.metadata["page"])

            # Creating new output Document
            assert isinstance(doc, Document)

            page_content = " ".join(page_content_list)

            metadata = {
                "primary_index": primary_index,
                "page": list(set(pages)),
                "type": "sentence_window",
            }

            if primary_doc_source:
                metadata["source"] = primary_doc_source

            output_doc = Document(page_content=page_content, metadata=metadata)

            doc_list.append(output_doc)

        return doc_list
