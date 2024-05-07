from typing import Dict, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_pinecone import PineconeVectorStore

from langchain_community.vectorstores import Chroma, Milvus


class SentenceWindowRetriever(BaseRetriever):
    """`Sentence Window` retriever.

    Sentence Window Retriever dissociates retrieval from generation
    by appending adjacent chunks to the retrieved chunk to provide
    additional context to the generation model

    Currently, it supports 23 backends:
    Milvus, Pinecone, Chroma
    """

    store: VectorStore
    """VectorStore containing the chunks of text. Only supports the 
        above listed databases"""

    k: int = 4
    """Number of top results to return"""

    window_size: int = 2
    """Number of adjacent chunks on each side to be added """

    @root_validator
    def validate_input_values(cls, values: Dict) -> Dict:
        k = values.get("k", -1)
        store = values.get("store")
        window_size = values.get("window_size", -1)

        if type(store) not in [Chroma, Milvus, PineconeVectorStore]:
            raise ValueError(
                """Current SWR implementation only supports the following 
            datastores : Milvus, Pinecone, Chroma"""
            )

        if k <= 0:
            raise ValueError("The value of k must be greater than 0")

        if window_size <= 0:
            raise ValueError("The value of window_size must be greater than 0")

        return values

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementations for retriever."""
        vector = self.store.embeddings.embed_query(query)  # type: ignore

        try:
            results = self.store.similarity_search_by_vector(
                embedding=vector, k=4, include_id=True
            )
        except NotImplementedError:
            results = self.store.similarity_search_by_vector_with_score(
                embedding=vector, k=4, include_id=True
            )

        output_docs = []

        for res in results:
            output_docs.append(self.get_swr_result(res))

        return output_docs

    def get_swr_result(self, doc: Document) -> Document:
        """For a given document, retrieves the adjacent documents
           and outputs a combined document

        Args:
            doc (Document):
        """

        doc_id = int(doc.metadata.get("id"))

        start_index = max(0, doc_id - self.window_size)
        end_index = doc_id + self.window_size

        window_docs = self.store.get_documents_by_ids(
            list(range(start_index, end_index + 1))
        )

        return self.merge_window_docs(doc, window_docs)

    def merge_window_docs(self, doc: Document, window_docs: List[Document]) -> Document:
        source_text = doc.metadata.get("source")

        page_content = []
        pages = []
        for doc_ in window_docs:
            if source_text == doc_.metadata.get("source"):
                page_content.append(doc_.page_content)
                pages.append(doc_.metadata.get("page"))

        metadata = {
            "source": source_text,
            "pages": list(set(pages)),
            "type": "sentence_window",
            "primary_id": doc.metadata.get("id"),
        }
        page_content = " ".join(page_content)

        return Document(page_content=page_content, metadata=metadata)
