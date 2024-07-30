from typing import List

from langchain_core.documents import Document

from langchain_community.retrievers.breebs import BreebsRetriever


class TestBreebsRetriever:
    def test_breeb_query(self) -> None:
        breeb_key = "Parivoyage"
        query = "What are the best churches to visit in Paris?"
        breeb_retriever = BreebsRetriever(breeb_key)
        documents: List[Document] = breeb_retriever.invoke(query)
        assert isinstance(documents, list), "Documents should be a list"
        for doc in documents:
            assert doc.page_content, "Document page_content should not be None"
            assert doc.metadata["source"], "Document metadata should contain 'source'"
            assert doc.metadata["score"] == 1, "Document score should be equal to 1"
