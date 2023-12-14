from typing import List
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_community.retrievers.breebs import BreebsKnowledgeRetriever


class TestBreebsKnowledgeRetriever:
    def test_breeb_query(self) -> None:
        K = 4
        breeb_key = "Parivoyage"
        query = "What are the best churches to visit in Paris?"
        breeb_retriever = BreebsKnowledgeRetriever(breeb_key)
        documents: List[Document] = breeb_retriever._get_relevant_documents(
            query, run_manager=CallbackManagerForRetrieverRun
        )
        assert isinstance(documents, list), "Documents should be a list"
        assert len(documents) == K, f"Documents should contain {K} items"
        for doc in documents:
            assert doc.page_content, "Document page_content should not be None"
            assert doc.metadata["source"], "Document metadata should contain 'source'"
            assert doc.metadata["score"] == 1, "Document score should be equal to 1"
