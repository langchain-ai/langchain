import os

from langchain.retrievers.you import YouRetriever


class TestYouRetriever:
    @classmethod
    def setup_class(cls) -> None:
        if not os.getenv("YDC_API_KEY"):
            raise ValueError("YDC_API_KEY environment variable is not set")

    def test_get_relevant_documents(self) -> None:
        retriever = YouRetriever()
        actual = retriever.get_relevant_documents("test")

        assert len(actual) > 0
