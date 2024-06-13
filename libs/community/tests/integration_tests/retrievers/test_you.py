import os

from langchain_community.retrievers.you import YouRetriever


class TestYouRetriever:
    @classmethod
    def setup_class(cls) -> None:
        if not os.getenv("YDC_API_KEY"):
            raise ValueError("YDC_API_KEY environment variable is not set")

    def test_invoke(self) -> None:
        retriever = YouRetriever()
        actual = retriever.invoke("test")

        assert len(actual) > 0
