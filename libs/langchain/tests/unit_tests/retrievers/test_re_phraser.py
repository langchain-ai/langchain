from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from langchain_classic.retrievers.re_phraser import RePhraseQueryRetriever


class _FakeRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return [Document(page_content=f"relevant for: {query}")]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> list[Document]:
        return [Document(page_content=f"relevant for: {query}")]


def test_sync_invoke() -> None:
    """Sync invoke rephrases the query and retrieves documents."""
    chain = RunnableLambda(lambda q: "rephrased query")
    retriever = RePhraseQueryRetriever(retriever=_FakeRetriever(), llm_chain=chain)

    result = retriever.invoke("what is langchain?")

    assert len(result) == 1
    assert result[0].page_content == "relevant for: rephrased query"


async def test_async_invoke() -> None:
    """Async invoke rephrases the query and retrieves documents.

    Regression test: _aget_relevant_documents used to raise NotImplementedError.
    """
    chain = RunnableLambda(lambda q: "rephrased query async")
    retriever = RePhraseQueryRetriever(retriever=_FakeRetriever(), llm_chain=chain)

    result = await retriever.ainvoke("what is langchain?")

    assert len(result) == 1
    assert result[0].page_content == "relevant for: rephrased query async"
