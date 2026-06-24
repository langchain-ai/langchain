import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from langchain_classic.retrievers.re_phraser import RePhraseQueryRetriever


class FakeRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content=f"relevant: {query}")]

    async def _aget_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content=f"relevant: {query}")]


@pytest.mark.asyncio
async def test_rephrase_query_retriever_async() -> None:
    chain = RunnableLambda(lambda q: f"rephrased: {q}")
    retriever = RePhraseQueryRetriever(retriever=FakeRetriever(), llm_chain=chain)

    sync_result = retriever.invoke("what is langchain?")
    assert len(sync_result) == 1
    assert sync_result[0].page_content == "relevant: rephrased: what is langchain?"

    async_result = await retriever.ainvoke("what is langchain?")
    assert len(async_result) == 1
    assert async_result[0].page_content == "relevant: rephrased: what is langchain?"
