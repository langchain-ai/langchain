"""Tests for MergerRetriever."""

from langchain_core.documents import Document
from langchain_core.language_models import FakeListLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.retrievers import MergerRetriever
from tests.unit_tests.retrievers.parrot_retriever import FakeParrotRetriever


class MockRetriever(BaseRetriever):
    """Mock retriever for testing."""

    def __init__(self, docs: list[Document]):
        super().__init__()
        self.docs = docs

    def _get_relevant_documents(
        self, query: str, *, run_manager
    ) -> list[Document]:
        return self.docs


def test_merger_retriever_with_base_retrievers() -> None:
    """Test MergerRetriever with BaseRetriever objects."""
    docs1 = [Document(page_content="doc1"), Document(page_content="doc2")]
    docs2 = [Document(page_content="doc3"), Document(page_content="doc4")]

    retriever1 = MockRetriever(docs1)
    retriever2 = MockRetriever(docs2)

    merger = MergerRetriever(retrievers=[retriever1, retriever2])

    result = merger.invoke("test query")

    # Should merge documents from both retrievers
    assert len(result) == 4
    assert result[0].page_content == "doc1"
    assert result[1].page_content == "doc3"
    assert result[2].page_content == "doc2"
    assert result[3].page_content == "doc4"


def test_merger_retriever_with_history_aware_retriever() -> None:
    """Test MergerRetriever with create_history_aware_retriever (RetrieverOutputLike)."""
    # Create a simple retriever
    docs = [Document(page_content="test document")]
    base_retriever = MockRetriever(docs)

    # Create a history aware retriever
    llm = FakeListLLM(responses=["rephrased query"])
    prompt = PromptTemplate.from_template("Rephrase: {input}")
    history_aware_retriever = create_history_aware_retriever(
        llm, base_retriever, prompt
    )

    # Create another simple retriever
    docs2 = [Document(page_content="another document")]
    retriever2 = MockRetriever(docs2)

    # Create MergerRetriever with both types
    merger = MergerRetriever(retrievers=[history_aware_retriever, retriever2])

    # This should work without ValidationError
    result = merger.invoke("test query")

    # Should have documents from both retrievers
    assert len(result) == 2
    assert any(doc.page_content == "test document" for doc in result)
    assert any(doc.page_content == "another document" for doc in result)


def test_merger_retriever_mixed_types() -> None:
    """Test MergerRetriever with mixed BaseRetriever and RetrieverOutputLike types."""
    # Create base retrievers
    docs1 = [Document(page_content="base retriever doc")]
    base_retriever1 = MockRetriever(docs1)

    docs2 = [Document(page_content="another base retriever doc")]
    base_retriever2 = MockRetriever(docs2)

    # Create history aware retriever
    llm = FakeListLLM(responses=["rephrased"])
    prompt = PromptTemplate.from_template("Rephrase: {input}")
    history_aware_retriever = create_history_aware_retriever(
        llm, base_retriever1, prompt
    )

    # Create MergerRetriever with mixed types
    merger = MergerRetriever(retrievers=[base_retriever2, history_aware_retriever])

    # This should work without ValidationError
    result = merger.invoke("test query")

    # Should have documents from both retrievers
    assert len(result) == 2
    assert any(doc.page_content == "another base retriever doc" for doc in result)
    assert any(doc.page_content == "base retriever doc" for doc in result)


async def test_merger_retriever_async() -> None:
    """Test MergerRetriever async functionality with mixed types."""
    # Create base retrievers
    docs1 = [Document(page_content="async doc 1")]
    base_retriever1 = MockRetriever(docs1)

    docs2 = [Document(page_content="async doc 2")]
    base_retriever2 = MockRetriever(docs2)

    # Create history aware retriever
    llm = FakeListLLM(responses=["async rephrased"])
    prompt = PromptTemplate.from_template("Async rephrase: {input}")
    history_aware_retriever = create_history_aware_retriever(
        llm, base_retriever1, prompt
    )

    # Create MergerRetriever with mixed types
    merger = MergerRetriever(retrievers=[base_retriever2, history_aware_retriever])

    # Test async invoke
    result = await merger.ainvoke("async test query")

    # Should have documents from both retrievers
    assert len(result) == 2
    assert any(doc.page_content == "async doc 2" for doc in result)
    assert any(doc.page_content == "async doc 1" for doc in result)
