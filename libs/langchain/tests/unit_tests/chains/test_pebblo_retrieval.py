"""
Unit tests for the PebbloRetrievalQA chain
"""
from typing import List
from unittest.mock import Mock

import pytest
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from langchain.chains.pebblo_retrieval.base import PebbloRetrievalQA
from tests.unit_tests.llms.fake_llm import FakeLLM


class FakeRetriever(VectorStoreRetriever):
    """
    Test util that parrots the query back as documents
    """

    vectorstore: VectorStore = Mock()

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return [Document(page_content=query)]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        return [Document(page_content=query)]


@pytest.fixture
def unsupported_retriever() -> FakeRetriever:
    """
    Create a FakeRetriever instance
    """
    retriever = FakeRetriever()
    retriever.search_kwargs = {}
    # Set the class of vectorstore to Chroma
    retriever.vectorstore.__class__ = Chroma
    return retriever


@pytest.fixture
def retriever() -> FakeRetriever:
    """
    Create a FakeRetriever instance
    """
    retriever = FakeRetriever()
    retriever.search_kwargs = {}
    # Set the class of vectorstore to Pinecone
    retriever.vectorstore.__class__ = Pinecone
    return retriever


@pytest.fixture
def pebblo_retrieval_qa(retriever: FakeRetriever) -> PebbloRetrievalQA:
    """
    Create a PebbloRetrievalQA instance
    """
    # Create a fake auth context
    auth_context = {"authorized_identities": ["fake_user", "fake_user2"]}
    semantic_context = {
        "pebblo_semantic_topics": {"deny": ["Harmful Advice"]},
        "pebblo_semantic_entities": {"deny": ["SSN"]},
    }
    pebblo_retrieval_qa = PebbloRetrievalQA.from_chain_type(
        llm=FakeLLM(),
        chain_type="stuff",
        retriever=retriever,
        auth_context=auth_context,
        semantic_context=semantic_context,
    )

    return pebblo_retrieval_qa


def test_invoke(pebblo_retrieval_qa: PebbloRetrievalQA) -> None:
    """
    Test that the invoke method returns a non-None result
    """
    question = "What is the meaning of life?"
    response = pebblo_retrieval_qa.invoke({"query": question})
    assert response is not None


@pytest.mark.asyncio
async def test_ainvoke(pebblo_retrieval_qa: PebbloRetrievalQA) -> None:
    """
    Test ainvoke method (async) raises NotImplementedError
    """
    with pytest.raises(NotImplementedError):
        _ = await pebblo_retrieval_qa.ainvoke({"query": "hello"})


def test_validate_vectorstore(
    retriever: FakeRetriever, unsupported_retriever: FakeRetriever
) -> None:
    """
    Test vectorstore validation
    """

    # No exception should be raised for supported vectorstores (Pinecone)
    _ = PebbloRetrievalQA.from_chain_type(
        llm=FakeLLM(),
        chain_type="stuff",
        retriever=retriever,
    )

    # validate_vectorstore method should raise a ValueError for unsupported vectorstores
    with pytest.raises(ValueError) as exc_info:
        _ = PebbloRetrievalQA.from_chain_type(
            llm=FakeLLM(),
            chain_type="stuff",
            retriever=unsupported_retriever,
        )
    assert (
        "Vectorstore must be an instance of one of the supported vectorstores"
        in str(exc_info.value)
    )


def test__set_identity_enforcement_filter(
    pebblo_retrieval_qa: PebbloRetrievalQA,
) -> None:
    """
    Test that the _set_identity_enforcement_filter method sets the correct filter in
    search_kwargs
    """
    pebblo_retrieval_qa._set_identity_enforcement_filter()
    assert "authorized_identities" in pebblo_retrieval_qa.retriever.search_kwargs.get(
        "filter", {}
    )


def test__set_semantic_enforcement_filter(
    pebblo_retrieval_qa: PebbloRetrievalQA,
) -> None:
    """
    Test that the _set_semantic_enforcement_filter method sets the correct filter in
    search_kwargs
    """
    pebblo_retrieval_qa._set_semantic_enforcement_filter()
    assert "pebblo_semantic_topics" in pebblo_retrieval_qa.retriever.search_kwargs.get(
        "filter", {}
    )
    assert (
        "pebblo_semantic_entities"
        in pebblo_retrieval_qa.retriever.search_kwargs.get("filter", {})
    )
