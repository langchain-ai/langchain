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
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
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
def pebblo_retrieval_qa(retriever: FakeRetriever) -> BaseRetrievalQA:
    """
    Create a PebbloRetrievalQA instance
    """
    # Create a fake auth context
    auth_context = {"authorized_identities": ["fake_user", "fake_user2"]}
    pebblo_retrieval_qa = PebbloRetrievalQA.from_chain_type(
        llm=FakeLLM(),
        chain_type="stuff",
        retriever=retriever,
        auth_context=auth_context,
    )
    return pebblo_retrieval_qa


def test_invoke(pebblo_retrieval_qa: BaseRetrievalQA) -> None:
    """
    Test that the invoke method returns a non-None result
    """
    question = "What is the meaning of life?"
    response = pebblo_retrieval_qa.invoke({"query": question})
    assert response is not None


@pytest.mark.asyncio
async def test_ainvoke(pebblo_retrieval_qa: BaseRetrievalQA) -> None:
    """
    Test ainvoke method (async) raises NotImplementedError
    """
    with pytest.raises(NotImplementedError):
        _ = await pebblo_retrieval_qa.ainvoke({"query": "hello"})


def test_validate_auth_context(retriever: FakeRetriever) -> None:
    """
    Test the auth_context validation with valid and invalid inputs
    """
    # Test with valid auth_context
    valid_auth_context = {"authorized_identities": ["fake_user", "fake_user2"]}
    _ = PebbloRetrievalQA.from_chain_type(
        llm=FakeLLM(),
        chain_type="stuff",
        retriever=retriever,
        auth_context=valid_auth_context,
    )

    # Test with invalid auth_context
    invalid_auth_context = {"authorized_user": "fake_user"}
    with pytest.raises(ValueError) as exc_info:
        _ = PebbloRetrievalQA.from_chain_type(
            llm=FakeLLM(),
            chain_type="stuff",
            retriever=retriever,
            auth_context=invalid_auth_context,
        )
    assert "auth_context must contain 'authorized_identities'" in str(exc_info.value)

    # Test with  auth_context invalid authorized_identities
    invalid_auth_context = {"authorized_identities": "fake_user"}
    with pytest.raises(ValueError) as exc_info:
        _ = PebbloRetrievalQA.from_chain_type(
            llm=FakeLLM(),
            chain_type="stuff",
            retriever=retriever,
            auth_context=invalid_auth_context,
        )
    assert "authorized_identities must be a list" in str(exc_info.value)


def test_validate_vectorstore(
    retriever: FakeRetriever, unsupported_retriever: FakeRetriever
) -> None:
    """
    Test vectorstore validation
    """
    auth_context = {"authorized_identities": ["fake_user", "fake_user2"]}

    # Test with a supported vectorstore (Pinecone)
    _ = PebbloRetrievalQA.from_chain_type(
        llm=FakeLLM(),
        chain_type="stuff",
        retriever=retriever,
        auth_context=auth_context,
    )

    # Test with an unsupported vectorstore
    with pytest.raises(ValueError) as exc_info:
        _ = PebbloRetrievalQA.from_chain_type(
            llm=FakeLLM(),
            chain_type="stuff",
            retriever=unsupported_retriever,
            auth_context=auth_context,
        )
    assert (
        "Vectorstore must be an instance of one of the supported vectorstores"
        in str(exc_info.value)
    )
