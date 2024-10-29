"""
Unit tests for the PebbloRetrievalQA chain
"""

from typing import Generator, List
from unittest.mock import Mock, patch

import pytest
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.vectorstores import (
    InMemoryVectorStore,
    VectorStore,
    VectorStoreRetriever,
)

from langchain_community.chains import PebbloRetrievalQA
from langchain_community.chains.pebblo_retrieval.models import (
    AuthContext,
    ChainInput,
    SemanticContext,
)
from langchain_community.vectorstores.pinecone import Pinecone
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
    pebblo_retrieval_qa = PebbloRetrievalQA.from_chain_type(
        llm=FakeLLM(),
        chain_type="stuff",
        retriever=retriever,
        owner="owner",
        description="description",
        app_name="app_name",
    )
    pebblo_retrieval_qa.pb_client = Mock()
    pebblo_retrieval_qa.pb_client.send_prompt = Mock()
    pebblo_retrieval_qa.pb_client.enforce_identity_policy = Mock(
        return_value=(None, None, False)
    )
    pebblo_retrieval_qa.pb_client.check_prompt_validity = Mock(
        return_value=(None, dict())
    )
    return pebblo_retrieval_qa


@pytest.fixture
def mock_update_enforcement_filters() -> Generator[Mock, None, None]:
    with patch(
        "langchain_community.chains.pebblo_retrieval.base.update_enforcement_filters"
    ) as mock:
        yield mock


def test_invoke(pebblo_retrieval_qa: PebbloRetrievalQA) -> None:
    """
    Test that the invoke method returns a non-None result
    """
    # Create a fake auth context and semantic context
    auth_context = AuthContext(
        user_id="fake_user@email.com",
        user_auth=["fake-group", "fake-group2"],
    )
    semantic_context_dict = {
        "pebblo_semantic_topics": {"deny": ["harmful-advice"]},
        "pebblo_semantic_entities": {"deny": ["credit-card"]},
    }
    semantic_context = SemanticContext(**semantic_context_dict)

    question = "What is the meaning of life?"

    chain_input_obj = ChainInput(
        query=question, auth_context=auth_context, semantic_context=semantic_context
    )
    response = pebblo_retrieval_qa.invoke(chain_input_obj.dict())
    assert response is not None


def test_validate_vectorstore(retriever: FakeRetriever) -> None:
    """
    Test vectorstore validation
    """

    # No exception should be raised for supported vectorstores (Pinecone)
    _ = PebbloRetrievalQA.from_chain_type(
        llm=FakeLLM(),
        chain_type="stuff",
        retriever=retriever,
        owner="owner",
        description="description",
        app_name="app_name",
    )

    unsupported_retriever = FakeRetriever()
    unsupported_retriever.search_kwargs = {}
    # Set the class of vectorstore
    unsupported_retriever.vectorstore.__class__ = InMemoryVectorStore

    # validate_vectorstore method should raise a ValueError for unsupported vectorstores
    with pytest.raises(ValueError) as exc_info:
        _ = PebbloRetrievalQA.from_chain_type(
            llm=FakeLLM(),
            chain_type="stuff",
            retriever=unsupported_retriever,
            owner="owner",
            description="description",
            app_name="app_name",
        )
    assert (
        "Vectorstore must be an instance of one of the supported vectorstores"
        in str(exc_info.value)
    )


@pytest.mark.parametrize(
    "is_privileged_user, expected_count",
    [
        (True, 0),  # Privileged user
        (False, 1),  # Non-privileged user
    ],
)
def test_policy_enforcement(
    pebblo_retrieval_qa: PebbloRetrievalQA,
    is_privileged_user: bool,
    expected_count: int,
) -> None:
    """
    Test policy enforcement for both Privileged user and Non-privileged user.
    The get_semantic_context and _set_semantic_enforcement_filter methods should be
    called based on the user's role.
    Privileged user should not have any enforcement filters applied so these methods
    should not be called.
    """
    question = "Tell me the secret of the universe"
    auth_context = AuthContext(user_id="user@email.com", user_auth=["group1", "group2"])
    semantic_ctx = SemanticContext(
        **{
            "pebblo_semantic_topics": {"deny": ["harmful-advice"]},
            "pebblo_semantic_entities": {"deny": ["credit-card"]},
        }
    )
    chain_input_obj = ChainInput(query=question, auth_context=auth_context)

    with patch.object(
        pebblo_retrieval_qa.pb_client,
        "is_privileged_user",
        return_value=is_privileged_user,
    ) as mock_is_privileged_user, patch.object(
        pebblo_retrieval_qa.pb_client, "get_semantic_context", return_value=semantic_ctx
    ) as mock_get_semantic_context, patch(
        "langchain_community.chains.pebblo_retrieval.enforcement_filters._set_semantic_enforcement_filter"
    ) as mock_set_semantic_enforcement_filter:
        _ = pebblo_retrieval_qa.invoke(chain_input_obj.dict())
        assert mock_is_privileged_user.call_count == 1
        assert mock_get_semantic_context.call_count == expected_count
        assert mock_set_semantic_enforcement_filter.call_count == expected_count
