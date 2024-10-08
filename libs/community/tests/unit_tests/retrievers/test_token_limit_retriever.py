import asyncio
from typing import Dict, List

import pytest
from langchain.pydantic_v1 import root_validator
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_community.retrievers.token_limit_retriever import TokenLimitRetriver


@pytest.fixture(scope="module")
def toy_retriever():
    class ToyRetriever(BaseRetriever):
        """A toy retriever that contains the top k documents
        that contain the user query.

        This retriever only implements the sync method _get_relevant_documents.

        If the retriever were to involve file access or network access, it could benefit
        from a native async implementation of `_aget_relevant_documents`.

        As usual, with Runnables, there's a default async implementation that's provided
        that delegates to the sync implementation running on another thread.
        """

        documents: List[Document] = [
            Document("this is a nice day"),
            Document("Peter looks great today"),
            Document("Will AI take over the world?"),
            Document("Langchain is a useful community contributed library."),
        ]

        """List of documents to retrieve from."""
        k: int
        """Number of top results to return"""

        @root_validator()
        def validate_environment(cls, values: Dict) -> Dict:
            return values

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            """Sync implementations for retriever."""
            matching_documents = []
            for document in self.documents:
                if len(matching_documents) > self.k:
                    return matching_documents

                if query.lower() in document.page_content.lower():
                    matching_documents.append(document)
            return matching_documents

    yield ToyRetriever(k=4)


@pytest.mark.requires("tiktoken")
@pytest.fixture(scope="module")
def get_tokenize_tiktoken():
    def tokenize_tiktoken(text):
        import tiktoken

        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = enc.encode(text)
        return tokens

    yield tokenize_tiktoken


@pytest.mark.requires("tiktoken")
@pytest.fixture(scope="module")
def get_decode_tiktoken_upto():
    def decode_tiktoken_upto(text: str, n_token_left: int):
        import tiktoken

        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return enc.decode(enc.encode(text)[:n_token_left])

    yield decode_tiktoken_upto


@pytest.mark.requires("tiktoken")
def test_tiktoken_partial(
    toy_retriever, get_tokenize_tiktoken, get_decode_tiktoken_upto
):
    test_retriever_to_be_wrapped = toy_retriever
    simple_token_limit_chain = TokenLimitRetriver(
        token_limit=20,
        remaining_token_fillup_callback=get_decode_tiktoken_upto,
        tokeniser_callback=get_tokenize_tiktoken,
        retriever=test_retriever_to_be_wrapped,
        token_cutoff_strategy="partial_document",
    )
    res = simple_token_limit_chain.invoke("a")
    assert len(res) == 4


def test_tiktoken_complete_parameter_error(
    toy_retriever, get_tokenize_tiktoken, get_decode_tiktoken_upto
):
    from pydantic.v1.error_wrappers import ValidationError

    test_retriever_to_be_wrapped = toy_retriever
    # test for invalid optoins
    try:
        simple_token_limit_chain = TokenLimitRetriver(
            token_limit=20,
            remaining_token_fillup_callback=get_decode_tiktoken_upto,
            tokeniser_callback=get_tokenize_tiktoken,
            retriever=test_retriever_to_be_wrapped,
            token_cutoff_strategy="complete_document",
        )

        res = simple_token_limit_chain.invoke("a")
        assert len(res) == 3
    except ValidationError:
        pass


def test_tiktoken_complete_parameter_correct(toy_retriever, get_tokenize_tiktoken):
    test_retriever_to_be_wrapped = toy_retriever
    # test for complete document options
    simple_token_limit_chain = TokenLimitRetriver(
        token_limit=20,
        tokeniser_callback=get_tokenize_tiktoken,
        retriever=test_retriever_to_be_wrapped,
        token_cutoff_strategy="complete_document",
    )
    res = simple_token_limit_chain.invoke("a")
    assert len(res) == 3


def test_no_dependency(toy_retriever):
    test_retriever_to_be_wrapped = toy_retriever

    def tokenize(text):
        return text.split(" ")

    def decode_token_upto(text: str, n_token_left: int):
        return " ".join(text.split(" ")[:n_token_left])

    simple_token_limit_chain = TokenLimitRetriver(
        token_limit=20,
        tokeniser_callback=tokenize,
        retriever=test_retriever_to_be_wrapped,
        token_cutoff_strategy="complete_document",
    )
    res = simple_token_limit_chain.invoke("a")
    assert len(res) == 3
    res = asyncio.run(simple_token_limit_chain.ainvoke("a"))
    assert len(res) == 3
    simple_token_limit_chain = TokenLimitRetriver(
        token_limit=20,
        remaining_token_fillup_callback=decode_token_upto,
        tokeniser_callback=tokenize,
        retriever=test_retriever_to_be_wrapped,
        token_cutoff_strategy="partial_document",
    )
    res = simple_token_limit_chain.invoke("a")
    assert len(res) == 4
    res = asyncio.run(simple_token_limit_chain.ainvoke("a"))
    assert len(res) == 4


def run_tests():
    pytest.main([__file__])


if __name__ == "__main__":
    run_tests()
