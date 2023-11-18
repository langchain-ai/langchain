"""Integration test for Wikipedia API Wrapper."""
from typing import List

import pytest

from langchain.retrievers import WikipediaRetriever
from langchain.schema import Document


@pytest.fixture
def retriever() -> WikipediaRetriever:
    return WikipediaRetriever()


def assert_docs(docs: List[Document], all_meta: bool = False) -> None:
    for doc in docs:
        assert doc.page_content
        assert doc.metadata
        main_meta = {"title", "summary"}
        assert set(doc.metadata).issuperset(main_meta)
        if all_meta:
            assert len(set(doc.metadata)) > len(main_meta)
        else:
            assert len(set(doc.metadata)) == len(main_meta)


def test_load_success(retriever: WikipediaRetriever) -> None:
    docs = retriever.get_relevant_documents("HUNTER X HUNTER")
    assert len(docs) > 1
    assert_docs(docs, all_meta=False)


def test_load_success_all_meta(retriever: WikipediaRetriever) -> None:
    retriever.load_all_available_meta = True
    docs = retriever.get_relevant_documents("HUNTER X HUNTER")
    assert len(docs) > 1
    assert_docs(docs, all_meta=True)


def test_load_success_init_args() -> None:
    retriever = WikipediaRetriever(
        lang="en", top_k_results=1, load_all_available_meta=True
    )
    docs = retriever.get_relevant_documents("HUNTER X HUNTER")
    assert len(docs) == 1
    assert_docs(docs, all_meta=True)


def test_load_no_result(retriever: WikipediaRetriever) -> None:
    docs = retriever.get_relevant_documents(
        "NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL"
    )
    assert not docs
