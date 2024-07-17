"""Integration test for Wikipedia Document Loader."""

from typing import List

from langchain_core.documents import Document

from langchain_community.document_loaders import WikipediaLoader


def assert_docs(docs: List[Document], all_meta: bool = False) -> None:
    for doc in docs:
        assert doc.page_content
        assert doc.metadata
        main_meta = {"title", "summary", "source"}
        assert set(doc.metadata).issuperset(main_meta)
        if all_meta:
            assert len(set(doc.metadata)) > len(main_meta)
        else:
            assert len(set(doc.metadata)) == len(main_meta)


def test_load_success() -> None:
    loader = WikipediaLoader(query="HUNTER X HUNTER")
    docs = loader.load()
    assert len(docs) > 1
    assert len(docs) <= 25
    assert_docs(docs, all_meta=False)


def test_load_success_all_meta() -> None:
    load_max_docs = 5
    load_all_available_meta = True
    loader = WikipediaLoader(
        query="HUNTER X HUNTER",
        load_max_docs=load_max_docs,
        load_all_available_meta=load_all_available_meta,
    )
    docs = loader.load()
    assert len(docs) == load_max_docs
    assert_docs(docs, all_meta=load_all_available_meta)


def test_load_success_more() -> None:
    load_max_docs = 10
    loader = WikipediaLoader(query="HUNTER X HUNTER", load_max_docs=load_max_docs)
    docs = loader.load()
    assert len(docs) == load_max_docs
    assert_docs(docs, all_meta=False)


def test_load_no_result() -> None:
    loader = WikipediaLoader(
        "NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL_NORESULTCALL"
    )
    docs = loader.load()
    assert not docs
