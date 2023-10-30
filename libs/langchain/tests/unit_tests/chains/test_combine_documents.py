"""Test functionality related to combining documents."""

from typing import Any, List

import pytest

from langchain.chains.combine_documents.reduce import (
    collapse_docs,
    split_list_of_docs,
)
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from tests.unit_tests.llms.fake_llm import FakeLLM


def _fake_docs_len_func(docs: List[Document]) -> int:
    return len(_fake_combine_docs_func(docs))


def _fake_combine_docs_func(docs: List[Document], **kwargs: Any) -> str:
    return "".join([d.page_content for d in docs])


def test_multiple_input_keys() -> None:
    chain = load_qa_with_sources_chain(FakeLLM(), chain_type="stuff")
    assert chain.input_keys == ["input_documents", "question"]


def test__split_list_long_single_doc() -> None:
    """Test splitting of a long single doc."""
    docs = [Document(page_content="foo" * 100)]
    with pytest.raises(ValueError):
        split_list_of_docs(docs, _fake_docs_len_func, 100)


def test__split_list_single_doc() -> None:
    """Test splitting works with just a single doc."""
    docs = [Document(page_content="foo")]
    doc_list = split_list_of_docs(docs, _fake_docs_len_func, 100)
    assert doc_list == [docs]


def test__split_list_double_doc() -> None:
    """Test splitting works with just two docs."""
    docs = [Document(page_content="foo"), Document(page_content="bar")]
    doc_list = split_list_of_docs(docs, _fake_docs_len_func, 100)
    assert doc_list == [docs]


def test__split_list_works_correctly() -> None:
    """Test splitting works correctly."""
    docs = [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
        Document(page_content="foo" * 2),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]
    doc_list = split_list_of_docs(docs, _fake_docs_len_func, 10)
    expected_result = [
        # Test a group of three.
        [
            Document(page_content="foo"),
            Document(page_content="bar"),
            Document(page_content="baz"),
        ],
        # Test a group of two, where one is bigger.
        [Document(page_content="foo" * 2), Document(page_content="bar")],
        # Test no errors on last
        [Document(page_content="baz")],
    ]
    assert doc_list == expected_result


def test__collapse_docs_no_metadata() -> None:
    """Test collapse documents functionality when no metadata."""
    docs = [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]
    output = collapse_docs(docs, _fake_combine_docs_func)
    expected_output = Document(page_content="foobarbaz")
    assert output == expected_output


def test__collapse_docs_one_doc() -> None:
    """Test collapse documents functionality when only one document present."""
    # Test with no metadata.
    docs = [Document(page_content="foo")]
    output = collapse_docs(docs, _fake_combine_docs_func)
    assert output == docs[0]

    # Test with metadata.
    docs = [Document(page_content="foo", metadata={"source": "a"})]
    output = collapse_docs(docs, _fake_combine_docs_func)
    assert output == docs[0]


def test__collapse_docs_metadata() -> None:
    """Test collapse documents functionality when metadata exists."""
    metadata1 = {"source": "a", "foo": 2, "bar": "1", "extra1": "foo"}
    metadata2 = {"source": "b", "foo": "3", "bar": 2, "extra2": "bar"}
    docs = [
        Document(page_content="foo", metadata=metadata1),
        Document(page_content="bar", metadata=metadata2),
    ]
    output = collapse_docs(docs, _fake_combine_docs_func)
    expected_metadata = {
        "source": "a, b",
        "foo": "2, 3",
        "bar": "1, 2",
        "extra1": "foo",
        "extra2": "bar",
    }
    expected_output = Document(page_content="foobar", metadata=expected_metadata)
    assert output == expected_output


def test_format_doc_with_metadata() -> None:
    """Test format doc on a valid document."""
    doc = Document(page_content="foo", metadata={"bar": "baz"})
    prompt = PromptTemplate(
        input_variables=["page_content", "bar"], template="{page_content}, {bar}"
    )
    expected_output = "foo, baz"
    output = format_document(doc, prompt)
    assert output == expected_output


def test_format_doc_missing_metadata() -> None:
    """Test format doc on a document with missing metadata."""
    doc = Document(page_content="foo")
    prompt = PromptTemplate(
        input_variables=["page_content", "bar"], template="{page_content}, {bar}"
    )
    with pytest.raises(ValueError):
        format_document(doc, prompt)
