"""Test OpenSearch functionality."""

import pytest

from langchain.docstore.document import Document
from langchain.vectorstores.opensearch_vector_search import (
    PAINLESS_SCRIPTING_SEARCH,
    SCRIPT_SCORING_SEARCH,
    OpenSearchVectorSearch,
)
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

DEFAULT_OPENSEARCH_URL = "http://localhost:9200"
texts = ["foo", "bar", "baz"]


def test_opensearch() -> None:
    """Test end to end indexing and search using Approximate Search."""
    docsearch = OpenSearchVectorSearch.from_texts(
        texts, FakeEmbeddings(), opensearch_url=DEFAULT_OPENSEARCH_URL
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_opensearch_with_custom_field_name() -> None:
    """Test indexing and search using custom vector field and text field name."""
    docsearch = OpenSearchVectorSearch.from_texts(
        texts,
        FakeEmbeddings(),
        opensearch_url=DEFAULT_OPENSEARCH_URL,
        vector_field="my_vector",
        text_field="custom_text",
    )
    output = docsearch.similarity_search(
        "foo", k=1, vector_field="my_vector", text_field="custom_text"
    )
    assert output == [Document(page_content="foo")]

    text_input = ["test", "add", "text", "method"]
    OpenSearchVectorSearch.add_texts(
        docsearch, text_input, vector_field="my_vector", text_field="custom_text"
    )
    output = docsearch.similarity_search(
        "add", k=1, vector_field="my_vector", text_field="custom_text"
    )
    assert output == [Document(page_content="foo")]


def test_opensearch_with_metadatas() -> None:
    """Test end to end indexing and search with metadata."""
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = OpenSearchVectorSearch.from_texts(
        texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        opensearch_url=DEFAULT_OPENSEARCH_URL,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]


def test_add_text() -> None:
    """Test adding additional text elements to existing index."""
    text_input = ["test", "add", "text", "method"]
    metadatas = [{"page": i} for i in range(len(text_input))]
    docsearch = OpenSearchVectorSearch.from_texts(
        texts, FakeEmbeddings(), opensearch_url=DEFAULT_OPENSEARCH_URL
    )
    docids = OpenSearchVectorSearch.add_texts(docsearch, text_input, metadatas)
    assert len(docids) == len(text_input)


def test_opensearch_script_scoring() -> None:
    """Test end to end indexing and search using Script Scoring Search."""
    pre_filter_val = {"bool": {"filter": {"term": {"text": "bar"}}}}
    docsearch = OpenSearchVectorSearch.from_texts(
        texts,
        FakeEmbeddings(),
        opensearch_url=DEFAULT_OPENSEARCH_URL,
        is_appx_search=False,
    )
    output = docsearch.similarity_search(
        "foo", k=1, search_type=SCRIPT_SCORING_SEARCH, pre_filter=pre_filter_val
    )
    assert output == [Document(page_content="bar")]


def test_add_text_script_scoring() -> None:
    """Test adding additional text elements and validating using Script Scoring."""
    text_input = ["test", "add", "text", "method"]
    metadatas = [{"page": i} for i in range(len(text_input))]
    docsearch = OpenSearchVectorSearch.from_texts(
        text_input,
        FakeEmbeddings(),
        opensearch_url=DEFAULT_OPENSEARCH_URL,
        is_appx_search=False,
    )
    OpenSearchVectorSearch.add_texts(docsearch, texts, metadatas)
    output = docsearch.similarity_search(
        "add", k=1, search_type=SCRIPT_SCORING_SEARCH, space_type="innerproduct"
    )
    assert output == [Document(page_content="test")]


def test_opensearch_painless_scripting() -> None:
    """Test end to end indexing and search using Painless Scripting Search."""
    pre_filter_val = {"bool": {"filter": {"term": {"text": "baz"}}}}
    docsearch = OpenSearchVectorSearch.from_texts(
        texts,
        FakeEmbeddings(),
        opensearch_url=DEFAULT_OPENSEARCH_URL,
        is_appx_search=False,
    )
    output = docsearch.similarity_search(
        "foo", k=1, search_type=PAINLESS_SCRIPTING_SEARCH, pre_filter=pre_filter_val
    )
    assert output == [Document(page_content="baz")]


def test_add_text_painless_scripting() -> None:
    """Test adding additional text elements and validating using Painless Scripting."""
    text_input = ["test", "add", "text", "method"]
    metadatas = [{"page": i} for i in range(len(text_input))]
    docsearch = OpenSearchVectorSearch.from_texts(
        text_input,
        FakeEmbeddings(),
        opensearch_url=DEFAULT_OPENSEARCH_URL,
        is_appx_search=False,
    )
    OpenSearchVectorSearch.add_texts(docsearch, texts, metadatas)
    output = docsearch.similarity_search(
        "add", k=1, search_type=PAINLESS_SCRIPTING_SEARCH, space_type="cosineSimilarity"
    )
    assert output == [Document(page_content="test")]


def test_opensearch_invalid_search_type() -> None:
    """Test to validate similarity_search by providing invalid search_type."""
    docsearch = OpenSearchVectorSearch.from_texts(
        texts, FakeEmbeddings(), opensearch_url=DEFAULT_OPENSEARCH_URL
    )
    with pytest.raises(ValueError):
        docsearch.similarity_search("foo", k=1, search_type="invalid_search_type")


def test_opensearch_embedding_size_zero() -> None:
    """Test to validate indexing when embedding size is zero."""
    with pytest.raises(RuntimeError):
        OpenSearchVectorSearch.from_texts(
            [], FakeEmbeddings(), opensearch_url=DEFAULT_OPENSEARCH_URL
        )


def test_appx_search_with_boolean_filter() -> None:
    """Test Approximate Search with Boolean Filter."""
    boolean_filter_val = {"bool": {"must": [{"term": {"text": "bar"}}]}}
    docsearch = OpenSearchVectorSearch.from_texts(
        texts,
        FakeEmbeddings(),
        opensearch_url=DEFAULT_OPENSEARCH_URL,
    )
    output = docsearch.similarity_search(
        "foo", k=3, boolean_filter=boolean_filter_val, subquery_clause="should"
    )
    assert output == [Document(page_content="bar")]


def test_appx_search_with_lucene_filter() -> None:
    """Test Approximate Search with Lucene Filter."""
    lucene_filter_val = {"bool": {"must": [{"term": {"text": "bar"}}]}}
    docsearch = OpenSearchVectorSearch.from_texts(
        texts, FakeEmbeddings(), opensearch_url=DEFAULT_OPENSEARCH_URL, engine="lucene"
    )
    output = docsearch.similarity_search("foo", k=3, lucene_filter=lucene_filter_val)
    assert output == [Document(page_content="bar")]
