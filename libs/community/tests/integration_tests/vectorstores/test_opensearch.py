"""Test OpenSearch functionality."""

import pytest
from langchain_community.vectorstores.opensearch_vector_search import (
    HYBRID_SEARCH,
    PAINLESS_SCRIPTING_SEARCH,
    SCRIPT_SCORING_SEARCH,
    OpenSearchVectorSearch,
)
from langchain_core.documents import Document
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
    FakeEmbeddings,
)

DEFAULT_OPENSEARCH_URL = "http://localhost:9200"
texts = ["foo", "bar", "baz"]
ids = ["id_foo", "id_bar", "id_baz"]


def test_opensearch() -> None:
    """Test end to end indexing and search using Approximate Search."""
    docsearch = OpenSearchVectorSearch.from_texts(
        texts,
        FakeEmbeddings(),
        opensearch_url=DEFAULT_OPENSEARCH_URL,
        ids=ids,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", id="id_foo")]


def test_similarity_search_with_score() -> None:
    """Test similarity search with score using Approximate Search."""
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = OpenSearchVectorSearch.from_texts(
        texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        opensearch_url=DEFAULT_OPENSEARCH_URL,
        ids=ids,
    )
    output = docsearch.similarity_search_with_score("foo", k=2)
    assert output == [
        (Document(page_content="foo", metadata={"page": 0}, id="id_foo"), 1.0),
        (Document(page_content="bar", metadata={"page": 1}, id="id_bar"), 0.5),
    ]


def test_opensearch_with_custom_field_name() -> None:
    """Test indexing and search using custom vector field and text field name."""
    docsearch = OpenSearchVectorSearch.from_texts(
        texts,
        FakeEmbeddings(),
        opensearch_url=DEFAULT_OPENSEARCH_URL,
        vector_field="my_vector",
        text_field="custom_text",
        ids=ids,
    )
    output = docsearch.similarity_search(
        "foo", k=1, vector_field="my_vector", text_field="custom_text"
    )
    assert output == [Document(page_content="foo", id="id_foo")]

    text_input = ["test", "add", "text", "method"]
    OpenSearchVectorSearch.add_texts(
        docsearch,
        text_input,
        vector_field="my_vector",
        text_field="custom_text",
    )
    output = docsearch.similarity_search(
        "add", k=1, vector_field="my_vector", text_field="custom_text"
    )
    assert output == [Document(page_content="foo", id="id_foo")]


def test_configure_search_pipeline() -> None:
    """Test configure search pipeline functionality."""
    test_search_pipeline_name = "test_search_pipeline"
    keyword_weight = 0.7
    vector_weight = 0.3

    docsearch = OpenSearchVectorSearch.from_texts(
        texts, FakeEmbeddings(), opensearch_url=DEFAULT_OPENSEARCH_URL
    )
    docsearch.configure_search_pipelines(
        pipeline_name=test_search_pipeline_name,
        keyword_weight=keyword_weight,
        vector_weight=vector_weight,
    )
    assert docsearch.search_pipeline_exists(test_search_pipeline_name)


def test_get_search_pipeline_info() -> None:
    """Test get search pipeline info functionality."""
    test_search_pipeline_name = "test_search_pipeline"

    docsearch = OpenSearchVectorSearch.from_texts(
        texts, FakeEmbeddings(), opensearch_url=DEFAULT_OPENSEARCH_URL
    )
    test_pipeline_info = docsearch.get_search_pipeline_info(test_search_pipeline_name)
    assert test_pipeline_info == {
        "test_search_pipeline": {
            "description": "Post processor for hybrid search",
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {"weights": [0.7, 0.3]},
                        },
                    }
                }
            ],
        }
    }


def test_hybrid_search() -> None:
    """Test hybrid search functionality."""
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = OpenSearchVectorSearch.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        opensearch_url=DEFAULT_OPENSEARCH_URL,
    )
    output = docsearch.similarity_search(
        query="foo",
        k=2,
        search_type=HYBRID_SEARCH,
        search_pipeline="test_search_pipeline",
    )

    assert output == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="bar", metadata={"page": 1}),
    ]


def test_hybrid_search_with_score() -> None:
    """Test hybrid search with score functionality."""
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = OpenSearchVectorSearch.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        opensearch_url=DEFAULT_OPENSEARCH_URL,
    )
    output = docsearch.similarity_search_with_score(
        query="foo",
        k=2,
        search_type=HYBRID_SEARCH,
        search_pipeline="test_search_pipeline",
    )
    assert output == [
        (Document(page_content="foo", metadata={"page": 0}), 1.0),
        (Document(page_content="bar", metadata={"page": 1}), 0.0003),
    ]


def test_hybrid_search_with_post_filter() -> None:
    """Test hybrid search with post filter functionality."""
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = OpenSearchVectorSearch.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        opensearch_url=DEFAULT_OPENSEARCH_URL,
    )
    output = docsearch.similarity_search(
        query="foo",
        k=2,
        search_type="hybrid_search",
        search_pipeline="test_search_pipeline",
        post_filter={"bool": {"filter": {"term": {"metadata.page": 1}}}},
    )

    assert output == [Document(page_content="bar", metadata={"page": 1})]


def test_opensearch_with_metadatas() -> None:
    """Test end to end indexing and search with metadata."""
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = OpenSearchVectorSearch.from_texts(
        texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        opensearch_url=DEFAULT_OPENSEARCH_URL,
        ids=ids,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0}, id="id_foo")]


def test_max_marginal_relevance_search() -> None:
    """Test end to end indexing and mmr search."""
    docsearch = OpenSearchVectorSearch.from_texts(
        texts,
        FakeEmbeddings(),
        opensearch_url=DEFAULT_OPENSEARCH_URL,
        ids=ids,
    )
    output = docsearch.max_marginal_relevance_search("foo", k=1)
    assert output == [Document(page_content="foo", id="id_foo")]


def test_add_text() -> None:
    """Test adding additional text elements to existing index."""
    text_input = ["test", "add", "text", "method"]
    metadatas = [{"page": i} for i in range(len(text_input))]
    docsearch = OpenSearchVectorSearch.from_texts(
        texts, FakeEmbeddings(), opensearch_url=DEFAULT_OPENSEARCH_URL
    )
    doc_ids = OpenSearchVectorSearch.add_texts(docsearch, text_input, metadatas)
    assert len(doc_ids) == len(text_input)


def test_add_embeddings() -> None:
    """
    Test add_embeddings, which accepts pre-built embeddings instead of
     using inference for the texts.
    This allows you to separate the embeddings text and the page_content
     for better proximity between user's question and embedded text.
    For example, your embedding text can be a question, whereas page_content
     is the answer.
    """
    embeddings = ConsistentFakeEmbeddings()
    text_input = ["foo1", "foo2", "foo3"]
    metadatas = [{"page": i} for i in range(len(text_input))]

    """In real use case, embedding_input can be questions for each text"""
    embedding_input = ["foo2", "foo3", "foo1"]
    embedding_vectors = embeddings.embed_documents(embedding_input)

    docsearch = OpenSearchVectorSearch.from_texts(
        ["filler"], embeddings, opensearch_url=DEFAULT_OPENSEARCH_URL
    )
    docsearch.add_embeddings(list(zip(text_input, embedding_vectors)), metadatas)
    output = docsearch.similarity_search("foo1", k=1)
    assert output[0].page_content == "foo3"
    assert output[0].metadata == {"page": 2}


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
    assert output[0].page_content == "bar"
    assert output[0].id is not None


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
    assert output[0].page_content == "test"
    assert output[0].id is not None


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
    assert output[0].page_content == "baz"
    assert output[0].id is not None


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
    assert output[0].page_content == "test"
    assert output[0].id is not None


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
    assert output[0].page_content == "bar"
    assert output[0].id is not None


def test_appx_search_with_lucene_filter() -> None:
    """Test Approximate Search with Lucene Filter."""
    lucene_filter_val = {"bool": {"must": [{"term": {"text": "bar"}}]}}
    docsearch = OpenSearchVectorSearch.from_texts(
        texts, FakeEmbeddings(), opensearch_url=DEFAULT_OPENSEARCH_URL, engine="lucene"
    )
    output = docsearch.similarity_search("foo", k=3, lucene_filter=lucene_filter_val)
    assert output[0].page_content == "bar"
    assert output[0].id is not None


def test_opensearch_with_custom_field_name_appx_true() -> None:
    """Test Approximate Search with custom field name appx true."""
    text_input = ["add", "test", "text", "method"]
    docsearch = OpenSearchVectorSearch.from_texts(
        text_input,
        FakeEmbeddings(),
        opensearch_url=DEFAULT_OPENSEARCH_URL,
        is_appx_search=True,
    )
    output = docsearch.similarity_search("add", k=1)
    assert output[0].page_content == "add"
    assert output[0].id is not None


def test_opensearch_with_custom_field_name_appx_false() -> None:
    """Test Approximate Search with custom field name appx true."""
    text_input = ["add", "test", "text", "method"]
    docsearch = OpenSearchVectorSearch.from_texts(
        text_input, FakeEmbeddings(), opensearch_url=DEFAULT_OPENSEARCH_URL
    )
    output = docsearch.similarity_search("add", k=1)
    assert output[0].page_content == "add"
    assert output[0].id is not None


def test_opensearch_serverless_with_scripting_search_indexing_throws_error() -> None:
    """Test to validate indexing using Serverless without Approximate Search."""
    import boto3
    from opensearchpy import AWSV4SignerAuth

    region = "test-region"
    service = "aoss"
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region, service)
    with pytest.raises(ValueError):
        OpenSearchVectorSearch.from_texts(
            texts,
            FakeEmbeddings(),
            opensearch_url=DEFAULT_OPENSEARCH_URL,
            is_appx_search=False,
            http_auth=auth,
        )


def test_opensearch_serverless_with_lucene_engine_throws_error() -> None:
    """Test to validate indexing using lucene engine with Serverless."""
    import boto3
    from opensearchpy import AWSV4SignerAuth

    region = "test-region"
    service = "aoss"
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region, service)
    with pytest.raises(ValueError):
        OpenSearchVectorSearch.from_texts(
            texts,
            FakeEmbeddings(),
            opensearch_url=DEFAULT_OPENSEARCH_URL,
            engine="lucene",
            http_auth=auth,
        )


def test_appx_search_with_efficient_and_bool_filter_throws_error() -> None:
    """Test Approximate Search with Efficient and Bool Filter throws Error."""
    efficient_filter_val = {"bool": {"must": [{"term": {"text": "baz"}}]}}
    boolean_filter_val = {"bool": {"must": [{"term": {"text": "bar"}}]}}
    docsearch = OpenSearchVectorSearch.from_texts(
        texts, FakeEmbeddings(), opensearch_url=DEFAULT_OPENSEARCH_URL, engine="lucene"
    )
    with pytest.raises(ValueError):
        docsearch.similarity_search(
            "foo",
            k=3,
            efficient_filter=efficient_filter_val,
            boolean_filter=boolean_filter_val,
        )


def test_appx_search_with_efficient_and_lucene_filter_throws_error() -> None:
    """Test Approximate Search with Efficient and Lucene Filter throws Error."""
    efficient_filter_val = {"bool": {"must": [{"term": {"text": "baz"}}]}}
    lucene_filter_val = {"bool": {"must": [{"term": {"text": "bar"}}]}}
    docsearch = OpenSearchVectorSearch.from_texts(
        texts, FakeEmbeddings(), opensearch_url=DEFAULT_OPENSEARCH_URL, engine="lucene"
    )
    with pytest.raises(ValueError):
        docsearch.similarity_search(
            "foo",
            k=3,
            efficient_filter=efficient_filter_val,
            lucene_filter=lucene_filter_val,
        )


def test_appx_search_with_boolean_and_lucene_filter_throws_error() -> None:
    """Test Approximate Search with Boolean and Lucene Filter throws Error."""
    boolean_filter_val = {"bool": {"must": [{"term": {"text": "baz"}}]}}
    lucene_filter_val = {"bool": {"must": [{"term": {"text": "bar"}}]}}
    docsearch = OpenSearchVectorSearch.from_texts(
        texts, FakeEmbeddings(), opensearch_url=DEFAULT_OPENSEARCH_URL, engine="lucene"
    )
    with pytest.raises(ValueError):
        docsearch.similarity_search(
            "foo",
            k=3,
            boolean_filter=boolean_filter_val,
            lucene_filter=lucene_filter_val,
        )


def test_appx_search_with_faiss_efficient_filter() -> None:
    """Test Approximate Search with Faiss Efficient Filter."""
    efficient_filter_val = {"bool": {"must": [{"term": {"text": "bar"}}]}}
    docsearch = OpenSearchVectorSearch.from_texts(
        texts, FakeEmbeddings(), opensearch_url=DEFAULT_OPENSEARCH_URL, engine="faiss"
    )
    output = docsearch.similarity_search(
        "foo", k=3, efficient_filter=efficient_filter_val
    )
    assert output[0].page_content == "bar"
    assert output[0].id is not None
