"""Test Upstash Vector functionality."""

import os
from time import sleep
from typing import List, Tuple

# to fix the following error in test with vcr and asyncio
#
# RuntimeError: asyncio.run() cannot be called from a running event loop
import pytest
from langchain_core.documents import Document

from langchain_community.vectorstores.upstash import UpstashVectorStore
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
)


@pytest.fixture(scope="module")
def vcr_cassette_dir() -> str:
    # save under cassettes/test_upstash/{item}.yaml
    return os.path.join("cassettes", "test_upstash")


@pytest.fixture(scope="function", autouse=True)
def fixture() -> None:
    store = UpstashVectorStore()
    embedding_store = UpstashVectorStore(
        index_url=os.environ["UPSTASH_VECTOR_URL_EMBEDDING"],
        index_token=os.environ["UPSTASH_VECTOR_TOKEN_EMBEDDING"],
    )

    store.delete(delete_all=True)
    embedding_store.delete(delete_all=True)

    wait_for_indexing(store)
    wait_for_indexing(embedding_store)


def wait_for_indexing(store: UpstashVectorStore) -> None:
    while store.info().pending_vector_count != 0:
        # Wait for indexing to complete
        sleep(0.5)


def check_response_with_score(
    result: List[Tuple[Document, float]],
    expected: List[Tuple[Document, float]],
) -> None:
    """
    check the result of a search with scores with an expected value

    scores in result will be rounded by two digits
    """
    result = list(map(lambda result: (result[0], round(result[1], 2)), result))

    assert result == expected


@pytest.mark.vcr()
def test_upstash_simple_insert() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    store = UpstashVectorStore.from_texts(texts=texts, embedding=FakeEmbeddings())
    wait_for_indexing(store)
    output = store.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_upstash_simple_insert_async() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    store = UpstashVectorStore.from_texts(texts=texts, embedding=FakeEmbeddings())
    wait_for_indexing(store)
    output = await store.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.vcr()
def test_upstash_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    store = UpstashVectorStore.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    wait_for_indexing(store)
    output = store.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_upstash_with_metadatas_async() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    store = UpstashVectorStore.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    wait_for_indexing(store)
    output = await store.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


@pytest.mark.vcr()
def test_upstash_with_metadatas_with_scores() -> None:
    """Test end to end construction and scored search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    store = UpstashVectorStore.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    wait_for_indexing(store)
    output = store.similarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_upstash_with_metadatas_with_scores_async() -> None:
    """Test end to end construction and scored search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    store = UpstashVectorStore.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
    )
    wait_for_indexing(store)
    output = await store.asimilarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]


@pytest.mark.vcr()
def test_upstash_with_metadatas_with_scores_using_vector() -> None:
    """Test end to end construction and scored search, using embedding vector."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    embeddings = FakeEmbeddings()

    store = UpstashVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )
    wait_for_indexing(store)
    embedded_query = embeddings.embed_query("foo")
    output = store.similarity_search_by_vector_with_score(embedding=embedded_query, k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_upstash_with_metadatas_with_scores_using_vector_async() -> None:
    """Test end to end construction and scored search, using embedding vector."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    embeddings = FakeEmbeddings()

    store = UpstashVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )
    wait_for_indexing(store)
    embedded_query = embeddings.embed_query("foo")
    output = await store.asimilarity_search_by_vector_with_score(
        embedding=embedded_query, k=1
    )
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]


@pytest.mark.vcr()
def test_upstash_mmr() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    store = UpstashVectorStore.from_texts(texts=texts, embedding=FakeEmbeddings())
    wait_for_indexing(store)
    output = store.max_marginal_relevance_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_upstash_mmr_async() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    store = UpstashVectorStore.from_texts(texts=texts, embedding=FakeEmbeddings())
    wait_for_indexing(store)
    output = await store.amax_marginal_relevance_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.vcr()
def test_upstash_mmr_by_vector() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    embeddings = FakeEmbeddings()
    store = UpstashVectorStore.from_texts(texts=texts, embedding=embeddings)
    wait_for_indexing(store)
    embedded_query = embeddings.embed_query("foo")
    output = store.max_marginal_relevance_search_by_vector(embedded_query, k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_upstash_mmr_by_vector_async() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    embeddings = FakeEmbeddings()
    store = UpstashVectorStore.from_texts(texts=texts, embedding=embeddings)
    wait_for_indexing(store)
    embedded_query = embeddings.embed_query("foo")
    output = await store.amax_marginal_relevance_search_by_vector(embedded_query, k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.vcr()
def test_init_from_index() -> None:
    from upstash_vector import Index

    index = Index.from_env()

    store = UpstashVectorStore(index=index)

    assert store.info() is not None


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_init_from_async_index() -> None:
    from upstash_vector import AsyncIndex

    index = AsyncIndex.from_env()

    store = UpstashVectorStore(async_index=index)

    assert await store.ainfo() is not None


@pytest.mark.vcr()
def test_init_from_credentials() -> None:
    store = UpstashVectorStore(
        index_url=os.environ["UPSTASH_VECTOR_REST_URL"],
        index_token=os.environ["UPSTASH_VECTOR_REST_TOKEN"],
    )

    assert store.info() is not None


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_init_from_credentials_async() -> None:
    store = UpstashVectorStore(
        index_url=os.environ["UPSTASH_VECTOR_REST_URL"],
        index_token=os.environ["UPSTASH_VECTOR_REST_TOKEN"],
    )

    assert await store.ainfo() is not None


@pytest.mark.vcr()
def test_upstash_add_documents_no_metadata() -> None:
    store = UpstashVectorStore(embedding=FakeEmbeddings())
    store.add_documents([Document(page_content="foo")])
    wait_for_indexing(store)

    search = store.similarity_search("foo")
    assert search == [Document(page_content="foo")]


@pytest.mark.vcr()
def test_upstash_add_documents_mixed_metadata() -> None:
    store = UpstashVectorStore(embedding=FakeEmbeddings())
    docs = [
        Document(page_content="foo"),
        Document(page_content="bar", metadata={"baz": 1}),
    ]
    ids = ["0", "1"]
    actual_ids = store.add_documents(docs, ids=ids)
    wait_for_indexing(store)
    assert actual_ids == ids
    search = store.similarity_search("foo bar")
    assert sorted(search, key=lambda d: d.page_content) == sorted(
        docs, key=lambda d: d.page_content
    )


@pytest.mark.vcr()
def test_upstash_similarity_search_with_metadata() -> None:
    store = UpstashVectorStore(embedding=FakeEmbeddings())
    docs = [
        Document(page_content="foo"),
        Document(page_content="bar", metadata={"waldo": 1}),
        Document(page_content="baz", metadata={"waldo": 1}),
        Document(page_content="fred", metadata={"waldo": 2}),
    ]
    ids = ["0", "1", "3", "4"]
    store.add_documents(docs, ids=ids)
    wait_for_indexing(store)

    result = store.similarity_search(query="foo", k=5, filter="waldo = 1")

    assert result == [
        Document(page_content="bar", metadata={"waldo": 1}),
        Document(page_content="baz", metadata={"waldo": 1}),
    ]

    search_result = store.similarity_search_with_score(
        query="foo", k=5, filter="waldo = 2"
    )

    check_response_with_score(
        search_result, [(Document(page_content="fred", metadata={"waldo": 2}), 0.85)]
    )


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_upstash_similarity_search_with_metadata_async() -> None:
    store = UpstashVectorStore(embedding=FakeEmbeddings())
    docs = [
        Document(page_content="foo"),
        Document(page_content="bar", metadata={"waldo": 1}),
        Document(page_content="baz", metadata={"waldo": 1}),
        Document(page_content="fred", metadata={"waldo": 2}),
    ]
    ids = ["0", "1", "3", "4"]
    store.add_documents(docs, ids=ids)
    wait_for_indexing(store)

    result = await store.asimilarity_search(query="foo", k=5, filter="waldo = 1")

    assert result == [
        Document(page_content="bar", metadata={"waldo": 1}),
        Document(page_content="baz", metadata={"waldo": 1}),
    ]

    search_result = await store.asimilarity_search_with_score(
        query="foo", k=5, filter="waldo = 2"
    )

    check_response_with_score(
        search_result, [(Document(page_content="fred", metadata={"waldo": 2}), 0.85)]
    )


@pytest.mark.vcr()
def test_upstash_similarity_search_by_vector_with_metadata() -> None:
    store = UpstashVectorStore(embedding=FakeEmbeddings())
    docs = [
        Document(page_content="foo"),
        Document(page_content="bar", metadata={"waldo": 1}),
        Document(page_content="baz", metadata={"waldo": 1}),
        Document(page_content="fred", metadata={"waldo": 2}),
    ]
    ids = ["0", "1", "3", "4"]
    store.add_documents(docs, ids=ids)
    wait_for_indexing(store)

    result = store.similarity_search_by_vector_with_score(
        embedding=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        k=5,
        filter="waldo = 1",
    )

    check_response_with_score(
        result,
        [
            (Document(page_content="bar", metadata={"waldo": 1}), 1.0),
            (Document(page_content="baz", metadata={"waldo": 1}), 0.98),
        ],
    )


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_upstash_similarity_search_by_vector_with_metadata_async() -> None:
    store = UpstashVectorStore(embedding=FakeEmbeddings())
    docs = [
        Document(page_content="foo"),
        Document(page_content="bar", metadata={"waldo": 1}),
        Document(page_content="baz", metadata={"waldo": 1}),
        Document(page_content="fred", metadata={"waldo": 2}),
    ]
    ids = ["0", "1", "3", "4"]
    store.add_documents(docs, ids=ids)
    wait_for_indexing(store)

    result = await store.asimilarity_search_by_vector_with_score(
        embedding=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        k=5,
        filter="waldo = 1",
    )

    check_response_with_score(
        result,
        [
            (Document(page_content="bar", metadata={"waldo": 1}), 1.0),
            (Document(page_content="baz", metadata={"waldo": 1}), 0.98),
        ],
    )


@pytest.mark.vcr()
def test_upstash_max_marginal_relevance_search_with_metadata() -> None:
    store = UpstashVectorStore(embedding=FakeEmbeddings())
    docs = [
        Document(page_content="foo"),
        Document(page_content="bar", metadata={"waldo": 1}),
        Document(page_content="baz", metadata={"waldo": 1}),
        Document(page_content="fred", metadata={"waldo": 2}),
    ]
    ids = ["0", "1", "3", "4"]
    store.add_documents(docs, ids=ids)
    wait_for_indexing(store)

    result = store.max_marginal_relevance_search(query="foo", k=3, filter="waldo = 1")

    assert result == [
        Document(page_content="bar", metadata={"waldo": 1}),
        Document(page_content="baz", metadata={"waldo": 1}),
    ]


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_upstash_max_marginal_relevance_search_with_metadata_async() -> None:
    store = UpstashVectorStore(embedding=FakeEmbeddings())
    docs = [
        Document(page_content="foo"),
        Document(page_content="bar", metadata={"waldo": 1}),
        Document(page_content="baz", metadata={"waldo": 1}),
        Document(page_content="fred", metadata={"waldo": 2}),
    ]
    ids = ["0", "1", "3", "4"]
    store.add_documents(docs, ids=ids)
    wait_for_indexing(store)

    result = await store.amax_marginal_relevance_search(
        query="foo", k=3, filter="waldo = 1"
    )

    assert result == [
        Document(page_content="bar", metadata={"waldo": 1}),
        Document(page_content="baz", metadata={"waldo": 1}),
    ]


@pytest.mark.vcr()
def test_embeddings_configurations() -> None:
    """
    test the behavior of the vector store for different `embeddings` parameter
    values
    """
    # case 1: use FakeEmbeddings, a subclass of Embeddings
    store = UpstashVectorStore(embedding=FakeEmbeddings())
    query_embedding = store._embed_query("query")
    assert query_embedding == [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

    document_embedding = store._embed_documents(["doc1", "doc2"])
    assert document_embedding == [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]

    # case 2: pass False as embedding
    store = UpstashVectorStore(embedding=False)
    with pytest.raises(ValueError):
        query_embedding = store._embed_query("query")

    with pytest.raises(ValueError):
        document_embedding = store._embed_documents(["doc1", "doc2"])

    # case 3: pass True as embedding
    # Upstash embeddings will be used
    store = UpstashVectorStore(
        index_url=os.environ["UPSTASH_VECTOR_URL_EMBEDDING"],
        index_token=os.environ["UPSTASH_VECTOR_TOKEN_EMBEDDING"],
        embedding=True,
    )
    query_embedding = store._embed_query("query")
    assert query_embedding == "query"
    document_embedding = store._embed_documents(["doc1", "doc2"])
    assert document_embedding == ["doc1", "doc2"]


@pytest.mark.vcr()
def test_embedding_index() -> None:
    store = UpstashVectorStore(
        index_url=os.environ["UPSTASH_VECTOR_URL_EMBEDDING"],
        index_token=os.environ["UPSTASH_VECTOR_TOKEN_EMBEDDING"],
        embedding=True,
    )

    # add documents
    store.add_documents(
        [
            Document(page_content="penguin", metadata={"topic": "bird"}),
            Document(page_content="albatros", metadata={"topic": "bird"}),
            Document(page_content="beethoven", metadata={"topic": "composer"}),
            Document(page_content="rachmaninoff", metadata={"topic": "composer"}),
        ]
    )
    wait_for_indexing(store)

    # similarity search
    search_result = store.similarity_search_with_score(query="eagle", k=2)
    check_response_with_score(
        search_result,
        [
            (Document(page_content="penguin", metadata={"topic": "bird"}), 0.82),
            (Document(page_content="albatros", metadata={"topic": "bird"}), 0.78),
        ],
    )

    # similarity search with relevance score
    search_result = store.similarity_search_with_relevance_scores(query="mozart", k=2)
    check_response_with_score(
        search_result,
        [
            (Document(page_content="beethoven", metadata={"topic": "composer"}), 0.88),
            (
                Document(page_content="rachmaninoff", metadata={"topic": "composer"}),
                0.84,
            ),
        ],
    )


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_embedding_index_async() -> None:
    store = UpstashVectorStore(
        index_url=os.environ["UPSTASH_VECTOR_URL_EMBEDDING"],
        index_token=os.environ["UPSTASH_VECTOR_TOKEN_EMBEDDING"],
        embedding=True,
    )

    # add documents
    await store.aadd_documents(
        [
            Document(page_content="penguin", metadata={"topic": "bird"}),
            Document(page_content="albatros", metadata={"topic": "bird"}),
            Document(page_content="beethoven", metadata={"topic": "composer"}),
            Document(page_content="rachmaninoff", metadata={"topic": "composer"}),
        ]
    )
    wait_for_indexing(store)

    # similarity search
    search_result = await store.asimilarity_search_with_score(query="eagle", k=2)
    check_response_with_score(
        search_result,
        [
            (Document(page_content="penguin", metadata={"topic": "bird"}), 0.82),
            (Document(page_content="albatros", metadata={"topic": "bird"}), 0.78),
        ],
    )

    # similarity search with relevance score
    search_result = await store.asimilarity_search_with_relevance_scores(
        query="mozart", k=2
    )
    check_response_with_score(
        search_result,
        [
            (Document(page_content="beethoven", metadata={"topic": "composer"}), 0.88),
            (
                Document(page_content="rachmaninoff", metadata={"topic": "composer"}),
                0.84,
            ),
        ],
    )
