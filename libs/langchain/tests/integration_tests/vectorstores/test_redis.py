"""Test Redis functionality."""
import os
from typing import Any, Dict, List, Optional

import pytest

from langchain.docstore.document import Document
from langchain.vectorstores.redis import (
    Redis,
    RedisFilter,
    RedisNum,
    RedisText,
)
from langchain.vectorstores.redis.filters import RedisFilterExpression
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
    FakeEmbeddings,
)

TEST_INDEX_NAME = "test"
TEST_REDIS_URL = "redis://localhost:6379"
TEST_SINGLE_RESULT = [Document(page_content="foo")]
TEST_SINGLE_WITH_METADATA = {"a": "b"}
TEST_RESULT = [Document(page_content="foo"), Document(page_content="foo")]
RANGE_SCORE = pytest.approx(0.0513, abs=0.002)
COSINE_SCORE = pytest.approx(0.05, abs=0.002)
IP_SCORE = -8.0
EUCLIDEAN_SCORE = 1.0


def drop(index_name: str) -> bool:
    return Redis.drop_index(
        index_name=index_name, delete_documents=True, redis_url=TEST_REDIS_URL
    )


def convert_bytes(data: Any) -> Any:
    if isinstance(data, bytes):
        return data.decode("ascii")
    if isinstance(data, dict):
        return dict(map(convert_bytes, data.items()))
    if isinstance(data, list):
        return list(map(convert_bytes, data))
    if isinstance(data, tuple):
        return map(convert_bytes, data)
    return data


def make_dict(values: List[Any]) -> dict:
    i = 0
    di = {}
    while i < len(values) - 1:
        di[values[i]] = values[i + 1]
        i += 2
    return di


@pytest.fixture
def texts() -> List[str]:
    return ["foo", "bar", "baz"]


def test_redis(texts: List[str]) -> None:
    """Test end to end construction and search."""
    docsearch = Redis.from_texts(texts, FakeEmbeddings(), redis_url=TEST_REDIS_URL)
    output = docsearch.similarity_search("foo", k=1, return_metadata=False)
    assert output == TEST_SINGLE_RESULT
    assert drop(docsearch.index_name)


def test_redis_new_vector(texts: List[str]) -> None:
    """Test adding a new document"""
    docsearch = Redis.from_texts(texts, FakeEmbeddings(), redis_url=TEST_REDIS_URL)
    docsearch.add_texts(["foo"])
    output = docsearch.similarity_search("foo", k=2, return_metadata=False)
    assert output == TEST_RESULT
    assert drop(docsearch.index_name)


def test_redis_from_existing(texts: List[str]) -> None:
    """Test adding a new document"""
    docsearch = Redis.from_texts(
        texts, FakeEmbeddings(), index_name=TEST_INDEX_NAME, redis_url=TEST_REDIS_URL
    )
    schema: Dict = docsearch.schema

    # write schema for the next test
    docsearch.write_schema("test_schema.yml")

    # Test creating from an existing
    docsearch2 = Redis.from_existing_index(
        FakeEmbeddings(),
        index_name=TEST_INDEX_NAME,
        redis_url=TEST_REDIS_URL,
        schema=schema,
    )
    output = docsearch2.similarity_search("foo", k=1, return_metadata=False)
    assert output == TEST_SINGLE_RESULT


def test_redis_add_texts_to_existing() -> None:
    """Test adding a new document"""
    # Test creating from an existing with yaml from file
    docsearch = Redis.from_existing_index(
        FakeEmbeddings(),
        index_name=TEST_INDEX_NAME,
        redis_url=TEST_REDIS_URL,
        schema="test_schema.yml",
    )
    docsearch.add_texts(["foo"])
    output = docsearch.similarity_search("foo", k=2, return_metadata=False)
    assert output == TEST_RESULT
    assert drop(TEST_INDEX_NAME)
    # remove the test_schema.yml file
    os.remove("test_schema.yml")


def test_redis_from_texts_return_keys(texts: List[str]) -> None:
    """Test from_texts_return_keys constructor."""
    docsearch, keys = Redis.from_texts_return_keys(
        texts, FakeEmbeddings(), redis_url=TEST_REDIS_URL
    )
    output = docsearch.similarity_search("foo", k=1, return_metadata=False)
    assert output == TEST_SINGLE_RESULT
    assert len(keys) == len(texts)
    assert drop(docsearch.index_name)


def test_redis_from_documents(texts: List[str]) -> None:
    """Test from_documents constructor."""
    docs = [Document(page_content=t, metadata={"a": "b"}) for t in texts]
    docsearch = Redis.from_documents(docs, FakeEmbeddings(), redis_url=TEST_REDIS_URL)
    output = docsearch.similarity_search("foo", k=1, return_metadata=True)
    assert "a" in output[0].metadata.keys()
    assert "b" in output[0].metadata.values()
    assert drop(docsearch.index_name)


# -- test filters -- #


@pytest.mark.parametrize(
    "filter_expr, expected_length, expected_nums",
    [
        (RedisText("text") == "foo", 1, None),
        (RedisFilter.text("text") == "foo", 1, None),
        (RedisText("text") % "ba*", 2, ["bar", "baz"]),
        (RedisNum("num") > 2, 1, [3]),
        (RedisNum("num") < 2, 1, [1]),
        (RedisNum("num") >= 2, 2, [2, 3]),
        (RedisNum("num") <= 2, 2, [1, 2]),
        (RedisNum("num") != 2, 2, [1, 3]),
        (RedisFilter.num("num") != 2, 2, [1, 3]),
        (RedisFilter.tag("category") == "a", 3, None),
        (RedisFilter.tag("category") == "b", 2, None),
        (RedisFilter.tag("category") == "c", 2, None),
        (RedisFilter.tag("category") == ["b", "c"], 3, None),
    ],
    ids=[
        "text-filter-equals-foo",
        "alternative-text-equals-foo",
        "text-filter-fuzzy-match-ba",
        "number-filter-greater-than-2",
        "number-filter-less-than-2",
        "number-filter-greater-equals-2",
        "number-filter-less-equals-2",
        "number-filter-not-equals-2",
        "alternative-number-not-equals-2",
        "tag-filter-equals-a",
        "tag-filter-equals-b",
        "tag-filter-equals-c",
        "tag-filter-equals-b-or-c",
    ],
)
def test_redis_filters_1(
    filter_expr: RedisFilterExpression,
    expected_length: int,
    expected_nums: Optional[list],
) -> None:
    metadata = [
        {"name": "joe", "num": 1, "text": "foo", "category": ["a", "b"]},
        {"name": "john", "num": 2, "text": "bar", "category": ["a", "c"]},
        {"name": "jane", "num": 3, "text": "baz", "category": ["b", "c", "a"]},
    ]
    documents = [Document(page_content="foo", metadata=m) for m in metadata]
    docsearch = Redis.from_documents(
        documents, FakeEmbeddings(), redis_url=TEST_REDIS_URL
    )

    output = docsearch.similarity_search("foo", k=3, filter=filter_expr)

    assert len(output) == expected_length

    if expected_nums is not None:
        for out in output:
            assert (
                out.metadata["text"] in expected_nums
                or int(out.metadata["num"]) in expected_nums
            )

    assert drop(docsearch.index_name)


# -- test index specification -- #


def test_index_specification_generation() -> None:
    index_schema = {
        "text": [{"name": "job"}, {"name": "title"}],
        "numeric": [{"name": "salary"}],
    }

    text = ["foo"]
    meta = {"job": "engineer", "title": "principal engineer", "salary": 100000}
    docs = [Document(page_content=t, metadata=meta) for t in text]
    r = Redis.from_documents(
        docs, FakeEmbeddings(), redis_url=TEST_REDIS_URL, index_schema=index_schema
    )

    output = r.similarity_search("foo", k=1, return_metadata=True)
    assert output[0].metadata["job"] == "engineer"
    assert output[0].metadata["title"] == "principal engineer"
    assert int(output[0].metadata["salary"]) == 100000

    info = convert_bytes(r.client.ft(r.index_name).info())
    attributes = info["attributes"]
    assert len(attributes) == 5
    for attr in attributes:
        d = make_dict(attr)
        if d["identifier"] == "job":
            assert d["type"] == "TEXT"
        elif d["identifier"] == "title":
            assert d["type"] == "TEXT"
        elif d["identifier"] == "salary":
            assert d["type"] == "NUMERIC"
        elif d["identifier"] == "content":
            assert d["type"] == "TEXT"
        elif d["identifier"] == "content_vector":
            assert d["type"] == "VECTOR"
        else:
            raise ValueError("Unexpected attribute in index schema")

    assert drop(r.index_name)


# -- test distance metrics -- #

cosine_schema: Dict = {"distance_metric": "cosine"}
ip_schema: Dict = {"distance_metric": "IP"}
l2_schema: Dict = {"distance_metric": "L2"}


def test_cosine(texts: List[str]) -> None:
    """Test cosine distance."""
    docsearch = Redis.from_texts(
        texts,
        FakeEmbeddings(),
        redis_url=TEST_REDIS_URL,
        vector_schema=cosine_schema,
    )
    output = docsearch.similarity_search_with_score("far", k=2)
    _, score = output[1]
    assert score == COSINE_SCORE
    assert drop(docsearch.index_name)


def test_l2(texts: List[str]) -> None:
    """Test Flat L2 distance."""
    docsearch = Redis.from_texts(
        texts, FakeEmbeddings(), redis_url=TEST_REDIS_URL, vector_schema=l2_schema
    )
    output = docsearch.similarity_search_with_score("far", k=2)
    _, score = output[1]
    assert score == EUCLIDEAN_SCORE
    assert drop(docsearch.index_name)


def test_ip(texts: List[str]) -> None:
    """Test inner product distance."""
    docsearch = Redis.from_texts(
        texts, FakeEmbeddings(), redis_url=TEST_REDIS_URL, vector_schema=ip_schema
    )
    output = docsearch.similarity_search_with_score("far", k=2)
    _, score = output[1]
    assert score == IP_SCORE
    assert drop(docsearch.index_name)


def test_similarity_search_limit_distance(texts: List[str]) -> None:
    """Test similarity search limit score."""
    docsearch = Redis.from_texts(
        texts,
        FakeEmbeddings(),
        redis_url=TEST_REDIS_URL,
    )
    output = docsearch.similarity_search(texts[0], k=3, distance_threshold=0.1)

    # can't check score but length of output should be 2
    assert len(output) == 2
    assert drop(docsearch.index_name)


def test_similarity_search_with_score_with_limit_distance(texts: List[str]) -> None:
    """Test similarity search with score with limit score."""

    docsearch = Redis.from_texts(
        texts, ConsistentFakeEmbeddings(), redis_url=TEST_REDIS_URL
    )
    output = docsearch.similarity_search_with_score(
        texts[0], k=3, distance_threshold=0.1, return_metadata=True
    )

    assert len(output) == 2
    for out, score in output:
        if out.page_content == texts[1]:
            score == COSINE_SCORE
    assert drop(docsearch.index_name)


def test_delete(texts: List[str]) -> None:
    """Test deleting a new document"""
    docsearch = Redis.from_texts(texts, FakeEmbeddings(), redis_url=TEST_REDIS_URL)
    ids = docsearch.add_texts(["foo"])
    got = docsearch.delete(ids=ids, redis_url=TEST_REDIS_URL)
    assert got
    assert drop(docsearch.index_name)


def test_redis_as_retriever() -> None:
    texts = ["foo", "foo", "foo", "foo", "bar"]
    docsearch = Redis.from_texts(
        texts, ConsistentFakeEmbeddings(), redis_url=TEST_REDIS_URL
    )

    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    results = retriever.get_relevant_documents("foo")
    assert len(results) == 3
    assert all([d.page_content == "foo" for d in results])

    assert drop(docsearch.index_name)


def test_redis_retriever_distance_threshold() -> None:
    texts = ["foo", "bar", "baz"]
    docsearch = Redis.from_texts(texts, FakeEmbeddings(), redis_url=TEST_REDIS_URL)

    retriever = docsearch.as_retriever(
        search_type="similarity_distance_threshold",
        search_kwargs={"k": 3, "distance_threshold": 0.1},
    )
    results = retriever.get_relevant_documents("foo")
    assert len(results) == 2

    assert drop(docsearch.index_name)


def test_redis_retriever_score_threshold() -> None:
    texts = ["foo", "bar", "baz"]
    docsearch = Redis.from_texts(texts, FakeEmbeddings(), redis_url=TEST_REDIS_URL)

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.91},
    )
    results = retriever.get_relevant_documents("foo")
    assert len(results) == 2

    assert drop(docsearch.index_name)
