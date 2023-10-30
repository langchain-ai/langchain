import time
from typing import List

from langchain.schema import Document
from langchain.vectorstores.alibabacloud_opensearch import (
    AlibabaCloudOpenSearch,
    AlibabaCloudOpenSearchSettings,
)
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

OS_TOKEN_COUNT = 1536

texts = ["foo", "bar", "baz"]


class FakeEmbeddingsWithOsDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, embedding_texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (OS_TOKEN_COUNT - 1) + [float(i)]
            for i in range(len(embedding_texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (OS_TOKEN_COUNT - 1) + [float(texts.index(text))]


"""
settings = AlibabaCloudOpenSearchSettings(
    endpoint="The endpoint of opensearch instance, If you want to access through
     the public network, you need to enable public network access in the network
     information of the instance details. If you want to access within 
     the Alibaba Cloud VPC, you can directly use the API domain name.",
    instance_id="The identify of opensearch instance",
    protocol (str): "Communication Protocol between SDK and Server, default is http.",
    username="The username specified when purchasing the instance.",
    password="The password specified when purchasing the instance.",
    namespace (str) : "The instance data will be partitioned based on the 
     namespace field, If the namespace is enabled, you need to specify the 
     namespace field name during  initialization. Otherwise, the queries cannot 
     be executed correctly, default is empty.",
    table_name="The table name is specified when adding a table after completing 
     the instance configuration.",
    field_name_mapping={
        # insert data into opensearch based on the mapping name of the field.
        "id": "The id field name map of index document.",
        "document": "The text field name map of index document.",
        "embedding": "The embedding field name map of index document，"
        "the values must be in float16 multivalue type "
        "and separated by commas.",
        "metadata_x": "The metadata field name map of index document, "
        "could specify multiple, The value field contains "
        "mapping name and operator, the operator would be "
        "used when executing metadata filter query",
    },
)
"""

settings = AlibabaCloudOpenSearchSettings(
    endpoint="ha-cn-5yd3fhdm102.public.ha.aliyuncs.com",
    instance_id="ha-cn-5yd3fhdm102",
    username="instance user name",
    password="instance password",
    table_name="instance table name",
    field_name_mapping={
        # insert data into opensearch based on the mapping name of the field.
        "id": "id",
        "document": "document",
        "embedding": "embedding",
        "string_field": "string_filed,=",
        "int_field": "int_filed,=",
        "float_field": "float_field,=",
        "double_field": "double_field,=",
    },
)

embeddings = FakeEmbeddingsWithOsDimension()


def test_create_alibabacloud_opensearch() -> None:
    opensearch = create_alibabacloud_opensearch()
    time.sleep(1)
    output = opensearch.similarity_search("foo", k=10)
    assert len(output) == 3


def test_alibabacloud_opensearch_with_text_query() -> None:
    opensearch = create_alibabacloud_opensearch()
    output = opensearch.similarity_search(query="foo", k=1)
    assert output == [
        Document(
            page_content="foo",
            metadata={
                "string_field": "value1",
                "int_field": 1,
                "float_field": 1.0,
                "double_field": 2.0,
            },
        )
    ]

    output = opensearch.similarity_search(query="bar", k=1)
    assert output == [
        Document(
            page_content="bar",
            metadata={
                "string_field": "value2",
                "int_field": 2,
                "float_field": 3.0,
                "double_field": 4.0,
            },
        )
    ]

    output = opensearch.similarity_search(query="baz", k=1)
    assert output == [
        Document(
            page_content="baz",
            metadata={
                "string_field": "value3",
                "int_field": 3,
                "float_field": 5.0,
                "double_field": 6.0,
            },
        )
    ]


def test_alibabacloud_opensearch_with_vector_query() -> None:
    opensearch = create_alibabacloud_opensearch()
    output = opensearch.similarity_search_by_vector(embeddings.embed_query("foo"), k=1)
    assert output == [
        Document(
            page_content="foo",
            metadata={
                "string_field": "value1",
                "int_field": 1,
                "float_field": 1.0,
                "double_field": 2.0,
            },
        )
    ]

    output = opensearch.similarity_search_by_vector(embeddings.embed_query("bar"), k=1)
    assert output == [
        Document(
            page_content="bar",
            metadata={
                "string_field": "value2",
                "int_field": 2,
                "float_field": 3.0,
                "double_field": 4.0,
            },
        )
    ]

    output = opensearch.similarity_search_by_vector(embeddings.embed_query("baz"), k=1)
    assert output == [
        Document(
            page_content="baz",
            metadata={
                "string_field": "value3",
                "int_field": 3,
                "float_field": 5.0,
                "double_field": 6.0,
            },
        )
    ]


def test_alibabacloud_opensearch_with_text_and_meta_query() -> None:
    opensearch = create_alibabacloud_opensearch()
    output = opensearch.similarity_search(
        query="foo", search_filter={"string_field": "value1"}, k=1
    )
    assert output == [
        Document(
            page_content="foo",
            metadata={
                "string_field": "value1",
                "int_field": 1,
                "float_field": 1.0,
                "double_field": 2.0,
            },
        )
    ]

    output = opensearch.similarity_search(
        query="bar", search_filter={"int_field": 2}, k=1
    )
    assert output == [
        Document(
            page_content="bar",
            metadata={
                "string_field": "value2",
                "int_field": 2,
                "float_field": 3.0,
                "double_field": 4.0,
            },
        )
    ]

    output = opensearch.similarity_search(
        query="baz", search_filter={"float_field": 5.0}, k=1
    )
    assert output == [
        Document(
            page_content="baz",
            metadata={
                "string_field": "value3",
                "int_field": 3,
                "float_field": 5.0,
                "double_field": 6.0,
            },
        )
    ]

    output = opensearch.similarity_search(
        query="baz", search_filter={"float_field": 6.0}, k=1
    )
    assert len(output) == 0


def test_alibabacloud_opensearch_with_text_and_meta_score_query() -> None:
    opensearch = create_alibabacloud_opensearch()
    output = opensearch.similarity_search_with_relevance_scores(
        query="foo",
        search_filter={
            "string_field": "value1",
            "int_field": 1,
            "float_field": 1.0,
            "double_field": 2.0,
        },
        k=1,
    )
    assert output == [
        (
            Document(
                page_content="foo",
                metadata={
                    "string_field": "value1",
                    "int_field": 1,
                    "float_field": 1.0,
                    "double_field": 2.0,
                },
            ),
            0.0,
        )
    ]


def test_alibabacloud_opensearch_delete_doc() -> None:
    opensearch = create_alibabacloud_opensearch()
    delete_result = opensearch.delete_documents_with_texts(["bar"])
    assert delete_result
    time.sleep(1)
    search_result = opensearch.similarity_search(
        query="bar", search_filter={"int_field": 2}, k=1
    )
    assert len(search_result) == 0


def create_alibabacloud_opensearch() -> AlibabaCloudOpenSearch:
    metadatas = [
        {
            "string_field": "value1",
            "int_field": 1,
            "float_field": 1.0,
            "double_field": 2.0,
        },
        {
            "string_field": "value2",
            "int_field": 2,
            "float_field": 3.0,
            "double_field": 4.0,
        },
        {
            "string_field": "value3",
            "int_field": 3,
            "float_field": 5.0,
            "double_field": 6.0,
        },
    ]

    return AlibabaCloudOpenSearch.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        metadatas=metadatas,
        config=settings,
    )
