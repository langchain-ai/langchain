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


settings = AlibabaCloudOpenSearchSettings(
    endpoint="The endpoint of opensearch instance, "
    "You can find it from the console of Alibaba Cloud OpenSearch.",
    instance_id="The identify of opensearch instance, "
    "You can find it from the console of Alibaba Cloud OpenSearch.",
    datasource_name="The name of the data source specified when creating it.",
    username="The username specified when purchasing the instance.",
    password="The password specified when purchasing the instance.",
    embedding_index_name="The name of the vector attribute "
    "specified when configuring the instance attributes.",
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

embeddings = FakeEmbeddingsWithOsDimension()


def test_create_alibabacloud_opensearch() -> None:
    opensearch = create_alibabacloud_opensearch()
    output = opensearch.similarity_search("foo", k=10)
    assert len(output) == 3


def test_alibabacloud_opensearch_with_text_query() -> None:
    opensearch = create_alibabacloud_opensearch()
    output = opensearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"metadata": "0"})]

    output = opensearch.similarity_search("bar", k=1)
    assert output == [Document(page_content="bar", metadata={"metadata": "1"})]

    output = opensearch.similarity_search("baz", k=1)
    assert output == [Document(page_content="baz", metadata={"metadata": "2"})]


def test_alibabacloud_opensearch_with_vector_query() -> None:
    opensearch = create_alibabacloud_opensearch()
    output = opensearch.similarity_search_by_vector(embeddings.embed_query("foo"), k=1)
    assert output == [Document(page_content="foo", metadata={"metadata": "0"})]

    output = opensearch.similarity_search_by_vector(embeddings.embed_query("bar"), k=1)
    assert output == [Document(page_content="bar", metadata={"metadata": "1"})]

    output = opensearch.similarity_search_by_vector(embeddings.embed_query("baz"), k=1)
    assert output == [Document(page_content="baz", metadata={"metadata": "2"})]


def test_alibabacloud_opensearch_with_text_and_meta_query() -> None:
    opensearch = create_alibabacloud_opensearch()
    output = opensearch.similarity_search(
        query="foo", search_filter={"metadata": "0"}, k=1
    )
    assert output == [Document(page_content="foo", metadata={"metadata": "0"})]

    output = opensearch.similarity_search(
        query="bar", search_filter={"metadata": "1"}, k=1
    )
    assert output == [Document(page_content="bar", metadata={"metadata": "1"})]

    output = opensearch.similarity_search(
        query="baz", search_filter={"metadata": "2"}, k=1
    )
    assert output == [Document(page_content="baz", metadata={"metadata": "2"})]

    output = opensearch.similarity_search(
        query="baz", search_filter={"metadata": "3"}, k=1
    )
    assert len(output) == 0


def test_alibabacloud_opensearch_with_text_and_meta_score_query() -> None:
    opensearch = create_alibabacloud_opensearch()
    output = opensearch.similarity_search_with_relevance_scores(
        query="foo", search_filter={"metadata": "0"}, k=1
    )
    assert output == [
        (Document(page_content="foo", metadata={"metadata": "0"}), 10000.0)
    ]


def create_alibabacloud_opensearch() -> AlibabaCloudOpenSearch:
    metadatas = [{"metadata": str(i)} for i in range(len(texts))]

    return AlibabaCloudOpenSearch.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        metadatas=metadatas,
        config=settings,
    )
