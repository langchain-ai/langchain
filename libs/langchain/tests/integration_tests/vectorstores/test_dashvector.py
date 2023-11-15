from time import sleep

from langchain.schema import Document
from langchain.vectorstores import DashVector
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

texts = ["foo", "bar", "baz"]
ids = ["1", "2", "3"]


def test_dashvector_from_texts() -> None:
    dashvector = DashVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        ids=ids,
    )

    # the vector insert operation is async by design, we wait here a bit for the
    # insertion to complete.
    sleep(0.5)
    output = dashvector.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_dashvector_with_text_with_metadatas() -> None:
    metadatas = [{"meta": i} for i in range(len(texts))]
    dashvector = DashVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        ids=ids,
    )

    # the vector insert operation is async by design, we wait here a bit for the
    # insertion to complete.
    sleep(0.5)
    output = dashvector.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"meta": 0})]


def test_dashvector_search_with_filter() -> None:
    metadatas = [{"meta": i} for i in range(len(texts))]
    dashvector = DashVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        metadatas=metadatas,
        ids=ids,
    )

    # the vector insert operation is async by design, we wait here a bit for the
    # insertion to complete.
    sleep(0.5)
    output = dashvector.similarity_search("foo", filter="meta=2")
    assert output == [Document(page_content="baz", metadata={"meta": 2})]


def test_dashvector_search_with_scores() -> None:
    dashvector = DashVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddings(),
        ids=ids,
    )

    # the vector insert operation is async by design, we wait here a bit for the
    # insertion to complete.
    sleep(0.5)
    output = dashvector.similarity_search_with_relevance_scores("foo")
    docs, scores = zip(*output)

    assert scores[0] < scores[1] < scores[2]
    assert list(docs) == [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]
