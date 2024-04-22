import os
import shutil

import pytest
from langchain_core.documents import Document

from langchain_community.vectorstores.objectbox import ObjectBox
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def remove_test_dir(test_dir: str):
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


@pytest.fixture(autouse=True)
def auto_cleanup():
    remove_test_dir("data")
    yield  # run the test function
    remove_test_dir("data")


def test_objectbox_db_initialisation():
    ob = ObjectBox(embedding=FakeEmbeddings(), embedding_dimensions=10)
    folder_path = "data"

    assert os.path.exists(folder_path), f"Folder '{folder_path}' does not exist."

    filepath = os.path.join(folder_path, "data.mdb")
    assert os.path.isfile(filepath), f"File '{filename}' not found in '{folder_path}'"


def test_similarity_search():
    ob = ObjectBox(embedding=FakeEmbeddings(), embedding_dimensions=10)
    texts = ["foo", "bar", "baz"]
    ob.add_texts(texts=texts)

    query = ob.similarity_search("foo", k=1)
    assert len(query) == 1

    query = ob.similarity_search("foo", k=2)
    assert len(query) == 2

    query = ob.similarity_search("foo", k=3)
    assert len(query) == 3


def test_from_texts():
    texts = ["foo", "bar", "baz"]
    ob = ObjectBox.from_texts(
        embedding=FakeEmbeddings(), embedding_dimensions=10, texts=texts
    )

    # positive test
    query = ob.similarity_search("foo", k=2)
    assert len(query) == 2


def test_similarity_search_with_score():
    ob = ObjectBox(embedding=FakeEmbeddings(), embedding_dimensions=10)
    texts = ["foo", "bar", "baz"]
    ob.add_texts(texts=texts)

    query = ob.similarity_search_with_score("foo", k=1)
    assert len(query) == 1

    query = ob.similarity_search_with_score("foo", k=2)
    assert len(query) == 2

    query = ob.similarity_search_with_score("foo", k=3)
    assert len(query) == 3


def test_similarity_search_by_vector():
    ob = ObjectBox(embedding=FakeEmbeddings(), embedding_dimensions=10)
    texts = ["foo", "bar", "baz"]
    ob.add_texts(texts=texts)

    query_embedding = FakeEmbeddings().embed_query("foo")
    query = ob.similarity_search_by_vector(query_embedding, k=1)
    assert len(query) == 1

    query = ob.similarity_search_by_vector(query_embedding, k=2)
    assert len(query) == 2

    query = ob.similarity_search_by_vector(query_embedding, k=3)
    assert len(query) == 3
