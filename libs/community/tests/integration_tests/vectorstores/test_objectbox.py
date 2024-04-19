import os
from langchain_core.documents import Document

from langchain_community.vectorstores.objectbox import ObjectBox
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

# TODO: insert destroying the db creation in every test
def test_objectbox_db_initialisation():
    ob = ObjectBox(
        embedding_function=FakeEmbeddings(), embedding_dimensions=3
    )
    # TODO: test don't pass when I enter db_name to the constructor -> check
    folder_path = "data"

    assert os.path.exists(folder_path), f"Folder '{folder_path}' does not exist."

    filepath = os.path.join(folder_path, "data.mdb")
    assert os.path.isfile(filepath), f"File '{filename}' not found in '{folder_path}'"


def test_add_embeddings_objectbox():
    ob = ObjectBox(embedding_function=FakeEmbeddings(), embedding_dimensions=3)
    texts = ["foo", "bar", "baz"]
    ob.add_texts(texts=texts)
    # TODO: similarity isn't working as expected
    query = ob.similarity_search("foo", k=10)
    assert len(query) == 1


# def test_similarity_search():
#     ob = ObjectBox(embedding_function=FakeEmbeddings(), embedding_dimensions=3)
#     texts = ["foo", "bar", "baz"]
#     ob.add_texts(texts=texts)
#     output = ob.similarity_search("foo", k=1)
#     assert output == [Document(page_content="foo")]
#
#     # TODO: add another test for negative test
