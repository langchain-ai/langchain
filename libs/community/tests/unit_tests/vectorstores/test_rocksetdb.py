import uuid
from unittest import mock

import pytest
import rockset

from langchain_community.vectorstores import Rockset
from rockset.model.add_documents_response import AddDocumentsResponse
from rockset.rockset_client import DocumentsApiWrapper
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
    fake_texts,
)

WORKSPACE = "langchain_tests"
COLLECTION_NAME = "langchain_demo"
EMBEDDING_KEY = "description_embedding"
TEXT_KEY = "description"


@pytest.fixture()
def embeddings():
    embeddings = ConsistentFakeEmbeddings()
    embeddings.embed_documents(fake_texts)
    return embeddings


@pytest.fixture()
def simple_rs_client():
    client = mock.Mock(rockset.RocksetClient)
    client.Documents = mock.Mock(DocumentsApiWrapper.add_documents)
    client.Documents.add_documents = mock.Mock(
        DocumentsApiWrapper.add_documents,
        return_value=AddDocumentsResponse(data=[])
    )
    return client


@pytest.fixture()
def vectorstore(simple_rs_client, embeddings):
    vectorstore = Rockset(
        simple_rs_client, embeddings, COLLECTION_NAME, TEXT_KEY, EMBEDDING_KEY, WORKSPACE
    )
    return vectorstore


def test_add_texts_does_not_modify_metadata(vectorstore: Rockset):
    """If metadata changes it will inhibit the langchain RecordManager
    functionality"""

    texts = ["kitty", "doggy"]
    metadatas = [{"source": "kitty.txt"}, {"source": "doggy.txt"}]
    ids = [uuid.uuid4(), uuid.uuid4()]

    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    for metadata in metadatas:
        assert len(metadata) == 1
        assert list(metadata.keys())[0] == "source"


def test_build_query_sql(vectorstore) -> None:
    vector = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    q_str = vectorstore._build_query_sql(
        vector,
        Rockset.DistanceFunction.COSINE_SIM,
        4,
    )
    vector_str = ",".join(map(str, vector))
    expected = f"""\
SELECT * EXCEPT({EMBEDDING_KEY}), \
COSINE_SIM({EMBEDDING_KEY}, [{vector_str}]) as dist
FROM {WORKSPACE}.{COLLECTION_NAME}
ORDER BY dist DESC
LIMIT 4
"""
    assert q_str == expected


def test_build_query_sql_with_where(vectorstore) -> None:
    vector = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    q_str = vectorstore._build_query_sql(
        vector,
        Rockset.DistanceFunction.COSINE_SIM,
        4,
        "age >= 10",
    )
    vector_str = ",".join(map(str, vector))
    expected = f"""\
SELECT * EXCEPT({EMBEDDING_KEY}), \
COSINE_SIM({EMBEDDING_KEY}, [{vector_str}]) as dist
FROM {WORKSPACE}.{COLLECTION_NAME}
WHERE age >= 10
ORDER BY dist DESC
LIMIT 4
"""
    assert q_str == expected
