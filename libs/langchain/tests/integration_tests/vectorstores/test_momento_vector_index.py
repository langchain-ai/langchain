import time
import uuid
from typing import Iterator, List

import pytest

from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MomentoVectorIndex

API_KEY_ENV_VAR = "MOMENTO_API_KEY"


def random_string() -> str:
    return str(uuid.uuid4())


@pytest.fixture(scope="function")
def random_index_name() -> str:
    return f"langchain-test-index-{random_string()}"


def wait() -> None:
    time.sleep(1)


@pytest.fixture(scope="function")
def vector_store(
    embedding_openai: OpenAIEmbeddings, random_index_name: str
) -> Iterator[MomentoVectorIndex]:
    from momento import (
        CredentialProvider,
        PreviewVectorIndexClient,
        VectorIndexConfigurations,
    )

    vector_store = None
    try:
        client = PreviewVectorIndexClient(
            VectorIndexConfigurations.Default.latest(),
            credential_provider=CredentialProvider.from_environment_variable(
                API_KEY_ENV_VAR
            ),
        )
        vector_store = MomentoVectorIndex(
            embedding=embedding_openai,
            client=client,
            index_name=random_index_name,
        )
        yield vector_store
    finally:
        if vector_store is not None:
            vector_store._client.delete_index(random_index_name)


def test_from_texts(
    random_index_name: str, embedding_openai: OpenAIEmbeddings, texts: List[str]
) -> None:
    from momento import (
        CredentialProvider,
        VectorIndexConfigurations,
    )

    random_text = random_string()
    random_document = f"Hello world {random_text} goodbye world!"
    texts.insert(0, random_document)

    vector_store = None
    try:
        vector_store = MomentoVectorIndex.from_texts(
            texts=texts,
            embedding=embedding_openai,
            index_name=random_index_name,
            configuration=VectorIndexConfigurations.Default.latest(),
            credential_provider=CredentialProvider.from_environment_variable(
                "MOMENTO_API_KEY"
            ),
        )
        wait()

        documents = vector_store.similarity_search(query=random_text, k=1)
        assert documents == [Document(page_content=random_document)]
    finally:
        if vector_store is not None:
            vector_store._client.delete_index(random_index_name)


def test_from_texts_with_metadatas(
    random_index_name: str, embedding_openai: OpenAIEmbeddings, texts: List[str]
) -> None:
    """Test end to end construction and search."""
    from momento import (
        CredentialProvider,
        VectorIndexConfigurations,
    )

    random_text = random_string()
    random_document = f"Hello world {random_text} goodbye world!"
    texts.insert(0, random_document)
    metadatas = [{"page": f"{i}", "source": "user"} for i in range(len(texts))]

    vector_store = None
    try:
        vector_store = MomentoVectorIndex.from_texts(
            texts=texts,
            embedding=embedding_openai,
            index_name=random_index_name,
            metadatas=metadatas,
            configuration=VectorIndexConfigurations.Default.latest(),
            credential_provider=CredentialProvider.from_environment_variable(
                API_KEY_ENV_VAR
            ),
        )

        wait()
        documents = vector_store.similarity_search(query=random_text, k=1)
        assert documents == [
            Document(
                page_content=random_document, metadata={"page": "0", "source": "user"}
            )
        ]
    finally:
        if vector_store is not None:
            vector_store._client.delete_index(random_index_name)


def test_from_texts_with_scores(vector_store: MomentoVectorIndex) -> None:
    #    """Test end to end construction and search with scores and IDs."""
    texts = ["apple", "orange", "hammer"]
    metadatas = [{"page": f"{i}"} for i in range(len(texts))]

    vector_store.add_texts(texts, metadatas)
    wait()
    search_results = vector_store.similarity_search_with_score("apple", k=3)
    docs = [o[0] for o in search_results]
    scores = [o[1] for o in search_results]

    assert docs == [
        Document(page_content="apple", metadata={"page": "0"}),
        Document(page_content="orange", metadata={"page": "1"}),
        Document(page_content="hammer", metadata={"page": "2"}),
    ]
    assert scores[0] > scores[1] > scores[2]


def test_add_documents_with_ids(vector_store: MomentoVectorIndex) -> None:
    """Test end to end construction and search with scores and IDs."""
    from momento.responses.vector_index import Search

    texts = ["apple", "orange", "hammer"]
    ids = [random_string() for _ in range(len(texts))]
    metadatas = [{"page": f"{i}"} for i in range(len(texts))]

    # Add texts with metadata and ids
    stored_ids = vector_store.add_texts(texts, metadatas, ids=ids)
    assert stored_ids == ids
    wait()

    # Verify that the ids are in the index
    response = vector_store._client.search(
        vector_store.index_name, vector_store.embeddings.embed_query("apple")
    )
    assert isinstance(response, Search.Success)
    assert [hit.id for hit in response.hits] == ids
