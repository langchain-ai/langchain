from typing import Any

import pytest

from langchain.retrievers import DocArrayRetriever
from tests.integration_tests.retrievers.docarray.fixtures import (  # noqa: F401
    init_elastic,
    init_hnsw,
    init_in_memory,
    init_qdrant,
    init_weaviate,
)


@pytest.mark.parametrize(
    "backend",
    ["init_hnsw", "init_in_memory", "init_qdrant", "init_elastic", "init_weaviate"],
)
def test_backends(request: Any, backend: Any) -> None:
    index, filter_query, embeddings = request.getfixturevalue(backend)

    # create a retriever
    retriever = DocArrayRetriever(
        index=index,
        embeddings=embeddings,
        search_field="title_embedding",
        content_field="title",
    )

    docs = retriever.get_relevant_documents("my docs")

    assert len(docs) == 1
    assert "My document" in docs[0].page_content
    assert "id" in docs[0].metadata and "year" in docs[0].metadata
    assert "other_emb" not in docs[0].metadata

    # create a retriever with filters
    retriever = DocArrayRetriever(
        index=index,
        embeddings=embeddings,
        search_field="title_embedding",
        content_field="title",
        filters=filter_query,
    )

    docs = retriever.get_relevant_documents("my docs")

    assert len(docs) == 1
    assert "My document" in docs[0].page_content
    assert "id" in docs[0].metadata and "year" in docs[0].metadata
    assert "other_emb" not in docs[0].metadata
    assert docs[0].metadata["year"] <= 90

    # create a retriever with MMR search
    retriever = DocArrayRetriever(
        index=index,
        embeddings=embeddings,
        search_field="title_embedding",
        search_type="mmr",
        content_field="title",
        filters=filter_query,
    )

    docs = retriever.get_relevant_documents("my docs")

    assert len(docs) == 1
    assert "My document" in docs[0].page_content
    assert "id" in docs[0].metadata and "year" in docs[0].metadata
    assert "other_emb" not in docs[0].metadata
    assert docs[0].metadata["year"] <= 90
