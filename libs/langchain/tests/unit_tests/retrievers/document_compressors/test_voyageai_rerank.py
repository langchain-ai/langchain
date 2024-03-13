import pytest

from langchain.retrievers.document_compressors import VoyageAIRerank


@pytest.mark.requires("voyageai")
def test_init() -> None:
    VoyageAIRerank(
        voyageai_api_key="foo",
        model="rerank-lite-1",
    )
