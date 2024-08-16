"""Test OpenAI embeddings."""

import numpy as np
import openai
import pytest

from langchain_openai.embeddings.base import OpenAIEmbeddings

from langchain_standard_tests.integration_tests import EmbeddingsIntegrationTests
from typing import Type

class TestOpenAIEmbeddings(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[OpenAIEmbeddings]:
        return OpenAIEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "text-embedding-3-small"}


def test_langchain_openai_embeddings_dimensions() -> None:
    """Test openai embeddings."""
    documents = ["foo bar"]
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=128)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 128


@pytest.mark.skip(reason="flaky")
def test_langchain_openai_embeddings_equivalent_to_raw() -> None:
    documents = ["disallowed special token '<|endoftext|>'"]
    embedding = OpenAIEmbeddings()

    lc_output = embedding.embed_documents(documents)[0]
    direct_output = (
        openai.OpenAI()
        .embeddings.create(input=documents, model=embedding.model)
        .data[0]
        .embedding
    )
    assert np.isclose(lc_output, direct_output).all()


@pytest.mark.skip(reason="flaky")
async def test_langchain_openai_embeddings_equivalent_to_raw_async() -> None:
    documents = ["disallowed special token '<|endoftext|>'"]
    embedding = OpenAIEmbeddings()

    lc_output = (await embedding.aembed_documents(documents))[0]
    client = openai.AsyncOpenAI()
    direct_output = (
        (await client.embeddings.create(input=documents, model=embedding.model))
        .data[0]
        .embedding
    )
    assert np.isclose(lc_output, direct_output).all()
