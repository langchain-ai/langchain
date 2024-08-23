from typing import List

from langchain_core.embeddings import Embeddings

from langchain_standard_tests.unit_tests.embeddings import EmbeddingsTests


class EmbeddingsIntegrationTests(EmbeddingsTests):
    def test_embed_query(self, model: Embeddings) -> None:
        embedding_1 = model.embed_query("foo")

        assert isinstance(embedding_1, List)
        assert isinstance(embedding_1[0], float)

        embedding_2 = model.embed_query("bar")

        assert len(embedding_1) > 0
        assert len(embedding_1) == len(embedding_2)

    def test_embed_documents(self, model: Embeddings) -> None:
        documents = ["foo", "bar", "baz"]
        embeddings = model.embed_documents(documents)

        assert len(embeddings) == len(documents)
        assert all(isinstance(embedding, List) for embedding in embeddings)
        assert all(isinstance(embedding[0], float) for embedding in embeddings)
        assert len(embeddings[0]) > 0
        assert all(len(embedding) == len(embeddings[0]) for embedding in embeddings)

    async def test_aembed_query(self, model: Embeddings) -> None:
        embedding_1 = await model.aembed_query("foo")

        assert isinstance(embedding_1, List)
        assert isinstance(embedding_1[0], float)

        embedding_2 = await model.aembed_query("bar")

        assert len(embedding_1) > 0
        assert len(embedding_1) == len(embedding_2)

    async def test_aembed_documents(self, model: Embeddings) -> None:
        documents = ["foo", "bar", "baz"]
        embeddings = await model.aembed_documents(documents)

        assert len(embeddings) == len(documents)
        assert all(isinstance(embedding, List) for embedding in embeddings)
        assert all(isinstance(embedding[0], float) for embedding in embeddings)
        assert len(embeddings[0]) > 0
        assert all(len(embedding) == len(embeddings[0]) for embedding in embeddings)
