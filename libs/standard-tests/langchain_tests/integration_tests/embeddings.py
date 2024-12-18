from typing import List

from langchain_core.embeddings import Embeddings

from langchain_tests.unit_tests.embeddings import EmbeddingsTests


class EmbeddingsIntegrationTests(EmbeddingsTests):
    """Base class for embeddings integration tests.

    Test subclasses must implement the ``embeddings_class`` property to specify the
    embeddings model to be tested. You can also override the
    ``embedding_model_params`` property to specify initialization parameters.

    Example:

    .. code-block:: python

        from typing import Type

        from langchain_tests.integration_tests import EmbeddingsIntegrationTests
        from my_package.embeddings import MyEmbeddingsModel


        class TestMyEmbeddingsModelIntegration(EmbeddingsIntegrationTests):
            @property
            def embeddings_class(self) -> Type[MyEmbeddingsModel]:
                # Return the embeddings model class to test here
                return MyEmbeddingsModel

            @property
            def embedding_model_params(self) -> dict:
                # Return initialization parameters for the model.
                return {"model": "model-001"}

    .. note::
          API references for individual test methods include troubleshooting tips.
    """

    def test_embed_query(self, model: Embeddings) -> None:
        """Test embedding a string query.

        .. dropdown:: Troubleshooting

            If this test fails, check that:

            1. The model will generate a list of floats when calling ``.embed_query`` on a string.
            2. The length of the list is consistent across different inputs.
        """  # noqa: E501
        embedding_1 = model.embed_query("foo")

        assert isinstance(embedding_1, List)
        assert isinstance(embedding_1[0], float)

        embedding_2 = model.embed_query("bar")

        assert len(embedding_1) > 0
        assert len(embedding_1) == len(embedding_2)

    def test_embed_documents(self, model: Embeddings) -> None:
        """Test embedding a list of strings.

        .. dropdown:: Troubleshooting

            If this test fails, check that:

            1. The model will generate a list of lists of floats when calling ``.embed_documents`` on a list of strings.
            2. The length of each list is the same.
        """  # noqa: E501
        documents = ["foo", "bar", "baz"]
        embeddings = model.embed_documents(documents)

        assert len(embeddings) == len(documents)
        assert all(isinstance(embedding, List) for embedding in embeddings)
        assert all(isinstance(embedding[0], float) for embedding in embeddings)
        assert len(embeddings[0]) > 0
        assert all(len(embedding) == len(embeddings[0]) for embedding in embeddings)

    async def test_aembed_query(self, model: Embeddings) -> None:
        """Test embedding a string query async.

        .. dropdown:: Troubleshooting

            If this test fails, check that:

            1. The model will generate a list of floats when calling ``.aembed_query`` on a string.
            2. The length of the list is consistent across different inputs.
        """  # noqa: E501
        embedding_1 = await model.aembed_query("foo")

        assert isinstance(embedding_1, List)
        assert isinstance(embedding_1[0], float)

        embedding_2 = await model.aembed_query("bar")

        assert len(embedding_1) > 0
        assert len(embedding_1) == len(embedding_2)

    async def test_aembed_documents(self, model: Embeddings) -> None:
        """Test embedding a list of strings async.

        .. dropdown:: Troubleshooting

            If this test fails, check that:

            1. The model will generate a list of lists of floats when calling ``.aembed_documents`` on a list of strings.
            2. The length of each list is the same.
        """  # noqa: E501
        documents = ["foo", "bar", "baz"]
        embeddings = await model.aembed_documents(documents)

        assert len(embeddings) == len(documents)
        assert all(isinstance(embedding, List) for embedding in embeddings)
        assert all(isinstance(embedding[0], float) for embedding in embeddings)
        assert len(embeddings[0]) > 0
        assert all(len(embedding) == len(embeddings[0]) for embedding in embeddings)
