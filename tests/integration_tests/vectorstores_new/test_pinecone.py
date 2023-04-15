import importlib
import os
from typing import Any

import pinecone
import pytest
from _pytest.monkeypatch import MonkeyPatch

from langchain.vectorstores.pinecone import Pinecone
from tests.integration_tests.vectorstores_new.base import (
    DEFAULT_COLLECTION_NAME,
    StaticTest,
)

DEFAULT_INDEX_NAME = "langchain-test-index"
dimension = 1536


def reset_pinecone() -> None:
    assert os.environ.get("PINECONE_API_KEY") is not None
    assert os.environ.get("PINECONE_ENVIRONMENT") is not None

    importlib.reload(pinecone)

    pinecone.init(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment=os.environ.get("PINECONE_ENVIRONMENT"),
    )


# Define the Pinecone vector store class to be used in the tests
vector_store_class = Pinecone


class TestPineconeStatic(StaticTest):
    vector_store_class = vector_store_class
    index: pinecone.Index

    monkeypatch: MonkeyPatch

    @classmethod
    def setup_class(cls) -> None:
        reset_pinecone()

        cls.index = pinecone.Index(DEFAULT_INDEX_NAME)

        if DEFAULT_INDEX_NAME in pinecone.list_indexes():
            index_stats = cls.index.describe_index_stats()
            if index_stats["dimension"] == dimension:
                # delete all the vectors in the index if the dimension is the same
                # from all namespaces
                index_stats = cls.index.describe_index_stats()
                for _namespace_name in index_stats["namespaces"].keys():
                    cls.index.delete(delete_all=True, namespace=_namespace_name)

            else:
                pinecone.delete_index(DEFAULT_INDEX_NAME)
                pinecone.create_index(name=DEFAULT_INDEX_NAME, dimension=dimension)
        else:
            pinecone.create_index(name=DEFAULT_INDEX_NAME, dimension=dimension)

        def patch_pinecone(monkeypatch: MonkeyPatch) -> None:
            def mock_from_texts(*args: Any, **kwargs: Any) -> Pinecone:
                kwargs["namespace"] = kwargs.pop(
                    "collection_name", DEFAULT_COLLECTION_NAME
                )
                kwargs["index_name"] = kwargs.pop("index_name", DEFAULT_INDEX_NAME)
                return original_from_texts(*args, **kwargs)

            original_from_texts = Pinecone.from_texts
            monkeypatch.setattr(Pinecone, "from_texts", mock_from_texts)

            def mock_from_documents(*args: Any, **kwargs: Any) -> Pinecone:
                kwargs["namespace"] = kwargs.pop(
                    "collection_name", DEFAULT_COLLECTION_NAME
                )
                kwargs["index_name"] = kwargs.pop("index_name", DEFAULT_INDEX_NAME)
                return original_from_documents(*args, **kwargs)

            original_from_documents = Pinecone.from_documents
            monkeypatch.setattr(Pinecone, "from_documents", mock_from_documents)

        cls.monkeypatch = MonkeyPatch()
        patch_pinecone(cls.monkeypatch)

    @classmethod
    def teardown_class(cls) -> None:
        # TODO: pinecone.delete_index(DEFAULT_INDEX_NAME)?

        # delete all the vectors in the index from all namespaces
        index_stats = cls.index.describe_index_stats()
        for _namespace_name in index_stats["namespaces"].keys():
            cls.index.delete(delete_all=True, namespace=_namespace_name)

        cls.monkeypatch.undo()
        reset_pinecone()

    def setup_method(self) -> None:
        # delete all the vectors in the index from all namespaces
        index_stats = self.index.describe_index_stats()
        for _namespace_name in index_stats["namespaces"].keys():
            self.index.delete(delete_all=True, namespace=_namespace_name)

    def teardown_method(self) -> None:
        pass

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Not implemented yet")
    async def test_from_texts_async(self, **args: Any) -> None:
        pass

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Not implemented yet")
    async def test_from_documents_async(self, **args: Any) -> None:
        pass
