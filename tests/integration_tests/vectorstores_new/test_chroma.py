import logging
from abc import ABC
from typing import Any

import chromadb
import pytest

from langchain.vectorstores import Chroma
from tests.integration_tests.vectorstores_new.base import (
    FilesystemTestInstance,
    FilesystemTestStatic,
)

logger = logging.getLogger(__name__)

"""
pytest --capture=no --log-cli-level=DEBUG -vvv tests/integration_tests/vectorstores_new/test_chroma.py 
"""  # noqa: E501

# Define the Chroma vector store class to be used in the tests
vector_store_class = Chroma

DEFAULT_COLLECTION_NAME = "langchain-test-collection"


class TestChromaFilesystemStatic(FilesystemTestStatic):
    """
    Tests the Chroma vector store's static methods to ensure they work correctly with a
    local static vector store that does not persist data to disk.
    """

    vector_store_class = vector_store_class

    @classmethod
    def setup_class(cls) -> None:
        super().setup_class()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Not implemented yet")
    async def test_from_texts_async(self, **args: Any) -> None:
        pass

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Not implemented yet")
    async def test_from_documents_async(self, **args: Any) -> None:
        pass


class TestChromaInstanceLocal(FilesystemTestInstance, ABC):
    """
    Tests the Chroma vector store's static methods to ensure they work correctly
    with a
    local static vector store that does not persist data to disk.
    """

    vector_store_class = vector_store_class

    @classmethod
    def setup_class(cls) -> None:
        super().setup_class()

    def setup_method(self) -> None:
        super().setup_method()
        assert self.embedding is not None

        client_settings = chromadb.config.Settings(anonymized_telemetry=False)

        self.vector_store = Chroma(
            embedding_function=self.embedding,
            client_settings=client_settings,
        )
