import logging
import os
from abc import ABC
from pathlib import PurePath
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


def change_default_db_directory(_persist_directory: PurePath) -> None:
    """
    Override the persist directory of Chroma with a temporary directory
    to avoid creating files in the current directory by default.
    """
    from _pytest.monkeypatch import MonkeyPatch
    from chromadb.config import Settings

    monkeypatch = MonkeyPatch()

    class CustomSettings(Settings):
        persist_directory = _persist_directory.__str__()

    monkeypatch.setattr(chromadb.config, "Settings", CustomSettings)


class TestChromaFilesystemStatic(FilesystemTestStatic):
    """
    Tests the Chroma vector store's static methods to ensure they work correctly with a
    local static vector store that does not persist data to disk.
    """

    vector_store_class = vector_store_class

    @classmethod
    def setup_class(cls) -> None:
        super().setup_class()

        assert cls.tmp_directory is not None
        assert cls.db_dir is not None

        assert os.path.exists(cls.tmp_directory.__str__())
        assert os.path.exists(cls.db_dir.__str__())

        change_default_db_directory(cls.db_dir)

    @pytest.mark.xfail(reason="duplicate documents are not handled correctly")
    async def test_from_texts_with_ids(self, *args: Any, **kwargs: Any) -> None:
        """
        FIXME:
        This test is failing because duplicate documents are not handled correctly
        """

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

        assert cls.tmp_directory is not None
        assert cls.db_dir is not None

        assert os.path.exists(cls.tmp_directory.__str__())
        assert os.path.exists(cls.db_dir.__str__())

        change_default_db_directory(cls.db_dir)

    def setup_method(self) -> None:
        super().setup_method()
        assert self.embedding is not None

        client_settings = chromadb.config.Settings(anonymized_telemetry=False)

        self.vector_store = Chroma(
            embedding_function=self.embedding,
            client_settings=client_settings,
            persist_directory=self.db_dir.__str__(),
            collection_name=self.collection_name,
        )
        """
        self.vector_store:
        'aadd_documents',
        aadd_texts',
        add_documents',
        add_texts',
        afrom_documents',
        afrom_texts',
        amax_marginal_relevance_search',
        amax_marginal_relevance_search_by_vector',
        as_retriever',
        asimilarity_search',
        asimilarity_search_by_vector',
        delete_collection',
        from_documents',
        from_texts',
        max_marginal_relevance_search',
        max_marginal_relevance_search_by_vector',
        persist',
        similarity_search',
        similarity_search_by_vector',
        'similarity_search_with_score'
         """
