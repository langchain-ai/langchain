import logging
import os
import shutil
import tempfile
from pathlib import PurePath
from typing import Union

import pytest

from langchain.vectorstores import Chroma
from tests.integration_tests.vectorstores_new.basic import BaseVectorStoreStaticTest

logger = logging.getLogger(__name__)

"""
pytest --capture=no --log-cli-level=DEBUG -vvv tests/integration_tests/vectorstores_new/test_chroma.py 
"""  # noqa: E501

# Define the Chroma vector store class to be used in the tests
vector_store_class = Chroma


class TestChromaVectorStoreLocalStaticMemory(BaseVectorStoreStaticTest):
    """
    Tests the Chroma vector store's static methods to ensure they work correctly with a
    local static vector store that does not persist data to disk.
    """

    vector_store_class = vector_store_class
    persist_directory: Union[PurePath, None] = None
    tmp_directory: Union[PurePath, None] = None

    @classmethod
    def setup_class(cls) -> None:
        assert cls.tmp_directory is None
        assert cls.persist_directory is None

        cls.tmp_directory = PurePath(tempfile.mkdtemp())
        cls.persist_directory = PurePath(
            os.path.join(cls.tmp_directory, tempfile.mkdtemp())
        )

        def change_persist_directory(_persist_directory: PurePath) -> None:
            """
            Override the persist directory of Chroma with a temporary directory
            to avoid creating files in the current directory by default.
            """
            import chromadb
            from _pytest.monkeypatch import MonkeyPatch
            from chromadb.config import Settings

            monkeypatch = MonkeyPatch()

            class CustomSettings(Settings):
                persist_directory = _persist_directory.name

            monkeypatch.setattr(chromadb.config, "Settings", CustomSettings)

        change_persist_directory(cls.persist_directory)

    @classmethod
    def teardown_class(cls) -> None:
        assert cls.tmp_directory is not None
        if os.path.exists(cls.tmp_directory.name):
            shutil.rmtree(cls.tmp_directory.name)

    def setup_method(self) -> None:
        assert self.persist_directory is not None
        if os.path.exists(self.persist_directory.name):
            shutil.rmtree(self.persist_directory.name)
        os.mkdir(self.persist_directory.name)

    def teardown_method(self) -> None:
        assert self.persist_directory is not None
        if os.path.exists(self.persist_directory.name):
            shutil.rmtree(self.persist_directory.name)


# TODO: Implement it
@pytest.mark.skip("not implemented it")
class TestChromaVectorStoreLocalStaticPersistent(BaseVectorStoreStaticTest):
    vector_store_class = vector_store_class

    @classmethod
    def setup_class(cls) -> None:
        pass

    @classmethod
    def teardown_class(cls) -> None:
        pass

    def setup_method(self) -> None:
        pass

    def teardown_method(self) -> None:
        pass


@pytest.mark.skip("need docker-compose file to tests it")
class TestChromaStaticTestRemote(BaseVectorStoreStaticTest):
    vector_store_class = vector_store_class

    @classmethod
    def setup_class(cls) -> None:
        pass

    @classmethod
    def teardown_class(cls) -> None:
        pass

    def setup_method(self) -> None:
        pass

    def teardown_method(self) -> None:
        pass
